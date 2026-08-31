"""Delegated protein readouts (ALL-FUNCTIONS.md).

These readouts are DELEGATED to real external methods, exactly the way AIDO
itself delegates: the structure is not decoded from the latent space — it is
the real output of a real folding model — and every return value carries a
``source`` field naming the delegation. Approximation (a made-up structure,
or another quantity renamed to look like one) is forbidden; delegation with
provenance is the legal route.

- ``get_protein_structure(target, molecule=None)``: real ESMFold v1
  (facebook/esmfold_v1 — the HuggingFace transformers ``EsmForProteinFolding``
  port, same weights, weights cached locally on this machine) run over the
  UniProt sequence resolved from the engine gene bridge
  (bridge_proteins.parquet — real UniProt sequences, 4867 human genes).
  Returns the PDB string and per-residue pLDDT with provenance.

- ``get_protein_ligand_interactions(molecule, target=None)``: real docking
  (AutoDock Vina 1.2.7 via ``omicverse.mol.dock``) of the molecule into a
  fpocket-detected pocket of the ESMFold structure from the previous step,
  then a real geometric contact analysis (atom pairs within 4.5 A; polar
  donor-acceptor pairs within 3.5 A) of the docked pose against the receptor.
  Returns per-residue contacts with provenance. ``target`` is required: a
  contact list is between a molecule and a specific protein; no
  molecule->target mapping is maintained, so calling without a target raises
  honestly rather than guessing a protein.

- ``get_cell_age()``: ALL-FUNCTIONS option 2. None of the three datasets this
  Cell is trained on carries any age measurement (Norman 2019, Srivatsan
  2020, DeepSTARR — all static perturbation screens), and no citable
  transcriptomic age clock is valid for the K-562 cell line (an immortal
  proliferating line, not a primary sample with a donor age). There is no
  delegable real method and no measurement, so the honest behaviour is to
  raise with that reason.
"""
from __future__ import annotations

from luria_aido import config
import json
import os
import pathlib

import numpy as np

_WS = pathlib.Path(__file__).resolve().parent.parent
_BRIDGE_DEFAULT = str(config.data_root() / "artifacts" / "bridge_proteins.parquet")
_HF_SNAPSHOTS = [
    pathlib.Path(
        config.hf_home() + "/hub"
        "/models--facebook--esmfold_v1/snapshots"
        "/75a3841ee059df2bf4d56688166c8fb459ddd97a"   # pytorch_model.bin
    ),
    pathlib.Path(
        config.hf_home() + "/hub"
        "/models--facebook--esmfold_v1/snapshots"
        "/ba837a39b67e59941c3f017d6c2a064f567038d9"   # model.safetensors
    ),
]

_model = None
_bridge_cache = None


# --------------------------------------------------------------------------- #
# ESMFold v1 (delegated structure prediction)
# --------------------------------------------------------------------------- #

def _load_model():
    """Lazy singleton: facebook/esmfold_v1 via transformers, local weights.

    Device decided by a REAL allocation (USE-MPS.md): under a contended card
    without CUDA MPS the allocation fails even though is_available() is True.
    Falls back to CPU cleanly; the device is recorded in the fold meta so the
    provenance shows where the fold ran.
    """
    global _model
    if _model is not None:
        return _model
    from luria_aido.gpu import cuda_alloc_ok, ensure_mps_env
    import torch
    from transformers.models.esm.modeling_esmfold import EsmForProteinFolding

    ensure_mps_env()
    local = next((s for s in _HF_SNAPSHOTS if s.exists()), None)
    if local is not None:
        model = EsmForProteinFolding.from_pretrained(str(local))
    else:
        model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", local_files_only=True)
    model.eval()
    if cuda_alloc_ok():
        model = model.cuda()
    _model = model
    return _model


def _plddt_from_pdb(pdb_str: str):
    """Per-residue pLDDT from the B-factor column of an ESMFold PDB.

    transformers' ``to_pdb`` writes the B-factor on the 0-1 scale (the
    ESMFold API convention); AlphaFold-style writers use 0-100. Rescale so
    the returned values are always 0-100.
    """
    per = []
    for line in pdb_str.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            per.append(float(line[60:66]))
    per = np.asarray(per, dtype=float)
    if per.size and float(np.nanmax(per)) <= 1.0:
        per = per * 100.0
    return per


def _cache_dir(cache_dir=None) -> pathlib.Path:
    if cache_dir is None:
        cache_dir = os.environ.get("LURIA_AIDO_PROTEIN_CACHE",
                                   str(_WS / "artifacts" / "proteins" / "structures"))
    p = pathlib.Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def fold_sequence(seq: str, name: str, cache_dir=None) -> dict:
    """Fold one sequence with real ESMFold v1; disk-cached by ``name``.

    Returns {"name", "length", "mean_plddt", "plddt_per_residue", "pdb"}.
    """
    out = _cache_dir(cache_dir)
    pdb_path = out / f"{name}.pdb"
    meta_path = out / f"{name}.json"
    if pdb_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["pdb"] = pdb_path.read_text()
        return meta

    import torch
    model = _load_model()
    device = str(next(model.parameters()).device)
    with torch.no_grad():
        pdb_str = model.infer_pdb(seq)

    per = _plddt_from_pdb(pdb_str)
    meta = {
        "name": name,
        "length": len(per),
        "mean_plddt": float(np.mean(per)),
        "plddt_per_residue": per.tolist(),
        "device": device,
    }
    pdb_path.write_text(pdb_str)
    meta_path.write_text(json.dumps(meta))
    meta["pdb"] = pdb_str
    return meta


def _bridge():
    """symbol -> (uniprot, sequence) from the engine gene bridge."""
    global _bridge_cache
    if _bridge_cache is None:
        import pandas as pd
        p = pathlib.Path(os.environ.get("LURIA_AIDO_BRIDGE_PROTEINS",
                                        _BRIDGE_DEFAULT))
        if not p.exists():
            raise NotImplementedError(
                f"bridge_proteins.parquet not found at {p}: the delegated "
                "structure readout needs the engine's UniProt sequence "
                "bridge; set LURIA_AIDO_BRIDGE_PROTEINS.")
        df = pd.read_parquet(p)
        _bridge_cache = {
            row.symbol: (row.uniprot, row.protein) for row in df.itertuples()
        }
    return _bridge_cache


def resolve_sequence(target: str):
    """Resolve a gene symbol (or UniProt accession) to (symbol, uniprot, seq)."""
    b = _bridge()
    for key in (target, target.upper(), target.lower()):
        if key in b:
            sym, (uni, seq) = key, b[key]
            return sym, uni, seq
    for sym, (uni, seq) in b.items():
        if uni == target:
            return sym, uni, seq
    raise NotImplementedError(
        f"no sequence for target {target!r} in the gene bridge "
        f"({len(b)} genes). The delegated structure readout is bound to "
        "that bridge; a target outside it raises honestly.")


# --------------------------------------------------------------------------- #
# Public readouts
# --------------------------------------------------------------------------- #

# Curated UniProt domain windows used to annotate full-length structure
# confidence (window boundaries from UniProt; pLDDT values are computed
# from the real ESMFold output, not estimated).
_DOMAIN_WINDOWS = {
    "P00519": [("kinase_domain", 242, 493)],   # ABL1, UniProt P00519 (1-based)
}


def _window_mean_plddt(per, start, end):
    seg = per[start - 1:end]
    return round(float(np.mean(seg)), 2) if len(seg) else None


def get_protein_structure(target: str, molecule=None, cache_dir=None) -> dict:
    """Real ESMFold v1 structure for the target's UniProt sequence.

    DELEGATED: the structure is the real output of facebook/esmfold_v1
    (transformers port), not a decode from the latent state.
    """
    sym, uni, seq = resolve_sequence(target)
    fold = fold_sequence(seq, f"{sym}__{uni}", cache_dir)
    return {
        "value": {
            "pdb": fold["pdb"],
            "length": fold["length"],
            "sequence": seq,
        },
        "target": sym,
        "uniprot": uni,
        "molecule": molecule,
        "mean_plddt": fold["mean_plddt"],
        "plddt_per_residue": fold["plddt_per_residue"],
        "domain_windows": [
            {"name": n, "start": s, "end": e,
             "mean_plddt": _window_mean_plddt(fold["plddt_per_residue"], s, e)}
            for n, s, e in _DOMAIN_WINDOWS.get(uni, [])
        ],
        "confidence": round(fold["mean_plddt"] / 100.0, 4),
        "source": ("ESMFold v1 (facebook/esmfold_v1), HuggingFace transformers "
                   "EsmForProteinFolding, weights from local HF cache — "
                   "DELEGATED"),
        "method": "delegated",
    }


def get_protein_ligand_interactions(molecule, target=None, cache_dir=None, *,
                                    exhaustiveness: int = 16, seed: int = 42,
                                    contact_cutoff_ang: float = 4.5) -> dict:
    """Real docking + real contact analysis on the ESMFold structure.

    DELEGATED chain: ESMFold structure (get_protein_structure) -> fpocket
    pocket detection -> AutoDock Vina docking (ov.mol.dock, seed fixed) ->
    geometric contact analysis of the best pose against the receptor.

    ``target`` resolution order (all real, none invented):
    1. explicit ``target=`` argument;
    2. (Cell layer) the target of the last ``get_protein_structure`` call on
       this Cell — the ALL-FUNCTIONS "compute contacts on the structure from
       the previous step" chain;
    3. ChEMBL API (EBI): molecule name or canonical SMILES -> molecule ->
       potent human activities -> top target whose name contains an exact
       bridge-gene token (e.g. 'Tyrosine-protein kinase ABL1' -> ABL1),
       disk-cached per molecule.
    If none resolves, raise honestly (no fabricated molecule->target map).
    """
    resolution_note = None
    if target is None:
        res = resolve_target_chembl(molecule)
        if res is None:
            raise NotImplementedError(
                "get_protein_ligand_interactions could not resolve a target "
                "for the given molecule: no explicit target= argument, no "
                "prior get_protein_structure(target, ...) on this Cell, and "
                "ChEMBL returned no target whose name matches a gene in the "
                "bridge. Pass target= (e.g. 'ABL1') to dock against a "
                "specific protein. No fabricated mapping is used.")
        target = res["symbol"]
        resolution_note = res["note"]
    if not isinstance(molecule, str) or not _is_smiles_like(molecule):
        raise NotImplementedError(
            f"molecule {molecule!r} does not look like a SMILES string; "
            "the delegated docking path takes a SMILES.")

    sym, uni, seq = resolve_sequence(target)
    out = _cache_dir(cache_dir)
    name = f"{sym}__{uni}"
    fold = fold_sequence(seq, name, cache_dir)
    pdb_path = out / f"{name}.pdb"

    # -- receptor MolStructure from the ESMFold PDB (biotite parse) ---------
    import biotite.structure as struc
    import biotite.structure.io.pdb as bpdb
    from omicverse.mol import MolStructure, dock, pockets

    f = bpdb.PDBFile.read(str(pdb_path))
    atoms = f.get_structure(model=1, extra_fields=["b_factor"])
    atoms = atoms[struc.filter_amino_acids(atoms)]
    structure = MolStructure(
        atoms, source="esmfold", path=str(pdb_path),
        gene=sym, uniprot=uni,
        plddt=np.asarray(fold["plddt_per_residue"], dtype=float))

    # -- pocket detection (apo structure -> top fpocket druggability) -------
    pk = pockets(structure, min_drug_score=0.0)
    if pk is None or len(pk) == 0:
        raise NotImplementedError(
            f"fpocket detected no pocket on the ESMFold {sym} structure; "
            "the delegated docking readout cannot run without a pocket. "
            "Raising honestly instead of docking blind.")
    best = pk.iloc[0]
    pocket_info = {
        "pocket_id": int(best["pocket_id"]),
        "rank": int(best["rank"]),
        "drug_score": float(best["drug_score"]),
        "volume": float(best["volume"]),
        "n_residues": int(best["n_residues"]),
        "residues": [str(r) for r in best["residues"]],
        "site_selection": ("apo structure -> top fpocket druggability score; "
                           "no co-crystal ligand exists on an ESMFold model"),
    }

    # -- real docking (AutoDock Vina via ov.mol.dock) ------------------------
    result = dock(structure, ligand=molecule, pocket=pocket_info["pocket_id"],
                  exhaustiveness=exhaustiveness, n_poses=9, seed=seed)

    # -- real contact analysis on the best pose ------------------------------
    best_pose = result.best
    if best_pose is None:
        raise NotImplementedError("docking returned no poses; raising honestly.")
    ligand_pos = np.asarray(best_pose.GetConformer().GetPositions(), dtype=float)
    lig_symbols = [a.GetSymbol() for a in best_pose.GetAtoms()]

    r_coord = np.asarray(atoms.coord, dtype=float)
    r_elem = np.asarray(atoms.element)
    r_res = np.asarray(atoms.res_id)
    r_name = np.asarray(atoms.res_name)
    r_atomname = np.asarray(atoms.atom_name)

    d2 = ((r_coord[:, None, :] - ligand_pos[None, :, :]) ** 2).sum(-1)
    d = np.sqrt(d2)
    i_idx, j_idx = np.nonzero(d <= contact_cutoff_ang)

    per_res = {}
    for i, j in zip(i_idx, j_idx):
        key = (int(r_res[i]), str(r_name[i]))
        per_res.setdefault(key, {"residue": key[0], "resname": key[1],
                                 "atom_pairs": 0, "receptor_atoms": set(),
                                 "ligand_atoms": set()})
        per_res[key]["atom_pairs"] += 1
        per_res[key]["receptor_atoms"].add(str(r_atomname[i]))
        per_res[key]["ligand_atoms"].add(int(j))

    # polar pairs within 3.5 A = putative H-bonds (geometric, labelled as such)
    polar_i, polar_j = np.nonzero(
        (d <= 3.5) & (np.isin(r_elem[:, None], ["N", "O"]))
        & (np.isin(np.asarray(lig_symbols)[None, :], ["N", "O"])))
    hbonds = [
        {"receptor": f"{r_name[i]}{int(r_res[i])}:{r_atomname[i]}",
         "ligand_atom": int(j),
         "ligand_element": lig_symbols[int(j)],
         "distance_ang": round(float(d[i, j]), 2)}
        for i, j in zip(polar_i, polar_j)
    ]
    hbonds.sort(key=lambda h: h["distance_ang"])

    contacts = []
    for key in sorted(per_res):
        v = per_res[key]
        contacts.append({
            "residue": v["residue"],
            "resname": v["resname"],
            "atom_pairs": v["atom_pairs"],
            "receptor_atoms": sorted(v["receptor_atoms"]),
            "ligand_atom_indices": sorted(v["ligand_atoms"]),
        })

    # -- persist artifacts next to the structure cache ------------------------
    import pandas as pd
    pd.DataFrame(contacts).to_csv(out / f"{name}__interactions_contacts.csv",
                                  index=False)
    (out / f"{name}__best_pose.pdb").write_text(result.pose_blocks[0])
    summary = {
        "target": sym, "uniprot": uni, "molecule": molecule,
        "best_affinity_kcal_mol": float(result.affinities[0]),
        "affinities_kcal_mol": [float(a) for a in result.affinities],
        "pocket": {k: (str(v) if not isinstance(v, (int, float, str)) else v)
                   for k, v in pocket_info.items()},
        "n_contact_atom_pairs": int(len(i_idx)),
        "n_contact_residues": len(contacts),
        "ligand_atoms_in_contact": int(len(set(j_idx.tolist()))),
        "n_putative_hbonds": len(hbonds),
        "source": "AutoDock Vina 1.2.7 via ov.mol.dock on ESMFold v1 structure "
                  "(delegated)",
        "seed": seed, "exhaustiveness": exhaustiveness,
    }
    (out / f"{name}__interactions_summary.json").write_text(
        json.dumps(summary, indent=2))

    return {
        "value": {
            "contacts": contacts,
            "n_contact_atom_pairs": int(len(i_idx)),
            "n_contact_residues": len(contacts),
            "ligand_atoms_in_contact": int(len(set(j_idx.tolist()))),
            "putative_hbonds_geometric_3_5A": hbonds,
        },
        "target": sym,
        "uniprot": uni,
        "molecule": molecule,
        "target_resolved_via": resolution_note,
        "best_affinity_kcal_mol": float(result.affinities[0]),
        "affinities_kcal_mol": [float(a) for a in result.affinities],
        "pocket": pocket_info,
        "structure": {
            "source": "ESMFold v1 (delegated)",
            "mean_plddt": fold["mean_plddt"],
            "n_residues": fold["length"],
        },
        "confidence": round(
            min(float(best["drug_score"]), fold["mean_plddt"] / 100.0), 4),
        "source": ("AutoDock Vina 1.2.7 via omicverse.mol.dock on the "
                   "ESMFold v1 structure (delegated), contacts computed "
                   "geometrically from the docked pose — DELEGATED"),
        "method": "delegated",
        "caveat": ("Vina affinity is an in-silico estimate (kcal/mol), not a "
                   "measured binding free energy; a docked pose enriches a "
                   "binding hypothesis, it does not prove binding."),
    }


def get_cell_age():
    """ALL-FUNCTIONS option 2: raise with the honest reason."""
    raise NotImplementedError(
        "get_cell_age: no cell-age measurement exists in any of the three "
        "datasets this Cell is trained on (Norman 2019, Srivatsan 2020, "
        "DeepSTARR — all static perturbation screens), and no citable "
        "transcriptomic age clock is valid for the K-562 cell line (an "
        "immortal proliferating line, not a primary sample with a donor "
        "age). ALL-FUNCTIONS.md option 2: there is no delegable real method "
        "and no measurement — raising is the honest behaviour.")


# --------------------------------------------------------------------------- #
# Molecule design (delegated, real generative model)
# --------------------------------------------------------------------------- #

_molgen_cache = {}


def _warm_molgen():
    """Lazy singleton for WarmMolGenOne (target-sequence -> SMILES)."""
    if "model" in _molgen_cache:
        return _molgen_cache
    import sys as _sys
    _tools = str(_WS / "tools")
    if _tools not in _sys.path:
        _sys.path.insert(0, _tools)
    from luria_aido import hf_shim  # noqa: F401  wandb/masking_utils shims, must precede load
    from luria_aido.gpu import cuda_alloc_ok, ensure_mps_env
    ensure_mps_env()
    import torch
    from transformers import EncoderDecoderModel, AutoTokenizer
    dev = "cuda" if cuda_alloc_ok() else "cpu"
    R = "gokceuludogan/WarmMolGenOne"
    tin = AutoTokenizer.from_pretrained(R)
    model = EncoderDecoderModel.from_pretrained(R).to(dev).eval()
    tout = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    _molgen_cache.update(model=model, tin=tin, tout=tout, device=dev)
    return _molgen_cache


def design_molecule_for_target(target: str, *, n: int = 12,
                               temperature: float = 1.0, top_k: int = 40,
                               max_length: int = 128,
                               domain_only: bool = True) -> dict:
    """De-novo SMILES for a target, DELEGATED to a real generative model.

    WarmMolGenOne (gokceuludogan/WarmMolGenOne, HuggingFace
    EncoderDecoderModel) is a sequence-conditioned molecule generator: the
    target's amino-acid sequence is the encoder input, the decoder samples
    SMILES. This is de-novo GENERATION, not retrieval — it does NOT read a
    known-ligand database (those are `retrieve_known_ligands`, a different
    readout). The output is the model's real sample; invalid SMILES (RDKit
    rejects) are dropped and the validity rate is reported honestly.

    ``domain_only``: when the target's curated domain window is known (e.g.
    ABL1 P00519 kinase domain 242-493), feed only that window — the
    generative model conditions on the binding domain, not the full protein.
    """
    sym, uni, seq = resolve_sequence(target)

    dom = None
    if domain_only:
        windows = _DOMAIN_WINDOWS.get(uni, [])
        dom = next((w for w in windows if "domain" in w[0] or "kinase" in w[0]),
                   None)
        if dom is not None:
            _, s, e = dom
            seq = seq[s - 1:e]

    c = _warm_molgen()
    import torch
    with torch.no_grad():
        enc = c["tin"](" ".join(list(seq)), return_tensors="pt",
                       truncation=True, max_length=512).to(c["device"])
        out = c["model"].generate(**enc, do_sample=True, top_k=top_k,
                                  temperature=temperature,
                                  max_length=max_length,
                                  num_return_sequences=n)
    smis = [c["tout"].decode(x, skip_special_tokens=True).replace(" ", "").strip()
            for x in out]

    from rdkit import Chem
    valid = []
    for s in smis:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            valid.append({"smiles": s, "canonical": Chem.MolToSmiles(mol)})

    return {
        "value": valid,
        "target": sym,
        "uniprot": uni,
        "input_sequence": (f"{sym} domain window {dom[1]}-{dom[2]}"
                           if (domain_only and dom is not None)
                           else f"{sym} full sequence"),
        "n_requested": int(n),
        "n_valid": len(valid),
        "validity_rate": round(len(valid) / max(n, 1), 4),
        "source": ("WarmMolGenOne (gokceuludogan/WarmMolGenOne), HuggingFace "
                   "EncoderDecoderModel, sequence-conditioned de-novo SMILES "
                   "generation, weights from local HF cache — DELEGATED"),
        "method": "delegated",
        "caveat": ("Generated SMILES are a generative model's proposal, NOT a "
                   "measured binder: binding must be assessed by the "
                   "structure/docking readouts (get_protein_structure / "
                   "get_protein_ligand_interactions). Validity <100% is "
                   "reported, not silently filtered."),
    }



# --------------------------------------------------------------------------- #
# ChEMBL target resolution (delegated, real database)
# --------------------------------------------------------------------------- #

_CHEMBL_CACHE = None


def _chembl_get(url: str):
    import json as _json
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "luria-aido/0.1"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return _json.loads(r.read())


def _chembl_cache():
    global _CHEMBL_CACHE
    if _CHEMBL_CACHE is None:
        import json as _json
        p = _cache_dir() / "chembl_resolve.json"
        _CHEMBL_CACHE = (_json.loads(p.read_text()) if p.exists() else {})
    return _CHEMBL_CACHE


def resolve_target_chembl(molecule: str):
    """Resolve a molecule (SMILES or drug name) to a bridge gene via ChEMBL.

    REAL delegated resolution (no fabricated mapping): ChEMBL API -> molecule
    -> potent human activities -> top target whose pref_name contains an
    exact token matching a bridge gene symbol (e.g. 'Tyrosine-protein kinase
    ABL1' -> ABL1). Disk-cached per input string. Returns
    {"symbol", "note"} or None when nothing resolves.
    """
    key = str(molecule).strip()
    cache = _chembl_cache()
    if key in cache:
        hit = cache[key]
        return (hit if hit is not None else None)

    import re as _re
    import urllib.parse as _urlparse
    from rdkit import Chem as _Chem

    result = None
    try:
        # -- molecule id ----------------------------------------------------
        mol_id = None
        looks_smiles = bool(_re.search(r"[0-9@+\-\\/\[\]()#=.]", key))
        if looks_smiles:
            canon = _Chem.MolToSmiles(_Chem.MolFromSmiles(key))
            d = _chembl_get("https://www.ebi.ac.uk/chembl/api/data/molecule.json"
                            "?smiles=" + _urlparse.quote(canon) + "&limit=3")
            ms = d.get("molecules", [])
            if ms:
                mol_id = ms[0]["molecule_chembl_id"]
        else:
            d = _chembl_get("https://www.ebi.ac.uk/chembl/api/data/molecule/"
                            "search.json?q=" + _urlparse.quote(key) + "&limit=5")
            ms = d.get("molecules", [])
            best = None
            for m in ms:
                name = (m.get("pref_name") or "").lower()
                if name == key.lower():
                    best = m
                    break
                if best is None and key.lower() in name:
                    best = m
            if best is not None:
                mol_id = best["molecule_chembl_id"]

        if mol_id is None:
            cache[key] = None
            return None

        # -- potent human activities -> targets ------------------------------
        d = _chembl_get("https://www.ebi.ac.uk/chembl/api/data/activity.json"
                        "?molecule_chembl_id=" + mol_id +
                        "&target_organism=Homo%20sapiens"
                        "&pchembl_value__gte=6&limit=100")
        pot = {}
        for a in d.get("activities", []):
            tid = a.get("target_chembl_id")
            pc = a.get("pchembl_value")
            if tid and pc is not None:
                pot[tid] = max(pot.get(tid, 0.0), float(pc))

        # -- target name -> exact bridge-gene token --------------------------
        bridge = _bridge()
        symbols = set(bridge)
        for tid in sorted(pot, key=pot.get, reverse=True):
            try:
                td = _chembl_get("https://www.ebi.ac.uk/chembl/api/data/"
                                 "target.json?target_chembl_id=" + tid)
                ts = td.get("targets", [])
                if not ts:
                    continue
                name = ts[0].get("pref_name") or ""
                toks = {t for t in _re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", name)}
                hit = next((s for s in symbols if s in toks or s.upper() in toks),
                           None)
                if hit is not None:
                    result = {
                        "symbol": hit,
                        "note": (f"ChEMBL API: {mol_id} '{molecule}' -> target "
                                 f"{tid} '{name}' (pChEMBL {pot[tid]:.2f}) -> "
                                 f"bridge gene {hit}"),
                    }
                    break
            except Exception:
                continue
    except Exception:
        result = None

    cache[key] = result
    try:
        import json as _json
        (_cache_dir() / "chembl_resolve.json").write_text(
            _json.dumps(cache, indent=1))
    except Exception:
        pass
    return result


def _is_smiles_like(s: str) -> bool:
    import re as _re
    return bool(_re.search(r"[A-Za-z0-9@+\-\\/\[\]()#=.]", s)) and len(s) > 2
