"""CellGF — the Task-1 Cell API: z = Geneformer(cell), decode -> real expression.

Replaces the free-learnable-z0 Cell semantics (RULER contract, new path):
    Cell("K-562")     -> baseline z = mean Geneformer(control cells)  (a real population mean)
    get_expression()  -> z @ W  (log1p expression, pd.Series over the gene vocab)
    gene_knockout(g)  -> z + anchor(g) @ Aad      (latent delta, then readout)
    gene_overexpression(g) -> z - anchor(g) @ Aad (CRISPRa sign convention)
    small_molecule_perturbation(smiles) -> z + ChemBERTa(smiles) @ Dad  (drug delta)

All parameters (W, Aad, z0) are trained and frozen at load; Geneformer is frozen.
"""
from __future__ import annotations

from luria_aido import config
import pathlib
import sys
from typing import Optional

import numpy as np
import pandas as pd

WS = str(config.data_root())
ENG = pathlib.Path(f"{WS}/outputs/51a03013-26bb-4d55-b93b-d343883bc800/engine")
sys.path.insert(0, str(ENG))


def _load_bridge():
    import data as edata
    return edata.load_bridge(ENG), edata.build_vocab(ENG)


class CellGF:
    def __init__(self, cell_line: str = "K-562", engine_root: Optional[str] = None):
        self.cell_line = cell_line
        root = pathlib.Path(engine_root) if engine_root else pathlib.Path(f"{WS}/artifacts/K-562-gf")
        self._W = np.load(root / "decoder_W.npy")
        self._Aad = np.load(root / "adapter_Aad.npy")
        self._Dad = np.load(root / "adapter_Dad.npy")  # (384,256) ChemBERTa -> z-delta, lambda=10
        self._z0 = np.load(root / "z_ctrl_mean.npy").astype(np.float32)
        self._mol_enc = None  # lazily loaded ChemBERTa (MoleculeEncoder)
        self.gene_vocab = tuple((root / "gene_vocab.txt").read_text().splitlines())
        self._br, self._voc = _load_bridge()
        self._idx = self._voc["idx"]
        self._dna_g = {g: i for i, g in enumerate(self._br["prom_genes"])}
        self._prot_g = {g: i for i, g in enumerate(self._br["prot_genes"])}
        self._gfemb = np.load(ENG / "geneformer_gene_emb.npy")
        self._z = self._z0.copy()
        self._history = []

    def clone(self) -> "CellGF":
        c = CellGF(self.cell_line)
        c._z = self._z.copy()
        c._history = list(self._history)
        return c

    def _anchor(self, gene: str) -> Optional[np.ndarray]:
        if gene in self._dna_g and gene in self._prot_g and gene in self._idx:
            g = self._idx[gene]
            return np.concatenate([
                self._br["dna"][self._dna_g[gene]],
                self._br["prot"][self._prot_g[gene]],
                self._gfemb[g],
            ]).astype(np.float32)
        return None

    def gene_knockout(self, gene: str) -> None:
        anc = self._anchor(gene)
        if anc is None:
            raise KeyError(f"{gene!r} not in the gene anchor vocabulary")
        self._z = self._z + (anc @ self._Aad).astype(np.float32)
        self._history.append(("gene_knockout", gene))

    def gene_overexpression(self, gene: str) -> None:
        anc = self._anchor(gene)
        if anc is None:
            raise KeyError(f"{gene!r} not in the gene anchor vocabulary")
        self._z = self._z - (anc @ self._Aad).astype(np.float32)
        self._history.append(("gene_overexpression", gene))

    def small_molecule_perturbation(self, smiles: str) -> None:
        """Apply a drug perturbation as `z + c @ Dad`, symmetric with the KO leg.

        `c` is the 384-dim ChemBERTa embedding of the SMILES (frozen
        DeepChem/ChemBERTa-77M-MLM, same encoder that produced the adapter's
        training features); `Dad` is the linear adapter fitted in the
        Geneformer z-space at 10 uM (artifacts/K-562-gf/adapter_Dad.npy).

        NOTE on the adapter's control status (camp_90021039c): on the RAW
        held-out delta cosine it ties train-mean / shuffle (0.2939 vs 0.2938)
        — it does not beat its trivial controls. On the within-fold CENTERED
        differential cosine it reaches 0.0926 vs shuffle-null max 0.0158
        (z=8.2), i.e. a real but weak per-drug signal. A drug trajectory
        drawn with this method is a model prediction; its per-drug direction
        does not clear the raw control.
        """
        if self._mol_enc is None:
            from luria_aido.engine.encoders import MoleculeEncoder
            self._mol_enc = MoleculeEncoder()
        c = self._mol_enc.embed([smiles])[0].astype(np.float32)  # (384,)
        self._z = self._z + (c @ self._Dad).astype(np.float32)
        self._history.append(("small_molecule", smiles))

    def get_expression(self) -> pd.Series:
        """Real (log1p) expression profile over the gene vocabulary."""
        out = (self._z @ self._W).astype(np.float64)
        return pd.Series(out, index=self.gene_vocab)

    # -- delegated protein/molecule readouts (ALL-FUNCTIONS.md) ------------
    def get_protein_structure(self, target: str, molecule=None, cache_dir=None):
        """Delegated ESMFold structure. Records the target so a following
        get_protein_ligand_interactions(molecule) can compute contacts on
        THIS structure."""
        from luria_aido.proteins import get_protein_structure as _gps
        self._last_structure_target = target
        return _gps(target, molecule=molecule, cache_dir=cache_dir)

    def get_protein_ligand_interactions(self, molecule, target=None, **kwargs):
        """Delegated docking + contacts. Without an explicit target, uses the
        target of the last get_protein_structure call on this Cell."""
        if target is None:
            target = getattr(self, "_last_structure_target", None)
        from luria_aido.proteins import get_protein_ligand_interactions as _gpli
        return _gpli(molecule, target=target, **kwargs)

    def design_molecule_for_target(self, target: str, **kwargs):
        """De-novo SMILES for a target, DELEGATED to WarmMolGenOne
        (sequence-conditioned generative model), NOT known-ligand retrieval."""
        from luria_aido.proteins import design_molecule_for_target as _dmt
        return _dmt(target, **kwargs)

    def get_cell_age(self):
        """ALL-FUNCTIONS option 2: honest raise."""
        from luria_aido.proteins import get_cell_age as _gca
        return _gca()

    @property
    def state(self):
        return self._z

    @property
    def history(self):
        return tuple(self._history)
