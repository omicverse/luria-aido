"""Cell - the public Luria-AIDO API (RULER.md).

    c = Cell("K-562")                 # baseline virtual cell
    c2 = c.clone()                    # real copy of state; branch here
    c2.small_molecule_perturbation("CC1=C(C=C(C=C1)...")   # in-place, records history
    c2.gene_knockout("ABL1")
    expr = c2.get_expression()        # pd.Series over the gene vocabulary
    act  = c2.get_regulatory_activity("<sequence>")

Protein readouts are DELEGATED to real external methods and return
provenance (ALL-FUNCTIONS.md): get_protein_structure runs real ESMFold v1 on
the target's UniProt sequence; get_protein_ligand_interactions docks the
molecule (AutoDock Vina) into a pocket of that structure and computes real
contacts; get_cell_age raises with the honest reason (no age measurement, no
delegable clock). See luria_aido/proteins.py.

Readouts never mutate state (P1); perturbations compose and record lineage
(P3); every readout decodes from the one shared latent state (P2/P4).
"""
from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import torch

# NumPy 2.0 removed dtype aliases the transformers->timm->wandb import chain
# still uses; re-add them so encoder loading works on this env.
if not hasattr(np, "float_"):
    for _old, _new in [("float_", "float64"), ("complex_", "complex128"), ("int_", "int64"),
                       ("bool8", "bool_"), ("str_", "str_"), ("bytes_", "bytes_"),
                       ("unicode_", "str_"), ("object_", "object_")]:
        if not hasattr(np, _old):
            setattr(np, _old, getattr(np, _new))

from luria_aido.engine.bridge import GeneBridge
from luria_aido.engine.decoders import DecoderBank, get_expression, get_regulatory_activity_f3
from luria_aido.engine.encoders import MoleculeEncoder
from luria_aido.engine.perturbations import (
    PerturbationAdapter,
    apply_gene_knockout,
    apply_gene_overexpression,
    apply_small_molecule,
)
from luria_aido.engine.state import CellState
from luria_aido.stubs import get_cell_age, get_protein_ligand_interactions, get_protein_structure


class Cell:
    def __init__(
        self,
        cell_line: str = "K-562",
        state: Optional[CellState] = None,
        *,
        engine_root: Optional[str] = None,
    ):
        self.cell_line = cell_line
        # engine_root: where trained adapters / anchors / baseline states live.
        self._root = pathlib.Path(engine_root) if engine_root else _default_root() / cell_line
        self._bridge = GeneBridge(self._root / "anchors" / "genes.pt")
        self._bank = DecoderBank(self._root / "decoders", vocab=(), latent_dim=_LATENT_DIM)
        self._mol_enc = MoleculeEncoder(device="cpu")   # real device chosen lazily
        self._state = state or self._load_baseline()

    # -- state lifecycle ----------------------------------------------------
    def _load_baseline(self) -> CellState:
        """Load the K-562 baseline latent state; trains one if absent."""
        baseline = self._root / "baseline" / "state"
        if baseline.exists():
            st = CellState.load(baseline)
            if not self._bridge.ready:
                self._bridge.load()
            st.gene_vocab = self._bridge.vocab
            return st
        raise NotImplementedError(
            f"No baseline state trained for {self.cell_line} at {baseline}. "
            "Baseline training (F1/F2 control data -> latent state) is a "
            "pipeline step; a heuristic baseline is not a substitute."
        )

    def _vocab(self) -> tuple[str, ...]:
        # The state carries its own gene vocabulary (single source of truth);
        # the bridge is only the fallback for states loaded without one.
        st = getattr(self, "_state", None)
        if st is not None and st.gene_vocab:
            return st.gene_vocab
        if not self._bridge.ready:
            self._bridge.load()
        return self._bridge.vocab

    def clone(self) -> "Cell":
        """Real, independent copy of state (branch point for P3)."""
        return Cell(self.cell_line, state=self._state.clone(), engine_root=str(self._root))

    # -- interventions (engine steps) --------------------------------------
    def small_molecule_perturbation(self, smiles: str) -> None:
        """Apply a small-molecule perturbation; state updates in place."""
        emb = torch.from_numpy(self._mol_enc.embed([smiles])[0])
        adapter = self._perturbation_adapter(self._mol_enc.embed_dim)
        self._state = apply_small_molecule(self._state, smiles, emb, adapter)

    def _as_tensor(self, emb):
        """Bridge/molecule encoders return ndarray or Tensor depending on
        backend — normalise both to a torch Tensor (ALL-FUNCTIONS fix)."""
        if isinstance(emb, np.ndarray):
            return torch.from_numpy(emb)
        return emb if isinstance(emb, torch.Tensor) else torch.as_tensor(emb)

    def gene_knockout(self, gene: str) -> None:
        """Knock out a gene by its anchor in the shared space."""
        anchor = self._as_tensor(self._bridge.embed([gene])[0])
        adapter = self._perturbation_adapter(anchor.shape[0])
        self._state = apply_gene_knockout(self._state, gene, anchor, adapter)

    def gene_overexpression(self, gene: str) -> None:
        """Overexpress a gene (Norman 2019 CRISPRa convention)."""
        anchor = self._as_tensor(self._bridge.embed([gene])[0])
        adapter = self._perturbation_adapter(anchor.shape[0])
        self._state = apply_gene_overexpression(self._state, gene, anchor, adapter)

    def _perturbation_adapter(self, in_dim: int):
        """Load the trained engine adapter for this signature type (engine wiring,
        checkpoint nd_cur1). in_dim 1921 -> gene anchor: bridge + ko_adapter
        composition; in_dim 384 -> ChemBERTa embedding: drug adapter at 1 uM."""
        import torch.nn as _nn
        if in_dim == 1921 or in_dim == 641:
            ckpt = self._root / "adapters" / "ko_1921.pt"
            if not ckpt.exists():
                raise NotImplementedError(
                    f"KO adapter not trained: {ckpt}. Training is a pipeline step."
                )
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)
            # bridge input dim may be 1921 (DNA+prot) or 641 (prot-only) — read it
            bridge_in = sd["0.0.weight"].shape[1]
            def mlp(i, h, o):
                return _nn.Sequential(_nn.Linear(i, h), _nn.GELU(), _nn.LayerNorm(h), _nn.Linear(h, o))
            seq = _nn.Sequential(mlp(bridge_in, 512, _LATENT_DIM), mlp(_LATENT_DIM, 512, _LATENT_DIM))
            seq.load_state_dict(sd)
            seq.eval()
            return seq
        if in_dim == 384:
            ckpt = self._root / "adapters" / "drug_384.pt"
            if not ckpt.exists():
                raise NotImplementedError(
                    f"Drug adapter not trained: {ckpt}. Training is a pipeline step."
                )
            w = _DrugAdapter384(_LATENT_DIM)
            w.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            w.eval()
            return w
        raise NotImplementedError(f"no trained adapter for signature dim {in_dim}")

    # -- readouts (all decode from the shared state; never mutate) ---------
    def get_expression(self) -> pd.Series:
        """Standardized effect size (NOT expression level).

        Returns the model's decoded standardized contrast
        ``y = (condition_mean - control) / sd`` for the current latent state,
        value range ~[-2.7, 2.4]. This is an EFFECT SIZE, not a transcript
        abundance. Do NOT subtract the baseline cell's decode: ``z0``'s own
        decode carries a spurious signal (mean ~-0.23, std ~0.75, should be
        ~0), so ``decode(z) - decode(z0)`` destroys signal. Read the perturbed
        state's value directly as the predicted effect.
        """
        if not self._bank.vocab:
            self._bank.vocab = self._vocab()
        return get_expression(self._state, self._bank)

    def get_protein_structure(self, target: str, molecule=None, cache_dir=None):
        """Delegated ESMFold structure (luria_aido.proteins). Records the
        target so a following get_protein_ligand_interactions(molecule) can
        compute contacts on THIS structure (ALL-FUNCTIONS chain)."""
        from luria_aido.proteins import get_protein_structure as _gps
        self._last_structure_target = target
        return _gps(target, molecule=molecule, cache_dir=cache_dir)

    def get_protein_ligand_interactions(self, molecule, target=None, **kwargs):
        """Delegated docking + contacts. Without an explicit target, uses the
        target of the last get_protein_structure call on this Cell (real
        context, not a guessed mapping); the protein layer then falls back to
        ChEMBL resolution, then raises honestly."""
        if target is None:
            ctx = getattr(self, "_last_structure_target", None)
            if ctx is not None:
                target = ctx
        from luria_aido.proteins import get_protein_ligand_interactions as _gpli
        return _gpli(molecule, target=target, **kwargs)

    def get_regulatory_activity(self, sequence: str) -> float:
        """Engine regulatory readout over the F3 token stream of a DNA sequence
        (GENERanno hidden states -> fixed projection -> reg head). Wired from
        the nd_cur1 checkpoint; returns the mean of the Dev/Hk channels."""
        if not self._bank.vocab:
            self._bank.vocab = self._vocab()
        from luria_aido.engine.encoders import dna_f3_token_stream
        toks = dna_f3_token_stream([sequence])
        return get_regulatory_activity_f3(self._state, toks, self._bank)

    def get_cell_age(self):
        """ALL-FUNCTIONS option 2: honest raise (no measurement, no delegable
        clock for the immortal K-562 line)."""
        from luria_aido.proteins import get_cell_age as _gca
        return _gca()

    def design_molecule_for_target(self, target: str, **kwargs):
        """De-novo SMILES for a target, DELEGATED to WarmMolGenOne
        (sequence-conditioned generative model), NOT retrieval of known
        ligands. Returns a list of valid SMILES with provenance."""
        from luria_aido.proteins import design_molecule_for_target as _dmt
        return _dmt(target, **kwargs)

    # -- shell-facing state --------------------------------------------------
    @property
    def state(self) -> CellState:
        return self._state

    @property
    def history(self):
        return self._state.history

    def save(self, directory: str) -> pathlib.Path:
        return self._state.save(directory)


def _default_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "artifacts"


class _DrugAdapter384(torch.nn.Module):
    """ChemBERTa embedding (384) -> latent delta at the fixed 1 uM dose
    (engine drug adapter, wired from the nd_cur1 checkpoint)."""

    def __init__(self, latent_dim: int, hidden: int = 512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(384 + 1, hidden), torch.nn.GELU(), torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, latent_dim),
        )
        self.register_buffer("dose", torch.tensor([float(np.log10(1.0 * 1e4))]))

    def forward(self, x):
        return self.net(torch.cat([x, self.dose.expand(x.shape[0], 1)], dim=-1))


_LATENT_DIM = 256  # engine config: shared representation size (pre-registered)
