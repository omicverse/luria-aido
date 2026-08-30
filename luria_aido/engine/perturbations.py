"""Interventions - operators that update the shared latent state.

Every intervention maps a perturbation signature to a delta on the latent
state through a trained adapter (the ONLY trainable parameters):

    small_molecule(SMILES): ChemBERTa(smiles) -> adapter -> delta
    gene_knockout(g):       anchor(g)         -> adapter -> delta
    gene_overexpression(g): anchor(g)         -> adapter -> delta

Perturbations compose: each returns a new CellState with history appended.
Order effects (A then B vs B then A) are measured in P3, not assumed away.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from luria_aido.engine.state import CellState

RNG = torch.Generator


class PerturbationAdapter(nn.Module):
    """MLP mapping a perturbation signature to a latent-state delta.

    Capacity is fixed by the pre-registered ablation; do not grow it after
    results exist. Latent dim and adapter dim come from the engine config.
    """

    def __init__(self, in_dim: int, latent_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, sig: torch.Tensor) -> torch.Tensor:
        return self.net(sig)


def apply_small_molecule(
    state: CellState, smiles: str, mol_embed: torch.Tensor, adapter: PerturbationAdapter
) -> CellState:
    with torch.no_grad():
        delta = adapter(mol_embed.unsqueeze(0)).squeeze(0)
    new = state.apply_step("small_molecule", {"smiles": smiles})
    new.latent = state.latent + delta
    return new


def apply_gene_perturbation(
    state: CellState,
    gene: str,
    anchor: torch.Tensor,
    adapter: PerturbationAdapter,
    kind: str,
) -> CellState:
    if gene not in state.gene_vocab:
        raise KeyError(f"{gene!r} not in the cell's gene vocabulary")
    with torch.no_grad():
        delta = adapter(anchor.unsqueeze(0)).squeeze(0)
    new = state.apply_step(kind, {"gene": gene})
    new.latent = state.latent + delta
    return new


def apply_gene_perturbation_signed(state, gene, anchor, adapter, kind, sign):
    if gene not in state.gene_vocab:
        raise KeyError(f"{gene!r} not in the cell's gene vocabulary")
    with torch.no_grad():
        delta = sign * adapter(anchor.unsqueeze(0)).squeeze(0)
    new = state.apply_step(kind, {"gene": gene})
    new.latent = state.latent + delta
    return new


def apply_gene_knockout(state, gene, anchor, adapter) -> CellState:
    # BUILD.md Fix B: the adapter was trained on CRISPRa (activation) data, so
    # +adapter(anchor) is the activation direction; knockout is its negation.
    # This makes KO and OE DISTINGUISHABLE (opposite signs), which the previous
    # code violated (both added the same delta).
    return apply_gene_perturbation_signed(state, gene, anchor, adapter, "gene_knockout", -1.0)


def apply_gene_overexpression(state, gene, anchor, adapter) -> CellState:
    return apply_gene_perturbation_signed(state, gene, anchor, adapter, "gene_overexpression", +1.0)
