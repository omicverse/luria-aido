"""Gene bridge - ~20k genes as shared anchors across the three families.

RULER.md: genes are the shared units (DNA sequence + protein sequence +
expression profile). The bridge precomputes one anchor vector per gene:

    anchor(g) = MLP_adapter([DNA_enc(promoter|CDS), prot_enc(AA), cellstate_enc(expr profile)])

anchors are cached to disk and reused by every readout; they are the fixed
coordinates in which the shared latent state lives.

Sequence acquisition (CDS / AA per gene) is a data-provision step: pulled
once from Ensembl/UniProt for the K-562-relevant gene vocabulary and cached;
never guessed from the gene symbol.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import torch

from luria_aido.engine.encoders import DnaEncoder, ProteinEncoder, CellStateEncoder


class GeneBridge:
    """Precomputed per-gene anchor table (n_genes x D_anchor)."""

    def __init__(self, anchor_path: Optional[pathlib.Path] = None):
        self.anchor_path = anchor_path
        self._table = None          # (n_genes, D_anchor) float32
        self._vocab: tuple = ()

    @property
    def ready(self) -> bool:
        return self._table is not None

    def load(self) -> None:
        if self.anchor_path is not None and self.anchor_path.exists():
            d = torch.load(self.anchor_path, map_location="cpu", weights_only=True)
            self._table = d["anchors"]
            self._vocab = tuple(d["genes"])
            return
        raise FileNotFoundError(
            f"Gene bridge anchors not computed yet: {self.anchor_path}. "
            "Run luria_aido.engine.bridge.build_anchor_table() first (one-time, "
            "cached). Anchors are never guessed."
        )

    def embed(self, genes: list[str]) -> np.ndarray:
        if not self.ready:
            self.load()
        idx = [self._vocab.index(g) for g in genes]
        return self._table[idx]

    @property
    def vocab(self):
        return self._vocab


def build_anchor_table(
    genes: list[str],
    dna_seqs: dict[str, str],
    aa_seqs: dict[str, str],
    expr_profiles: dict[str, np.ndarray],
    out_path: pathlib.Path,
    device: str = "cuda",
) -> pathlib.Path:
    """One-time computation of the anchor table.

    dna_seqs: gene -> DNA sequence (promoter or CDS, decided by provisioning)
    aa_seqs : gene -> amino-acid sequence
    expr_profiles: gene -> expression-profile embedding (via Geneformer), or
                  None for genes without a profile (masked during training)
    """
    dna_enc, prot_enc = DnaEncoder(device=device), ProteinEncoder(device=device)
    genes = [g for g in genes if g in dna_seqs and g in aa_seqs]
    dna = dna_enc.embed([dna_seqs[g] for g in genes])
    prot = prot_enc.embed([aa_seqs[g] for g in genes])
    expr = np.stack([expr_profiles.get(g, np.zeros(prot.shape[1], np.float32)) for g in genes])
    anchors = np.concatenate([dna, prot, expr], axis=1).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"genes": genes, "anchors": torch.from_numpy(anchors)}, out_path)
    return out_path
