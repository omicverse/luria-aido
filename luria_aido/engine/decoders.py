"""Decoders - readouts that all decode from the SAME latent state.

This is the world-model contract: get_expression and get_regulatory_activity
are functions of one shared state. Two independent predictors would have no
reason to agree; decoders over a shared state are consistent by construction
(P2 tests exactly this).

Decoders are small MLP adapters trained on top of the frozen state
representation. Until a trained checkpoint is loaded, readouts raise
NotImplementedError - they are never approximated with a heuristic.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn


class ExpressionDecoder(nn.Module):
    """latent state (D,) -> expression (G,) over the fixed gene vocabulary."""

    def __init__(self, latent_dim: int, n_genes: int, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_genes),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class RegulatoryActivityDecoder(nn.Module):
    """latent state (D,) x DNA sequence embedding (E,) -> scalar activity."""

    def __init__(self, latent_dim: int, dna_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + dna_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, latent: torch.Tensor, dna_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, dna_emb], dim=-1))


class F3RegulatoryHead(nn.Module):
    """Engine regulatory head: conv over the F3 token stream (B, L, 128) +
    latent z -> (B, 2) activity (Dev, Hk). Same architecture as the engine's
    RegulatoryHead; wired from the nd_cur1 checkpoint (decoders/regulatory.pt)."""

    def __init__(self, latent_dim: int = 256, token_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(token_dim, 64, kernel_size=21, padding=10), nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=15, padding=7), nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=9, padding=4), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim + 64, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, 2),
        )

    def forward(self, z, toks):
        h = self.conv(toks.transpose(1, 2))
        pooled = torch.cat([h.mean(2), h.max(2).values], dim=1)
        return self.head(torch.cat([z.expand(pooled.shape[0], -1), pooled], dim=1))


class DecoderBank:
    """Loads the trained decoders once; raises if checkpoints are absent."""

    def __init__(self, checkpoint_dir: Optional[pathlib.Path], vocab: tuple[str, ...], latent_dim: int):
        self.checkpoint_dir = pathlib.Path(checkpoint_dir) if checkpoint_dir else None
        self.vocab = vocab
        self.latent_dim = latent_dim
        self._expr: Optional[ExpressionDecoder] = None
        self._reg: Optional[RegulatoryActivityDecoder] = None

    def _must(self, name: str) -> pathlib.Path:
        p = self.checkpoint_dir / name
        if self.checkpoint_dir is None or not p.exists():
            raise NotImplementedError(
                f"Decoder {name} has no trained checkpoint (expected {p}). "
                "Training is part of the pipeline; until then no readout is "
                "approximated with a heuristic."
            )
        return p

    @property
    def expression(self) -> ExpressionDecoder:
        if self._expr is None:
            hidden = self._expr_hidden()
            self._expr = ExpressionDecoder(self.latent_dim, len(self.vocab), hidden=hidden)
            self._expr.load_state_dict(torch.load(self._must("expression.pt"), map_location="cpu", weights_only=True))
            self._expr.eval()
        return self._expr

    def _expr_hidden(self) -> int:
        import json
        meta = self.checkpoint_dir / "expression_meta.json"
        if meta is not None and meta.exists():
            return int(json.loads(meta.read_text()).get("hidden", 1024))
        return 1024

    @property
    def regulatory(self) -> RegulatoryActivityDecoder:
        if self._reg is None:
            self._reg = RegulatoryActivityDecoder(self.latent_dim, self.latent_dim)
            self._reg.load_state_dict(torch.load(self._must("regulatory.pt"), map_location="cpu", weights_only=True))
            self._reg.eval()
        return self._reg

    @property
    def regulatory_f3(self) -> F3RegulatoryHead:
        if self._reg is None:
            self._reg = F3RegulatoryHead(self.latent_dim)
            self._reg.load_state_dict(torch.load(self._must("regulatory.pt"), map_location="cpu", weights_only=True))
            self._reg.eval()
        return self._reg


def get_expression(state, bank: DecoderBank) -> pd.Series:
    with torch.no_grad():
        out = bank.expression(state.latent.unsqueeze(0)).squeeze(0)
    return pd.Series(out.numpy(), index=state.gene_vocab)


def get_regulatory_activity_f3(state, toks: torch.Tensor, bank: DecoderBank) -> float:
    """Engine regulatory readout: z + F3 token stream -> (Dev, Hk), mean channel."""
    with torch.no_grad():
        head = bank.regulatory_f3.to(toks.device)
        out = head(state.latent.unsqueeze(0).to(toks.device), toks)
    return float(out.mean().item())
