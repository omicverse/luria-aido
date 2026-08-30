"""CellState - the single source of truth for a virtual cell.

The whole point of Luria-AIDO (RULER.md): there is ONE latent state. Every
readout decodes from it; every intervention updates it. Nothing else carries
information about the cell.

State is immutable in spirit: interventions return a NEW CellState (the old
one stays intact, which is what makes clone() a real copy and P1 trivially
enforceable). Gene vocabulary and lineage history travel with the state so a
saved cell can be audited and replayed.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class PerturbationStep:
    """One applied intervention, recorded for lineage/replay."""
    kind: str            # "small_molecule" | "gene_knockout" | "gene_overexpression" | ...
    payload: dict        # {"smiles": ...} or {"gene": ...}
    order: int


@dataclass
class CellState:
    cell_line: str                     # e.g. "K-562"
    latent: torch.Tensor               # (D,) shared representation
    gene_vocab: Tuple[str, ...]        # canonical gene symbols, fixed order
    history: Tuple[PerturbationStep, ...] = field(default_factory=tuple)
    adapter_version: Optional[str] = None   # checkpoint identity of the adapters that trained this state
    metadata: dict = field(default_factory=dict)

    # -- copy semantics -----------------------------------------------------
    def clone(self) -> "CellState":
        """Deep, independent copy of state (clone() semantics per RULER)."""
        return CellState(
            cell_line=self.cell_line,
            latent=self.latent.detach().clone(),
            gene_vocab=self.gene_vocab,
            history=self.history,
            adapter_version=self.adapter_version,
            metadata=dict(self.metadata),
        )

    def apply_step(self, kind: str, payload: dict) -> "CellState":
        """Append a lineage step to a copy of this state (no in-place edit)."""
        new = self.clone()
        new.history = new.history + (PerturbationStep(kind, payload, order=len(new.history) + 1),)
        return new

    # -- persistence --------------------------------------------------------
    def save(self, directory: str | pathlib.Path) -> pathlib.Path:
        d = pathlib.Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        manifest = {
            "cell_line": self.cell_line,
            "latent_shape": list(self.latent.shape),
            "dtype": str(self.latent.dtype),
            "gene_vocab_size": len(self.gene_vocab),
            "gene_vocab_hash": _hash_vocab(self.gene_vocab),
            "adapter_version": self.adapter_version,
            "history": [
                {"kind": s.kind, "payload": s.payload, "order": s.order}
                for s in self.history
            ],
            "metadata": self.metadata,
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
        torch.save(self.latent, d / "latent.pt")
        return d / "manifest.json"

    @classmethod
    def load(cls, directory: str | pathlib.Path) -> "CellState":
        d = pathlib.Path(directory)
        manifest = json.loads((d / "manifest.json").read_text())
        latent = torch.load(d / "latent.pt", map_location="cpu", weights_only=True)
        return cls(
            cell_line=manifest["cell_line"],
            latent=latent,
            gene_vocab=tuple(),   # vocab restored by the owning Cell from its registry
            history=tuple(
                PerturbationStep(s["kind"], s["payload"], s["order"])
                for s in manifest["history"]
            ),
            adapter_version=manifest.get("adapter_version"),
            metadata=manifest.get("metadata", {}),
        )


def _hash_vocab(vocab) -> str:
    import hashlib
    return hashlib.sha256("|".join(vocab).encode("utf-8")).hexdigest()[:16]
