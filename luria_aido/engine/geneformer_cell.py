"""Geneformer-V1-10M cell-state encoder + trained linear decoder / adapter.

Task 1 (z = Geneformer(cell) instead of a free learnable z0):

    z           = Geneformer(cell)          # rank-encode -> forward -> mean-pool
    get_expression() = W @ z                # linear decoder -> log1p expression
    effect      = W @ (Aad @ anchor(g))     # perturbation delta -> effect

Geneformer is FROZEN (RULER.md); W and Aad are the trained parameters. The
Geneformer tokenizer is a custom one (gene dictionary pkl, no vocab.txt), so
rank-encoding is done here directly rather than through AutoTokenizer (which
cannot load this model).
"""
from __future__ import annotations

from luria_aido import config
import os
import pickle
import pathlib
from typing import Optional

import numpy as np
import torch

GENE_PKL = "geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl"
MODEL_SUBDIR = "Geneformer-V1-10M"
MAXLEN = 2048
LATENT_DIM = 256


def _snapshot() -> pathlib.Path:
    base = pathlib.Path(os.environ.get("HF_HUB_CACHE", config.hf_home() + "/hub"))
    snaps = base / "models--ctheodoris--Geneformer" / "snapshots"
    if not snaps.is_dir():
        raise FileNotFoundError(
            f"Geneformer snapshot not found under {snaps}. Set HF_HOME (or "
            f"HF_HUB_CACHE) to a cache holding ctheodoris/Geneformer, or fetch "
            f"it with `huggingface-cli download ctheodoris/Geneformer`."
        )
    return sorted(snaps.iterdir())[0]


def load_token_dict() -> dict:
    return pickle.load(open(str(_snapshot() / GENE_PKL), "rb"))


def load_cell_encoder(device: Optional[str] = None):
    """Returns (model, token_dict). Geneformer is frozen."""
    from transformers import BertModel  # concrete class, avoids the broken AutoModel enumeration
    model = BertModel.from_pretrained(str(_snapshot() / MODEL_SUBDIR)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if device is None:
        from luria_aido.gpu import cuda_alloc_ok
        device = "cuda" if cuda_alloc_ok() else "cpu"
    model = model.to(device)
    return model, load_token_dict()


def col_to_tokens(ensemble_ids: list[str], token_dict: dict) -> np.ndarray:
    """Ensembl id per gene column -> token id (or -1 if absent)."""
    return np.array([token_dict.get(e, -1) for e in ensemble_ids])


def encode_cells(X: np.ndarray, col2tok: np.ndarray, model, device: str = "cpu",
                 bs: int = 512, top_n: int = MAXLEN) -> np.ndarray:
    """X: (n_cells, n_genes) counts. Returns (n_cells, 256) cell embeddings."""
    embs = []
    for s in range(0, X.shape[0], bs):
        Xb = X[s:s + bs]
        ids = np.zeros((Xb.shape[0], top_n), dtype=np.int64)
        for i in range(Xb.shape[0]):
            order = np.argsort(-Xb[i])[:top_n]
            toks = col2tok[order]
            toks = toks[toks >= 0]
            ids[i, :len(toks)] = toks
        with torch.no_grad():
            inp = torch.from_numpy(ids).long().to(device)
            h = model(inp).last_hidden_state.cpu()
            m = (ids != 0).astype(np.float32)[:, :, None]
            embs.append(((h * m).sum(1) / (m.sum(1) + 1e-8)).numpy())
    return np.concatenate(embs, 0)


def save_artifacts(out_dir: pathlib.Path, W: np.ndarray, Aad: np.ndarray,
                   z_ctrl_mean: np.ndarray, gene_vocab: list[str],
                   anchor_vocab: list[str]) -> pathlib.Path:
    """Persist the trained decoder (W), adapter (Aad), ctrl mean z, and vocabs."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "decoder_W.npy", W)
    np.save(out_dir / "adapter_Aad.npy", Aad)
    np.save(out_dir / "z_ctrl_mean.npy", z_ctrl_mean)
    (out_dir / "gene_vocab.txt").write_text("\n".join(gene_vocab))
    (out_dir / "anchor_vocab.txt").write_text("\n".join(anchor_vocab))
    return out_dir / "gene_vocab.txt"
