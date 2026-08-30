"""Frozen encoder wrappers (the four families) with offline HF-cache loading.

RULER.md: encoders are frozen. Only adapters train. Cache lives at
HF_HOME (see luria_aido.config) and is never re-downloaded
unless a model is genuinely absent - absence is reported, not papered over.

Each wrapper:
  - lazy-loads on first use (torch.no_grad, requires_grad_(False)),
  - mean-pools hidden states to a fixed embedding dimension,
  - caches its embedding outputs on disk (keyed by input string hash) so the
    ~20k-gene anchor bridge is computed once.
"""
from __future__ import annotations

from luria_aido import config
import hashlib
import os
import pathlib
from typing import Optional

import numpy as np
import torch

HF_HOME = config.hf_home()
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", str(pathlib.Path(HF_HOME) / "hub"))


class FrozenEncoder:
    """Base: frozen HF model, mean-pooled embedding, disk cache of outputs."""

    model_id: str = ""
    embed_dim: int = 0
    cache_dir: str = ""

    def __init__(self, device: Optional[str] = None, cache_root: Optional[str] = None):
        # USE-MPS.md: is_available() is a driver probe; decide by a real
        # allocation so a contended card falls back to CPU cleanly.
        from luria_aido.gpu import cuda_alloc_ok
        self.device = device or ("cuda" if cuda_alloc_ok() else "cpu")
        root = pathlib.Path(cache_root or (pathlib.Path(HF_HOME) / "embeddings"))
        self.cache_dir = root / self.__class__.__name__
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._tok = None

    # -- lazy model loading ------------------------------------------------
    def _load(self):
        raise NotImplementedError

    @property
    def model(self):
        if self._model is None:
            self._model, self._tok = self._load()
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
        return self._model

    # -- embedding ----------------------------------------------------------
    def embed(self, inputs: list[str]) -> np.ndarray:
        """Mean-pooled embeddings for a batch of inputs; disk-cached."""
        _ = self.model  # lazy _load() first — sets self._tok/_model and embed_dim
        out = np.zeros((len(inputs), self.embed_dim), dtype=np.float32)
        todo = []
        for i, s in enumerate(inputs):
            hit = self._cache_get(s)
            if hit is not None:
                out[i] = hit
            else:
                todo.append(i)
        if todo:
            emb = self._embed_uncached([inputs[i] for i in todo])
            for k, i in enumerate(todo):
                out[i] = emb[k]
                self._cache_put(inputs[i], emb[k])
        return out

    def _embed_uncached(self, inputs: list[str]) -> np.ndarray:
        raise NotImplementedError

    def _cache_key(self, s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24] + ".npy"

    def _cache_get(self, s: str) -> Optional[np.ndarray]:
        p = self.cache_dir / self._cache_key(s)
        if p.exists():
            return np.load(p)
        return None

    def _cache_put(self, s: str, emb: np.ndarray) -> None:
        np.save(self.cache_dir / self._cache_key(s), emb)


class DnaEncoder(FrozenEncoder):
    """DNA family: GenerTeam/GENERanno-eukaryote-0.5b-base (frozen)."""
    model_id = "GenerTeam/GENERanno-eukaryote-0.5b-base"

    def _load(self):
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.model_id, cache_dir=os.environ["HF_HUB_CACHE"])
        cfg = AutoConfig.from_pretrained(self.model_id, cache_dir=os.environ["HF_HUB_CACHE"])
        model = AutoModel.from_pretrained(self.model_id, cache_dir=os.environ["HF_HUB_CACHE"])
        model.to(self.device)
        self.embed_dim = getattr(cfg, "hidden_size", model.config.hidden_size)
        return model, tok

    def _embed_uncached(self, inputs: list[str]) -> np.ndarray:
        toks = self._tok(inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            out = self.model(**toks)
        return out.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)


class ProteinEncoder(FrozenEncoder):
    """Protein family: facebook/esm2_t30_150M_UR50D (frozen)."""
    model_id = "facebook/esm2_t30_150M_UR50D"

    def _load(self):
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.model_id, cache_dir=os.environ["HF_HUB_CACHE"])
        model = AutoModel.from_pretrained(self.model_id, cache_dir=os.environ["HF_HUB_CACHE"])
        model.to(self.device)
        self.embed_dim = model.config.hidden_size
        return model, tok

    def _embed_uncached(self, inputs: list[str]) -> np.ndarray:
        toks = self._tok(inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            out = self.model(**toks)
        return out.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)


class MoleculeEncoder(FrozenEncoder):
    """Molecule family: DeepChem/ChemBERTa-77M-MLM (frozen)."""
    model_id = "DeepChem/ChemBERTa-77M-MLM"

    def _load(self):
        # Concrete class (RobertaModel) — AutoModel.from_pretrained enumerates
        # the full architecture mapping on this env, pulling in timm->wandb,
        # which is broken under NumPy 2.0/protobuf.
        from transformers import AutoTokenizer, RobertaModel
        tok = AutoTokenizer.from_pretrained(self.model_id, cache_dir=os.environ["HF_HUB_CACHE"])
        model = RobertaModel.from_pretrained(self.model_id, cache_dir=os.environ["HF_HUB_CACHE"])
        model.to(self.device)
        self.embed_dim = model.config.hidden_size
        return model, tok

    def _embed_uncached(self, inputs: list[str]) -> np.ndarray:
        toks = self._tok(inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            out = self.model(**toks)
        return out.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)


def dna_f3_token_stream(seqs: list[str], max_len: int = 8192) -> "torch.Tensor":
    """DNA sequence(s) -> F3-format token stream (B, L, 128): GENERanno base
    model last hidden states projected by the fixed R (seed 42) used by the F3
    calibration. Feeds the engine's regulatory head (reg_head(z, toks)).
    AutoModel.from_pretrained is broken on this env (timm->wandb under NumPy
    2.0), so the remote-code module is loaded directly from the HF snapshot.
    """
    import glob
    import importlib.util
    import sys as _sys
    import types
    from transformers import AutoConfig, AutoTokenizer
    mid = "GenerTeam/GENERanno-eukaryote-0.5b-base"
    snap = sorted(glob.glob(os.environ["HF_HUB_CACHE"] + "/models--GenerTeam--GENERanno-eukaryote-0.5b-base/snapshots/*"))[0]
    config = AutoConfig.from_pretrained(mid, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    pkg = types.ModuleType("gen_pkg")
    pkg.__path__ = [snap]
    _sys.modules["gen_pkg"] = pkg
    spec_c = importlib.util.spec_from_file_location("gen_pkg.configuration_generanno", f"{snap}/configuration_generanno.py")
    conf = importlib.util.module_from_spec(spec_c)
    _sys.modules["gen_pkg.configuration_generanno"] = conf
    spec_c.loader.exec_module(conf)
    spec_m = importlib.util.spec_from_file_location("gen_pkg.modeling_generanno", f"{snap}/modeling_generanno.py")
    mod = importlib.util.module_from_spec(spec_m)
    _sys.modules["gen_pkg.modeling_generanno"] = mod
    spec_m.loader.exec_module(mod)
    model = mod.GenerannoModel(config).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    import numpy as _np
    calib = pathlib.Path(__file__).resolve().parent.parent.parent / "outputs" / "51a03013-26bb-4d55-b93b-d343883bc800" / "calibration" / "f3_tok" / "projection.npy"
    R = torch.from_numpy(_np.load(calib)).to("cuda")
    with torch.no_grad():
        toks = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to("cuda")
        h = model(**toks).last_hidden_state.float()
        proj = torch.einsum("blk,kd->bld", h, R)
    return proj


class CellStateEncoder(FrozenEncoder):
    """Cell-state family: ctheodoris/Geneformer (V1-10M preferred, per RULER)."""
    model_id = "ctheodoris/Geneformer"
    geneformer_model = "Geneformer-V1-10M"   # swap to V2-104M is a REPORTABLE change

    def _load(self):
        # Geneformer requires its own loader (gene-order tokenizer); the
        # HF snapshot holds the pretrained model dir under model_id/<geneformer_model>.
        snapshot = _resolve_snapshot(self.model_id)
        model_dir = snapshot / self.geneformer_model
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Geneformer snapshot present at {snapshot} but {self.geneformer_model} "
                f"not found; check the cache. Absence is reported, never silently "
                f"substituted (RULER: fallback swap is a reportable change)."
            )
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, cache_dir=os.environ["HF_HUB_CACHE"])
        model = AutoModel.from_pretrained(model_dir, cache_dir=os.environ["HF_HUB_CACHE"])
        model.to(self.device)
        self.embed_dim = model.config.hidden_size
        return model, tok

    def _embed_uncached(self, inputs: list[str]) -> np.ndarray:
        toks = self._tok(inputs, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            out = self.model(**toks)
        return out.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)


def _resolve_snapshot(model_id: str) -> pathlib.Path:
    import json
    base = pathlib.Path(os.environ["HF_HUB_CACHE"]) / ("models--" + model_id.replace("/", "--"))
    refs = base / "refs"
    if not refs.exists():
        raise FileNotFoundError(f"HF cache entry for {model_id} not found (refs missing): {base}")
    main = (refs / "main").read_text().strip()
    return base / "snapshots" / main
