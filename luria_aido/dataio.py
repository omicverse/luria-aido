"""Dataset loaders for the three families' data (all cached locally).

  F1: data/NormanWeissman2019_filtered.h5ad   (K562 CRISPRa, ~200k cells)
  F2: data/SrivatsanTrapnell2020_sciplex3.h5ad (sci-Plex3, A549/K562/MCF7)
  F3: GenerTeam datasets in the HF cache (DeepSTARR / gener-tasks / variant-effect)
"""
from __future__ import annotations

from luria_aido import config
import os
import pathlib

import anndata as ad

WORKSPACE = config.data_root()
DATA = WORKSPACE / "data"
HF_HOME = config.hf_home()
HF_HUB = pathlib.Path(HF_HOME) / "hub"

F1_PATH = DATA / "NormanWeissman2019_filtered.h5ad"
F2_PATH = DATA / "SrivatsanTrapnell2020_sciplex3.h5ad"


def load_norman(backed: bool = True) -> ad.AnnData:
    return ad.read_h5ad(F1_PATH, backed="r" if backed else None)


def load_sciplex3(backed: bool = True) -> ad.AnnData:
    return ad.read_h5ad(F2_PATH, backed="r" if backed else None)


def gener_dataset(name: str, split: str | None = None):
    """Load a GenerTeam HF dataset from cache (offline)."""
    import datasets
    path = {
        "deepstarr": "GenerTeam/DeepSTARR-enhancer-activity",
        "tasks": "GenerTeam/gener-tasks",
        "variants": "GenerTeam/variant-effect-prediction",
    }[name]
    os.environ.setdefault("HF_HOME", HF_HOME)
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB))
    ds = datasets.load_dataset(path, split=split, cache_dir=str(HF_HUB))
    return ds
