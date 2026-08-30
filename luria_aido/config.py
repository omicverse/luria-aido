"""Runtime paths and device setup.

Every path is an environment variable with a sensible default, so the package
runs outside the machine it was developed on.

    LURIA_AIDO_DATA   artifacts/weights root (default: ~/.cache/luria-aido)
    HF_HOME           HuggingFace cache for the frozen encoders
    CUDA_MPS_PIPE_DIRECTORY / CUDA_MPS_LOG_DIRECTORY
                      set only if already present in the environment; on a
                      shared GPU they are required to obtain a CUDA context.
"""
from __future__ import annotations
import os
import pathlib

_DEFAULT_DATA = pathlib.Path.home() / ".cache" / "luria-aido"


def data_root() -> pathlib.Path:
    """Root for artifacts (adapters, anchor tables, gene vocabularies)."""
    return pathlib.Path(os.environ.get("LURIA_AIDO_DATA", str(_DEFAULT_DATA)))


def artifact(*parts: str) -> pathlib.Path:
    return data_root().joinpath(*parts)


def hf_home() -> str:
    return os.environ.get("HF_HOME", str(pathlib.Path.home() / ".cache" / "huggingface"))


def require(path: pathlib.Path, what: str) -> pathlib.Path:
    """Fail with an actionable message rather than a bare FileNotFoundError."""
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}.\n"
            f"Set LURIA_AIDO_DATA to the directory holding the released artifacts, "
            f"or see docs/DATA.md for how to rebuild them."
        )
    return path


def cuda_ok() -> bool:
    """A real allocation, not torch.cuda.is_available().

    On a shared GPU `is_available()` returns True even when no context can be
    obtained; only an allocation tells you whether the device is usable.
    """
    import torch

    for var, default in (
        ("CUDA_MPS_PIPE_DIRECTORY", None),
        ("CUDA_MPS_LOG_DIRECTORY", None),
    ):
        if default and var not in os.environ:
            os.environ[var] = default
    try:
        torch.zeros(8, device="cuda")
        return True
    except Exception:
        return False


def subprocess_env() -> dict:
    """Environment for child processes; MPS vars are not inherited by default."""
    env = dict(os.environ)
    for var in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY"):
        if var in os.environ:
            env[var] = os.environ[var]
    return env
