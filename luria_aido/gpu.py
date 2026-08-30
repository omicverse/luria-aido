"""GPU availability for the shared H100 (USE-MPS.md).

The card is shared with concurrent jobs under CUDA MPS. Two rules:

1. ``CUDA_MPS_PIPE_DIRECTORY`` must be set BEFORE the first CUDA context is
   created, or allocation fails with ``CUDA-capable device(s) is/are busy or
   unavailable``. This module sets it (if missing) before any CUDA use, so
   callers never depend on the environment being pre-configured.
2. ``torch.cuda.is_available()`` only probes the driver — it returns True
   even when no context can actually be obtained. The only honest check is a
   real tiny allocation; callers use :func:`cuda_alloc_ok` and fall back to
   CPU cleanly (with the fallback visible in the returned provenance) instead
   of raising a CUDA error.
"""

import os

_MPS_PIPE = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "")
_MPS_LOG = os.environ.get("CUDA_MPS_LOG_DIRECTORY", "")


def ensure_mps_env() -> None:
    """Set the MPS env vars if the caller has not (idempotent)."""
    os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", _MPS_PIPE)
    os.environ.setdefault("CUDA_MPS_LOG_DIRECTORY", _MPS_LOG)


def cuda_alloc_ok() -> bool:
    """True only if a REAL CUDA allocation succeeds.

    ``torch.cuda.is_available()`` is not sufficient (driver probe only).
    """
    ensure_mps_env()
    import torch
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.zeros(8, device="cuda")
        del t
        torch.cuda.empty_cache()
        return True
    except Exception:
        return False
