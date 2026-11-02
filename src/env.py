"""Offline environment guard, numeric pinning, and path resolution.

Every compute entrypoint imports this module first. It pins the HuggingFace and
torch caches inside the project tree and asserts offline mode, so that nothing
under src/ outside src/acquire/ can reach the network. This is a hard design
rule for the air-gapped HPC target, enforced from the
first commit rather than retrofitted.

It also pins float32 arithmetic, so that a result does not depend on which GPU
produced it. Work is split across the local GV100 (Volta, sm_70) and the SURF
A10 (Ampere, sm_86); `pin_numerics` and `run_provenance` exist to keep that
split from leaking into the reported numbers. Both take a lazy torch import,
because six CPU-only callers of `go_offline` do not import torch at all.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

# Project root is the parent of src/. No hardcoded absolute paths, no "~".
ROOT = Path(__file__).resolve().parent.parent

_OFFLINE_VARS = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


def go_offline() -> None:
    """Pin project-local caches and force every library into offline mode.

    Call at the top of every module in src/ except src/acquire/.
    """
    os.environ.update(_OFFLINE_VARS)
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
    (ROOT / ".cache" / "hf").mkdir(parents=True, exist_ok=True)
    (ROOT / ".cache" / "torch").mkdir(parents=True, exist_ok=True)


def pin_numerics() -> dict:
    """Fix float32 arithmetic so a result does not depend on which GPU ran it.

    Volta has no TF32 hardware, so torch's TF32 switches are inert on the
    GV100. Ampere honours them, and `cudnn.allow_tf32` defaults to True -- so
    identical fp32 code silently drops mantissa bits on the A10 and not
    locally. That would fold hardware variance into the seed dispersion
    we report, undeclared.

    Call immediately after `go_offline()` in every module that imports torch.
    Returns the pinned state so `run_provenance` can record it.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return _numerics_state()


def _numerics_state() -> dict:
    import torch

    return {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def run_provenance() -> dict:
    """Which machine produced an artefact, and under what arithmetic.

    Seeds of one config must never straddle the two machines (partition by
    experiment, not by seed). Recording the host in every summary makes a
    violation auditable afterwards rather than a thing someone remembers. The
    numeric state is read, not set, so an artefact written by a module that
    forgot to call `pin_numerics` is visible in its own provenance.
    """
    import platform

    import torch

    gpu = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {"name": p.name, "arch": f"sm_{p.major}{p.minor}",
               "memory_gib": round(p.total_memory / 2 ** 30, 1)}
    return {
        "host": platform.node(),
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numerics": _numerics_state(),
    }


def load_config(path: str | Path | None = None) -> dict:
    """Load a config file and resolve every declared path relative to ROOT."""
    path = Path(path) if path else ROOT / "configs" / "paths.yaml"
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for key, value in cfg.get("paths", {}).items():
        cfg["paths"][key] = ROOT / value
    return cfg
