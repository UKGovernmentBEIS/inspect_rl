"""CUDA_HOME bootstrap — must run before any import of torch/vllm/deepspeed.

``torch.utils.cpp_extension.CUDA_HOME`` is computed at torch import time from
``os.environ["CUDA_HOME"]`` (plus a few fallbacks) and then cached. If the env
var is unset when torch loads, that cache is ``None`` forever — even if a
later caller sets the env var, DeepSpeed/TRL ops will still raise
``MissingCUDAException``. So the bootstrap has to be called before the first
torch import, not from inside the CLI callback (which fires too late).

Both entry points (``import inspect_rl`` and the ``irl`` console script) route
through ``inspect_rl/__init__.py``, which calls :func:`bootstrap_cuda_home`
before anything heavy is imported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def autodetect_cuda_home() -> str | None:
    """Return the first plausible CUDA install path, or None.

    Probed paths, in order:
    - ``/opt/nvidia/hpc_sdk/Linux_<arch>/<release>/cuda/`` — NVHPC toolchain
    - ``/usr/local/cuda`` — standard install
    """
    candidates: list[Path] = []

    nvhpc_root = Path("/opt/nvidia/hpc_sdk")
    if nvhpc_root.is_dir():
        for arch_dir in nvhpc_root.glob("Linux_*"):
            for release_dir in sorted(arch_dir.iterdir(), reverse=True):
                candidates.append(release_dir / "cuda")

    candidates.append(Path("/usr/local/cuda"))

    for cand in candidates:
        if (cand / "bin" / "nvcc").is_file():
            return str(cand)
    return None


def bootstrap_cuda_home() -> None:
    """Export CUDA_HOME if unset, using :func:`autodetect_cuda_home`.

    Idempotent. Logs to stderr on first set so downstream users see which
    toolchain was picked up.
    """
    if os.environ.get("CUDA_HOME"):
        return
    detected = autodetect_cuda_home()
    if detected is None:
        return
    os.environ["CUDA_HOME"] = detected
    print(f"[inspect_rl] CUDA_HOME={detected} (auto-detected)", file=sys.stderr)
