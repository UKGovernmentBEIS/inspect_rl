"""Timestamped run directory: <base>/<timestamp>_<suffix>/{checkpoints,eval_logs,manifest.json}.

Naming convention (chosen by `create_run_dir` at call time):

- Outside Slurm: ``irl_output/<YYYY-MM-DD-HH-MM-SS>_irl/``
- Inside Slurm (``$SLURM_JOB_ID`` set): ``irl_output_slurm/<YYYY-MM-DD-HH-MM-SS>_slurm_<jobid>/``

A single ``latest`` symlink in the base directory points at the most recent
run dir — one symlink per base, not the previous fanout of ``latest_vllm`` /
``latest_combined_*`` files.

Sbatch scripts can pre-create the run directory (so vllm.out, trainer.out etc.
land alongside the trainer's artifacts) and export
``INSPECT_RL_RUN_DIR=<abs path>`` — in that case `create_run_dir` uses the
provided path as-is and skips the timestamp.

Base-path resolution for relative ``base`` arguments:

1. ``$INSPECT_RL_OUTPUT_ROOT`` — explicit override, wins if set.
2. ``$SCRATCH`` — Slurm HPC scratch (where many clusters point it at fast
   storage). Training writes a lot of bytes; scratch is the right default
   where it exists.
3. Nearest ``pyproject.toml`` walking up from cwd — so a consumer that
   ``uv add``'d us gets outputs inside *their* project, not inside our
   installed package.
4. cwd — final fallback.

Absolute ``base`` paths bypass all of this.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _find_pyproject_root(start: Path) -> Path | None:
    """Walk up from *start* looking for a directory containing pyproject.toml."""
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def resolve_output_root() -> Path:
    """Pick the base directory for relative output paths.

    See module docstring for the resolution order.
    """
    override = os.environ.get("INSPECT_RL_OUTPUT_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch).expanduser().resolve()

    pyproject = _find_pyproject_root(Path.cwd().resolve())
    if pyproject is not None:
        return pyproject

    return Path.cwd().resolve()


def _slurm_job_id() -> str | None:
    """Return the Slurm job ID if we are inside a job allocation."""
    return os.environ.get("SLURM_JOB_ID") or None


def _default_base(base: str) -> str:
    """If the caller passed the local-default ``irl_output``, swap to the
    Slurm-default ``irl_output_slurm`` when running under Slurm."""
    if base == "irl_output" and _slurm_job_id():
        return "irl_output_slurm"
    return base


def _run_dir_suffix() -> str:
    """Trailing tag on the timestamped run dir — ``slurm_<jobid>`` or ``irl``."""
    job = _slurm_job_id()
    return f"slurm_{job}" if job else "irl"


def _update_latest_symlink(parent: Path, run: Path) -> None:
    """Atomically point ``<parent>/latest`` at ``run``."""
    latest = parent / "latest"
    tmp = parent / f".latest-{run.name}"
    try:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        tmp.symlink_to(run)
        tmp.replace(latest)
    except OSError:
        pass


def create_run_dir(
    base: str = "irl_output",
    config: dict[str, Any] | None = None,
) -> Path:
    """Create a run directory and return its path.

    If ``$INSPECT_RL_RUN_DIR`` is set, that absolute path is used verbatim —
    sbatch scripts use this to share one directory between the launcher (which
    writes vllm.out / trainer.out) and the trainer (which writes
    checkpoints/, eval_logs/, manifest.json). Otherwise the directory is
    ``<resolved_base>/<timestamp>_<suffix>/`` with the suffix chosen by
    ``_run_dir_suffix`` (slurm vs not).

    Layout:
        <run>/
            checkpoints/   — model weights (GRPOConfig.output_dir)
            eval_logs/     — Inspect .eval files
            manifest.json  — snapshot of inputs (when *config* is given)
    """
    explicit = os.environ.get("INSPECT_RL_RUN_DIR")
    if explicit:
        run = Path(explicit).expanduser().resolve()
        run.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
        base_path = Path(_default_base(base)).expanduser()
        if not base_path.is_absolute():
            base_path = resolve_output_root() / base_path
        run = base_path / f"{ts}_{_run_dir_suffix()}"

    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "eval_logs").mkdir(parents=True, exist_ok=True)

    if config:
        (run / "manifest.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n"
        )

    _update_latest_symlink(run.parent, run)
    return run
