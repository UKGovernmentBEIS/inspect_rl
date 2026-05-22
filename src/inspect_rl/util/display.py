"""Training step display — works in terminal, Jupyter, and wandb streams."""

from __future__ import annotations

import logging
import time
from typing import Any

from inspect_rl.util import summary as _summary

logger = logging.getLogger(__name__)

_step_counter = 0
_last_time: float | None = None

verbose = False
max_steps: int | None = None  # set once from trainer.py for the "N/M" progress view


def set_step_counter(step: int) -> None:
    global _step_counter
    _step_counter = step


def log_rollout(
    prompts: list[Any],
    scores: list[dict[str, float]],
    completions: list[str] | None = None,
    log_location: str | None = None,
) -> None:
    global _step_counter, _last_time

    _step_counter += 1
    step = _step_counter
    step_label = f"{step}/{max_steps}" if max_steps else str(step)

    now = time.time()
    elapsed = f"{now - _last_time:.1f}s" if _last_time else "--"
    _last_time = now

    n = len(prompts)
    if not scores:
        logger.info("[step %s] %d samples | no scores | %s", step_label, n, elapsed)
        return

    scorer_names = sorted({k for s in scores for k in s})
    means = {
        name: sum(s.get(name, 0.0) for s in scores) / len(scores)
        for name in scorer_names
    }
    parts = " ".join(f"{name}={means[name]:+.2f}" for name in scorer_names)
    _summary.add_rollout_scores(step, scores)

    if verbose:
        logger.info("[step %s] %d samples | %s | %s", step_label, n, parts, elapsed)
        if log_location:
            logger.info("  eval log: %s", log_location)
        if completions:
            preview = completions[0][:120].replace("\n", " ")
            logger.info("  sample: %s", preview)
    else:
        logger.info("[step %s] %s | %s", step_label, parts, elapsed)


def log_eval(metrics: dict[str, float], step: int, elapsed: float) -> None:
    """One-liner for val eval results — shown in notebook + log file."""
    parts = " | ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
    logger.info("[eval %d] %s | %.1fs", step, parts, elapsed)
    _summary.add_val_eval(step, metrics)


def log_run_start(log_file: str | object) -> None:
    """Print the log file path so the user knows where to tail."""
    logger.info("Logging to %s", log_file)


def log_off_policy_decision(
    mode: str,
    depth: int,
    t_rollout_s: float | None = None,
    t_train_s: float | None = None,
) -> None:
    """Console line announcing the off-policy queue depth in effect.

    `mode` is "fixed" (user passed a positive depth), "auto" (calibrator picked
    one), or "auto-sync" (calibrator decided prefetch wouldn't help). When
    `t_rollout_s` / `t_train_s` are provided they explain the choice.
    """
    if mode == "auto-sync":
        logger.info(
            "[off-policy] auto-tune: trainer is the bottleneck — staying sync "
            "(T_rollout=%.1fs ≤ T_train=%.1fs)",
            t_rollout_s if t_rollout_s is not None else 0.0,
            t_train_s if t_train_s is not None else 0.0,
        )
        return
    extra = ""
    if t_rollout_s is not None and t_train_s is not None:
        extra = f" (T_rollout={t_rollout_s:.1f}s T_train={t_train_s:.1f}s)"
    logger.info("[off-policy] %s depth=%d%s", mode, depth, extra)


def log_run_summary(
    n_steps: int,
    total_s: float,
    avg_rollout_s: float | None,
    avg_train_s: float | None,
    prefetch_enabled: bool | None,
    discarded: int | None,
) -> None:
    """One-line summary printed when `inspect_rl_train` returns.

    `prefetch_enabled` is True/False if a decision was made (or forced), None if
    auto-mode never finished warmup. `discarded` counts rollouts the prefetcher
    completed but the trainer never popped — a non-zero value is healthy (the
    prefetcher was outpacing the trainer, exactly the regime the freshest-only
    design is for).

    We avoid a wall-clock speedup estimate — it'd compare against a synthetic
    `(R+T)*N` baseline that ignores warmup steps, baseline eval, and any
    irregular per-step costs, and the resulting ratio is easy to misread.
    """
    parts = [f"{n_steps} steps in {total_s:.1f}s"]
    if avg_rollout_s is not None and avg_train_s is not None:
        parts.append(f"rollout avg {avg_rollout_s:.1f}s · train avg {avg_train_s:.1f}s")
    if prefetch_enabled is None:
        parts.append("off-policy: warmup-only")
    elif prefetch_enabled is False:
        parts.append("off-policy: sync")
    else:
        msg = "off-policy: prefetch on"
        if discarded is not None and discarded > 0:
            msg += f" · {discarded} stale rollouts discarded"
        parts.append(msg)
    logger.info("[summary] %s", " · ".join(parts))

    for line in _summary.before_after_lines():
        logger.info("[summary] %s", line)
