"""Run summary state — before/after stats and report.md generation.

Two responsibilities, kept in one module because they share the same buffer of
per-step scorer means + per-eval metrics:

1. ``add_rollout_scores(step, scores)`` / ``add_val_eval(step, metrics)`` —
   called by ``util.display`` as each rollout / val eval lands. We keep a
   compact dict of arrays per scorer.

2. ``before_after_lines()`` returns short ``"name baseline → final (Δ)"`` strings
   that the run-end ``[summary]`` lines splices in.

3. ``write_report(run_dir, …)`` drops a ``report.md`` in the run directory
   with summary deltas, full per-step history (HTML tables, render natively
   on GitHub), and relative links to the rest of the artifact tree.

Buffers are module-level (one trainer per process), so this module is a
glorified namespace, not a class. Cleared by ``reset_state`` at the start of
each ``inspect_rl_train``.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# {scorer_name: [(step, mean_value), …]} for the train-rollout side.
_rollout_history: dict[str, list[tuple[int, float]]] = {}
# {metric_name: [(step, value), …]} for held-out val evals.
_eval_history: dict[str, list[tuple[int, float]]] = {}


def reset_state() -> None:
    """Forget per-run buffers — called at the start of each `inspect_rl_train`."""
    _rollout_history.clear()
    _eval_history.clear()


def add_rollout_scores(step: int, scores: list[dict[str, float]]) -> None:
    """Record the per-scorer mean of one training rollout."""
    if not scores:
        return
    names = {k for s in scores for k in s}
    for name in names:
        mean = sum(s.get(name, 0.0) for s in scores) / len(scores)
        _rollout_history.setdefault(name, []).append((step, mean))


def add_val_eval(step: int, metrics: dict[str, float]) -> None:
    """Record one ``eval/<scorer>/<metric>`` point per held-out evaluation."""
    for name, value in metrics.items():
        _eval_history.setdefault(name, []).append((step, value))


def _window_mean(series: list[tuple[int, float]], window: int) -> float | None:
    if not series:
        return None
    n = min(window, len(series))
    return sum(v for _, v in series[-n:]) / n


def _first_window_mean(series: list[tuple[int, float]], window: int) -> float | None:
    if not series:
        return None
    n = min(window, len(series))
    return sum(v for _, v in series[:n]) / n


def before_after_lines(window: int = 5) -> list[str]:
    """Compact ``name first-N → last-N (Δ)`` strings for the [summary] line.

    Two blocks: held-out val metrics (more meaningful, higher-signal) followed
    by per-step rollout means (noisier but always available).
    """
    out: list[str] = []
    for name in sorted(_eval_history):
        series = _eval_history[name]
        if not series:
            continue
        first = series[0][1]
        last = series[-1][1]
        out.append(f"{name} {first:+.3f}→{last:+.3f} (Δ{last - first:+.3f})")
    for name in sorted(_rollout_history):
        series = _rollout_history[name]
        if len(series) < 2:
            continue
        first = _first_window_mean(series, window)
        last = _window_mean(series, window)
        if first is None or last is None:
            continue
        out.append(
            f"rollout/{name} first{min(window, len(series))}={first:+.3f}→"
            f"last{min(window, len(series))}={last:+.3f} (Δ{last - first:+.3f})"
        )
    return out


def _fmt(v: float) -> str:
    return f"{v:+.3f}"


def _summary_table(window: int) -> str:
    """HTML table with one row per scorer: baseline / final / delta.

    Val rows use absolute baseline (step 0) → final. Rollout rows use windowed
    means (first-N vs last-N) because per-step rollout values are too noisy at
    a single step to call a "baseline".
    """
    rows: list[str] = []
    for name in sorted(_eval_history):
        series = _eval_history[name]
        if not series:
            continue
        first, last = series[0][1], series[-1][1]
        rows.append(
            f"<tr><td><code>{name}</code></td><td>{_fmt(first)}</td>"
            f"<td>{_fmt(last)}</td><td>{_fmt(last - first)}</td></tr>"
        )
    for name in sorted(_rollout_history):
        series = _rollout_history[name]
        if len(series) < 2:
            continue
        first = _first_window_mean(series, window)
        last = _window_mean(series, window)
        if first is None or last is None:
            continue
        n = min(window, len(series))
        rows.append(
            f"<tr><td><code>rollout/{name}</code> (first-{n} → last-{n})</td>"
            f"<td>{_fmt(first)}</td><td>{_fmt(last)}</td>"
            f"<td>{_fmt(last - first)}</td></tr>"
        )
    if not rows:
        return "<p><i>(no scorer history captured — single-step run?)</i></p>"
    return (
        "<table>\n"
        "<tr><th>metric</th><th>baseline</th><th>final</th><th>Δ</th></tr>\n"
        + "\n".join(rows)
        + "\n</table>"
    )


def _history_table(
    history: dict[str, list[tuple[int, float]]],
    step_col: str = "step",
    name_prefix: str = "",
) -> str:
    """HTML table with one row per step, one column per scorer. Steps that
    have a value for at least one column are included; missing cells render
    as an em-dash."""
    if not history:
        return "<p><i>(none)</i></p>"

    names = sorted(history)
    # Union of steps in step order.
    step_to_row: dict[int, dict[str, float]] = {}
    for name, series in history.items():
        for step, value in series:
            step_to_row.setdefault(step, {})[name] = value
    sorted_steps = sorted(step_to_row)

    header = (
        f"<tr><th>{step_col}</th>"
        + "".join(f"<th><code>{name_prefix}{n}</code></th>" for n in names)
        + "</tr>"
    )
    body = []
    for step in sorted_steps:
        row = step_to_row[step]
        cells = "".join(
            f"<td>{_fmt(row[n])}</td>" if n in row else "<td>—</td>" for n in names
        )
        body.append(f"<tr><td>{step}</td>{cells}</tr>")
    return "<table>\n" + header + "\n" + "\n".join(body) + "\n</table>"


def write_report(
    run_dir: Path | str,
    *,
    model: str,
    max_steps: int,
    n_steps_run: int,
    total_s: float,
    avg_rollout_s: float | None,
    avg_train_s: float | None,
    prefetch_enabled: bool | None,
    discarded: int | None,
    wandb_run_id: str | None = None,
    window: int = 5,
) -> Path:
    """Write a ``report.md`` in the run directory.

    Uses HTML for tables (renders natively on GitHub, in code editors, and in
    most markdown viewers) but keeps the surrounding structure as markdown so
    it's still useful at the command line.
    """
    run_path = Path(run_dir).resolve()
    report = run_path / "report.md"

    eval_logs = run_path / "eval_logs"
    rollout_dirs = sorted(eval_logs.glob("*_rollout")) if eval_logs.exists() else []
    val_dirs = sorted(eval_logs.glob("*_val")) if eval_logs.exists() else []
    checkpoints = (
        sorted(
            (run_path / "checkpoints").glob("checkpoint-*"),
            key=lambda p: (
                int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1
            ),
        )
        if (run_path / "checkpoints").exists()
        else []
    )

    def _maybe(label: str, path: Path) -> str | None:
        if path.exists():
            return f"- [{label}]({path.name})"
        return None

    artifact_links = [
        link
        for link in (
            _maybe("manifest.json", run_path / "manifest.json"),
            _maybe("train.log", run_path / "train.log"),
        )
        if link is not None
    ]
    if checkpoints:
        artifact_links.append(
            f"- latest checkpoint: [`{checkpoints[-1].name}`](checkpoints/{checkpoints[-1].name})"
            f" (of {len(checkpoints)})"
        )
    if val_dirs:
        artifact_links.append(
            f"- val eval logs: `eval_logs/*_val/` ({len(val_dirs)} runs;"
            f" baseline=[`{val_dirs[0].name}`](eval_logs/{val_dirs[0].name}),"
            f" final=[`{val_dirs[-1].name}`](eval_logs/{val_dirs[-1].name}))"
        )
    if rollout_dirs:
        artifact_links.append(
            f"- rollout logs: `eval_logs/*_rollout/` ({len(rollout_dirs)} steps;"
            f" final=[`{rollout_dirs[-1].name}`](eval_logs/{rollout_dirs[-1].name}))"
        )

    timing_bits = [f"{n_steps_run} steps in {total_s:.1f}s"]
    if avg_rollout_s is not None and avg_train_s is not None:
        timing_bits.append(
            f"rollout avg {avg_rollout_s:.1f}s · train avg {avg_train_s:.1f}s"
        )
    if prefetch_enabled is None:
        timing_bits.append("off-policy: warmup-only")
    elif prefetch_enabled is False:
        timing_bits.append("off-policy: sync")
    else:
        msg = "off-policy: prefetch on"
        if discarded is not None and discarded > 0:
            msg += f" · {discarded} stale rollouts discarded"
        timing_bits.append(msg)

    parts = [
        f"# Run report — `{run_path.name}`",
        "",
        f"- model: `{model}`",
        f"- planned steps: {max_steps}; completed: {n_steps_run}",
        f"- runtime: {' · '.join(timing_bits)}",
    ]
    if wandb_run_id:
        parts.append(f"- wandb run ID: `{wandb_run_id}`")

    parts.extend(
        [
            "",
            "## Did the model learn?",
            "",
            _summary_table(window),
            "",
            "## Val eval history",
            "",
            _history_table(_eval_history),
            "",
            "## Rollout history",
            "",
            _history_table(_rollout_history),
            "",
            "## Artifacts",
            "",
            *artifact_links,
            "",
        ]
    )
    report.write_text("\n".join(parts))
    logger.info("[summary] wrote %s", report)
    return report
