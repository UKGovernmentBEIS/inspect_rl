"""Heartbeat thread for hang detection.

A daemon thread emits one log line every `interval_s` seconds with the current
training phase, step, and seconds elapsed in the phase. A monitoring agent (or
human) tailing train.log can treat absent heartbeat lines for >2 ticks as
evidence the trainer has wedged — most commonly inside `inspect_eval`'s
parse-error retry loop on bad tool-call JSON, or a stalled NCCL weight sync.

Phases set by the trainer / rollout / eval-callback:
- ``init``        — before trainer.train()
- ``rollout``     — inside the rollout_func (incl. active-sampling resamples)
- ``training``    — between rollout return and next rollout/eval (loss + opt step)
- ``eval``        — inside _InspectEvalCallback's held-out inspect_eval call
- ``shutdown``    — after trainer.train() returns

Output format: ``[heartbeat] step=N phase=PHASE elapsed_in_phase=Xs``
Single-line, grep-able, deterministic.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

_state_lock = threading.Lock()
_phase: str = "init"
_phase_started_ts: float = time.perf_counter()
_step: int | None = None

_stop_event: threading.Event | None = None
_thread: threading.Thread | None = None


def set_phase(phase: str, step: int | None = None) -> None:
    """Record a new training phase. Resets the in-phase timer."""
    global _phase, _phase_started_ts, _step
    with _state_lock:
        _phase = phase
        _phase_started_ts = time.perf_counter()
        if step is not None:
            _step = step


def _read_state() -> tuple[str, int | None, float]:
    with _state_lock:
        return _phase, _step, time.perf_counter() - _phase_started_ts


def _heartbeat_loop(interval_s: float, stop: threading.Event) -> None:
    while not stop.wait(interval_s):
        phase, step, elapsed = _read_state()
        step_str = f"step={step} " if step is not None else ""
        logger.info(
            "[heartbeat] %sphase=%s elapsed_in_phase=%.0fs",
            step_str,
            phase,
            elapsed,
        )


def start_heartbeat(interval_s: float = 30.0) -> None:
    """Start (or restart) the heartbeat daemon. Idempotent if already alive."""
    global _stop_event, _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop_event = threading.Event()
    _thread = threading.Thread(
        target=_heartbeat_loop,
        args=(interval_s, _stop_event),
        daemon=True,
        name="inspect-rl-heartbeat",
    )
    _thread.start()


def stop_heartbeat() -> None:
    """Stop the heartbeat thread; safe to call without a prior start."""
    global _stop_event, _thread
    if _stop_event is not None:
        _stop_event.set()
    _stop_event = None
    _thread = None
