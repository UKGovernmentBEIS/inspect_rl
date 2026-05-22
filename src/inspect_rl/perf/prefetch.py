"""Prefetch wrapper that keeps vLLM generating while the trainer trains.

The trainer is idle while vLLM generates and vLLM is idle while the trainer
updates weights. `FreshestPrefetchRolloutFunc` breaks that lock-step by running
a background thread that continuously regenerates rollouts and keeps only the
freshest one in a single slot. The trainer always pops the most recent rollout
when it's ready; older completed rollouts get discarded.

Why a single slot instead of a queue:
- Single-worker generation (ThreadPoolExecutor(max_workers=1) + the process-wide
  inspect_eval lock) means at most one rollout runs at a time anyway. A queue of
  depth N would just hold N-1 stale rollouts waiting to be popped FIFO — the
  trainer would consume the oldest, not the freshest.
- The point of prefetching is to keep vLLM busy whenever the trainer is busy,
  so when the trainer is ready it grabs the most-recently-completed result.
  Staleness is bounded by one rollout duration (≤ T_rollout) rather than by
  queue depth × T_rollout.

Correctness: TRL's GRPOTrainer applies truncated importance sampling on the
(current_logp - sampling_logp) difference (grpo_trainer.py:2050-2079) with
`vllm_importance_sampling_cap=3.0` on by default. The `logprobs` our rollout
returns become the IS denominator; staleness up to the cap is corrected.

The wrapper ignores TRL's per-call `prompts` and iterates its own cursor over
the same HF dataset. TRL re-derives prompt text from the returned `prompt_ids`
(grpo_trainer.py:2112), so this is self-consistent — but trained-on order
differs from TRL's, so per-prompt deterministic replays are not preserved.

`AutoCalibratingRolloutFunc` is the default entry point. It runs three warmup
steps synchronously, measures T_rollout vs T_train, and either enables the
freshest prefetcher or stays synchronous if the trainer is already the
bottleneck (T_rollout ≤ T_train → vLLM would have to idle anyway).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from datasets import Dataset as HFDataset

logger = logging.getLogger(__name__)


class FreshestPrefetchRolloutFunc:
    """Background producer that keeps the slot filled with the freshest rollout.

    Producer loop:
        while not stopped: generate → atomic write to slot → loop.
    Consumer (trainer's `rollout_func` call):
        wait until slot is populated, take it atomically, return.

    The producer never sleeps on the consumer — it immediately starts the next
    rollout after writing. So vLLM stays busy. When the trainer pops, it gets
    whatever the producer most recently finished — at worst, a rollout that
    started no earlier than T_rollout ago.
    """

    def __init__(
        self,
        inner_fn: Any,  # RolloutFunc
        hf_dataset: HFDataset,
        batch_size: int,
        num_generations: int,
        world_size: int = 1,
    ) -> None:
        self._inner_fn = inner_fn
        self._dataset = hf_dataset
        self._batch_size = batch_size
        self._num_generations = num_generations
        self._world_size = world_size
        self._latest: dict[str, Any] | None = None
        self._latest_ready = threading.Event()
        self._slot_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._trainer: Any = None
        self._cursor = 0
        self._discarded = 0  # rollouts overwritten before the consumer popped

    def __call__(self, prompts: list[Any], trainer: Any) -> dict[str, Any]:
        # First call runs synchronously — no producer yet. Captures the trainer
        # reference for the producer thread, which we then start.
        if self._thread is None:
            self._trainer = trainer
            result = self._inner_fn(prompts, trainer)
            self._thread = threading.Thread(
                target=self._producer_loop,
                name="inspect-rl-prefetch",
                daemon=True,
            )
            self._thread.start()
            return result

        # Steady state: block until the producer has written a fresh rollout.
        self._latest_ready.wait()
        with self._slot_lock:
            result = self._latest
            self._latest = None
            self._latest_ready.clear()
        assert result is not None, "slot was empty after _latest_ready"
        return result

    def _producer_loop(self) -> None:
        while not self._stop.is_set():
            try:
                next_prompts = self._next_batch()
                result = self._inner_fn(next_prompts, self._trainer)
            except Exception:
                logger.exception("Prefetch producer crashed — stopping background loop")
                return
            with self._slot_lock:
                if self._latest is not None:
                    self._discarded += 1
                self._latest = result
                self._latest_ready.set()
            # No sleep: immediately start the next rollout so vLLM never idles.

    def _next_batch(self) -> list[list[dict[str, Any]]]:
        """Build one rollout batch from the parallel cursor.

        Replicates each unique prompt `num_generations` times to match TRL's
        RepeatSampler convention. In multi-rank runs the producer only lives on
        rank 0 and must generate `world_size * batch_size` prompts so the
        rank-gated wrapper can slice each rank's share.
        """
        n = len(self._dataset)
        total = self._batch_size * self._world_size
        unique = max(1, total // self._num_generations)
        batch: list[list[dict[str, Any]]] = []
        for _ in range(unique):
            row = self._dataset[self._cursor % n]
            self._cursor += 1
            for _ in range(self._num_generations):
                batch.append(row["prompt"])
        return batch

    def shutdown(self) -> None:
        self._stop.set()
        self._latest_ready.set()  # unblock any waiter
        # Daemon thread — don't join. It'll exit when the inner_fn returns from
        # whatever it's mid-doing, or when the interpreter exits.

    @property
    def discarded_count(self) -> int:
        return self._discarded


class AutoCalibratingRolloutFunc:
    """Run synchronously for `warmup_steps`, measure timings, then upgrade.

    The first `warmup_steps + 1` calls run the inner rollout_func directly,
    timing rollout duration and the gap-between-calls (≈ weight update). After
    we have enough measurements we decide:

      T_rollout > T_train → upgrade to FreshestPrefetchRolloutFunc.
      T_rollout ≤ T_train → trainer is already the bottleneck; vLLM would have
                            to idle every step regardless. Stay synchronous.
    """

    def __init__(
        self,
        inner_fn: Any,
        hf_dataset: HFDataset,
        batch_size: int,
        num_generations: int,
        warmup_steps: int = 3,
        world_size: int = 1,
    ) -> None:
        self._inner_fn = inner_fn
        self._dataset = hf_dataset
        self._batch_size = batch_size
        self._num_generations = num_generations
        self._warmup_steps = warmup_steps
        self._world_size = world_size
        self._t_rollouts: list[float] = []
        self._t_trains: list[float] = []
        self._last_call_start: float | None = None
        self._upgraded: FreshestPrefetchRolloutFunc | None = None
        self._enabled_after_warmup: bool | None = None

    def __call__(self, prompts: list[Any], trainer: Any) -> dict[str, Any]:
        if self._upgraded is not None:
            return self._upgraded(prompts, trainer)

        call_start = time.perf_counter()
        if self._last_call_start is not None and self._t_rollouts:
            t_train = call_start - self._last_call_start - self._t_rollouts[-1]
            if t_train > 0:
                self._t_trains.append(t_train)
        self._last_call_start = call_start

        t0 = time.perf_counter()
        result = self._inner_fn(prompts, trainer)
        self._t_rollouts.append(time.perf_counter() - t0)

        if len(self._t_trains) >= self._warmup_steps:
            self._upgrade(trainer)

        return result

    def _upgrade(self, trainer: Any) -> None:
        from inspect_rl.util.display import log_off_policy_decision

        rollout_mean = sum(self._t_rollouts) / len(self._t_rollouts)
        train_mean = sum(self._t_trains) / len(self._t_trains)
        enable_prefetch = rollout_mean > train_mean and train_mean > 0

        logger.info(
            "Auto off-policy calibration: T_rollout=%.1fs T_train=%.1fs → %s",
            rollout_mean,
            train_mean,
            "prefetch" if enable_prefetch else "stay sync",
        )

        if not enable_prefetch:
            log_off_policy_decision("auto-sync", 0, rollout_mean, train_mean)
            self._enabled_after_warmup = False

            class _SyncPassthrough:
                def __init__(self, fn: Any) -> None:
                    self._fn = fn

                def __call__(self, p: list[Any], t: Any) -> dict[str, Any]:
                    return self._fn(p, t)

                def shutdown(self) -> None:
                    pass

                @property
                def discarded_count(self) -> int:
                    return 0

            self._upgraded = _SyncPassthrough(self._inner_fn)  # type: ignore[assignment]
            return

        log_off_policy_decision("auto", 1, rollout_mean, train_mean)
        self._enabled_after_warmup = True
        self._upgraded = FreshestPrefetchRolloutFunc(
            inner_fn=self._inner_fn,
            hf_dataset=self._dataset,
            batch_size=self._batch_size,
            num_generations=self._num_generations,
            world_size=self._world_size,
        )

    def shutdown(self) -> None:
        if self._upgraded is not None:
            self._upgraded.shutdown()

    def prefetch_enabled(self) -> bool | None:
        return self._enabled_after_warmup

    def discarded_count(self) -> int:
        return getattr(self._upgraded, "discarded_count", 0) if self._upgraded else 0
