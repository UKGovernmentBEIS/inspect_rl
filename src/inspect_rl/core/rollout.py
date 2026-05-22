"""Rollout function factory: bridges Inspect eval → TRL's RolloutFunc interface.

`make_inspect_rollout_func(task)` returns a callable that TRL's GRPOTrainer
invokes each training step. It runs an Inspect eval to generate completions
and score them, then returns the token-level data TRL needs for policy gradients.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
from inspect_ai import eval as inspect_eval, task_with
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai import Task
from inspect_ai.scorer import Score

from inspect_rl.util.display import log_rollout
from inspect_rl.util.heartbeat import set_phase
from inspect_rl.util.logging_config import capture_inspect_log

logger = logging.getLogger(__name__)

# RolloutFunc = Callable[[list[Any], GRPOTrainer], dict[str, Any]]
RolloutFunc = Any

# Shared timestamp set when rollout_func returns; _StepTimerCallback reads it
# to compute the weight-update-only slice of step time.
last_rollout_end_ts: float | None = None

# inspect_ai enforces a process-wide "no concurrent eval_async" check at
# _eval_async_inner. When PrefetchingRolloutFunc runs eval in a background
# thread while the validation callback runs eval on the main thread, both call
# inspect_eval() and the second one raises. Hold this around every inspect_eval
# call in our code paths so they serialise globally — the lock only blocks
# during the (rare) overlap between val-eval and a still-running prefetch.
INSPECT_EVAL_LOCK = threading.Lock()


class _RankGatedRolloutFunc:
    """Wrap a RolloutFunc so only rank 0 runs the eval; others wait for broadcast.

    TRL calls `rollout_func(prompts, trainer)` on every rank with that rank's
    local share of prompts. Stock TRL's vLLM path
    (`trl.generation.vllm_generation.generate`) handles this by gathering all
    prompts on rank 0, generating once, and broadcasting the result. We mirror
    that pattern so the underlying `inspect_eval` only runs once per step
    instead of being called by every rank against the same vLLM server.

    Each rank contributes its prompts via `gather_object`, rank 0 runs the
    eval over the full set, and the result is broadcast back. Each rank then
    slices `[process_index * local_size : (process_index + 1) * local_size]`.
    Single-rank (world_size==1) is a no-op fast path.

    Attribute lookups for `shutdown`, `discarded_count`, `prefetch_enabled`
    forward to the inner so trainer.py's summary code works unchanged.
    """

    def __init__(self, inner_fn: Any) -> None:
        self._inner_fn = inner_fn

    def __call__(self, prompts: list[Any], trainer: Any) -> dict[str, Any]:
        from accelerate.utils import broadcast_object_list, gather_object

        accelerator = trainer.accelerator
        if accelerator.num_processes == 1:
            return self._inner_fn(prompts, trainer)

        local_size = len(prompts)
        all_prompts = gather_object(prompts)

        if accelerator.is_main_process:
            try:
                payload: Any = self._inner_fn(all_prompts, trainer)
            except BaseException as exc:  # propagate so other ranks unblock
                payload = exc
        else:
            payload = None

        obj_list = [payload]
        broadcast_object_list(obj_list, from_process=0)
        full = obj_list[0]
        if isinstance(full, BaseException):
            raise full

        start = accelerator.process_index * local_size
        end = start + local_size
        sliced: dict[str, Any] = {}
        for key, value in full.items():
            if (
                isinstance(value, list)
                and len(value) == accelerator.num_processes * local_size
            ):
                sliced[key] = value[start:end]
            else:
                # scalars / paths shared across ranks (e.g. eval_log_path)
                sliced[key] = value
        return sliced

    def __getattr__(self, name: str) -> Any:
        # Forward shutdown / discarded_count / prefetch_enabled to the wrapped
        # chain so trainer.py's summary + cleanup code is rank-agnostic.
        return getattr(self._inner_fn, name)


def make_rank_gated_rollout_func(inner_fn: Any) -> _RankGatedRolloutFunc:
    return _RankGatedRolloutFunc(inner_fn)


def make_inspect_rollout_func(
    task: Task,
    vllm_base_url: str = "http://localhost:8000",
    eval_log_dir: str | None = None,
    max_resample_rounds: int = 0,
) -> RolloutFunc:
    """Create a TRL RolloutFunc from an Inspect Task.

    The returned function:
    1. Converts TRL's prompt dicts → Inspect Samples (restoring targets)
    2. Runs eval_async with the TRLVLLMProvider
    3. Extracts token IDs, logprobs, and scores from the eval results
    4. Returns the dict TRL expects

    Args:
        max_resample_rounds: When >0, detect prompt groups where all completions
            received identical scores (zero gradient) and regenerate them, up to
            this many rounds. OLMo 3's "active sampling" technique.
    """
    sample_lookup: dict[str, Sample] = {}
    for sample in task.dataset:
        if isinstance(sample.input, str):
            sample_lookup[sample.input] = sample
        else:
            for msg in reversed(sample.input):
                if msg.role == "user":
                    sample_lookup[msg.text or ""] = sample
                    break

    def _run_eval(
        samples: list[Sample],
        trainer: Any,
        step_idx: int,
        log_suffix: str = "rollout",
    ) -> dict[str, Any]:
        """Run inspect_eval on samples, return per-sample token data + scores."""
        tokenizer = trainer.processing_class
        model = get_model(
            f"trl-vllm/{trainer.model.config.name_or_path}",
            base_url=vllm_base_url,
            tokenizer=tokenizer,
            config=GenerateConfig(
                max_tokens=trainer.max_completion_length,
                temperature=trainer.args.temperature,
                top_p=getattr(trainer.args, "top_p", 1.0),
                top_k=getattr(trainer.args, "top_k", -1),
            ),
            batch_timeout=1.0,
            max_batch_size=len(samples),
        )

        eval_t = task_with(
            task,
            dataset=MemoryDataset(samples=samples),
            epochs=1,
        )

        eval_kwargs: dict[str, Any] = {}
        step_log_dir = None
        if eval_log_dir:
            step_log_dir = Path(eval_log_dir) / f"{step_idx:03d}_{log_suffix}"
            step_log_dir.mkdir(parents=True, exist_ok=True)
            eval_kwargs["log_dir"] = str(step_log_dir)

        with INSPECT_EVAL_LOCK, capture_inspect_log(step_log_dir):
            logs = inspect_eval(
                eval_t,
                model=model,
                max_samples=len(samples) * 2,
                display="log",
                log_level="warning",
                fail_on_error=True,
                retry_on_error=3,
                **eval_kwargs,
            )

        prompt_ids: list[list[int]] = []
        completion_ids: list[list[int]] = []
        logprobs_out: list[list[float]] = []
        env_masks: list[list[int]] = []
        per_sample_scores: list[dict[str, float]] = []

        log = logs[0]
        assert log.samples is not None
        samples_by_id: dict[str, Any] = {}
        for s in log.samples:
            samples_by_id[str(s.id)] = s

        # iterate by input order, look up by the sample's own id — resample subsets
        # keep their original ids (e.g. "4".."7"), so range(len(samples)) misses them.
        for sample_in in samples:
            sample = samples_by_id[str(sample_in.id)]
            turn_data = _collect_turn_data(sample)

            if turn_data:
                p_ids, c_ids, lps, mask = _aggregate_turns(turn_data)
            else:
                trl_data = sample.output.metadata.get("trl_completion_data", {})
                p_ids = trl_data.get("prompt_ids", [])
                c_ids = trl_data.get("completion_ids", [])
                lps = trl_data.get("logprobs", [])
                mask = [1] * len(c_ids)

            prompt_ids.append(p_ids)
            completion_ids.append(c_ids)
            logprobs_out.append(lps)
            env_masks.append(mask)

            sample_scores: dict[str, float] = {}
            if sample.scores:
                for scorer_name, score in sample.scores.items():
                    sample_scores[scorer_name] = _score_to_float(score)
            per_sample_scores.append(sample_scores)

        if hasattr(model, "aclose"):
            try:
                asyncio.run(model.aclose())
            except RuntimeError:
                pass

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs_out,
            "env_masks": env_masks,
            "inspect_scores": per_sample_scores,
            "eval_log_path": str(step_log_dir) if step_log_dir else None,
        }

    def rollout_func(prompts: list[Any], trainer: Any) -> dict[str, Any]:
        t0 = time.perf_counter()
        step_idx = int(getattr(getattr(trainer, "state", None), "global_step", 0)) + 1
        samples = _prompts_to_samples(prompts, sample_lookup)
        logger.info("[rollout %d] starting: %d samples", step_idx, len(samples))
        set_phase("rollout", step=step_idx)

        raw = _run_eval(samples, trainer, step_idx)

        # --- Active sampling: regenerate zero-gradient groups ---
        num_gen = getattr(trainer, "num_generations", 1)
        if max_resample_rounds > 0 and num_gen > 1 and len(samples) >= num_gen:
            total_resampled = 0
            num_groups = len(samples) // num_gen
            for round_idx in range(max_resample_rounds):
                zero_grad = _find_zero_gradient_groups(raw["inspect_scores"], num_gen)
                if not zero_grad:
                    break

                resample_indices: list[int] = []
                for g in zero_grad:
                    resample_indices.extend(range(g * num_gen, (g + 1) * num_gen))
                resample_samples = [samples[i] for i in resample_indices]

                resample_raw = _run_eval(
                    resample_samples,
                    trainer,
                    step_idx,
                    log_suffix=f"resample{round_idx + 1}",
                )

                for key in [
                    "prompt_ids",
                    "completion_ids",
                    "logprobs",
                    "env_masks",
                    "inspect_scores",
                ]:
                    for j, idx in enumerate(resample_indices):
                        raw[key][idx] = resample_raw[key][j]

                total_resampled += len(zero_grad)
                logger.info(
                    "[rollout %d] resample round %d: regenerated %d/%d groups",
                    step_idx,
                    round_idx + 1,
                    len(zero_grad),
                    num_groups,
                )

            # Log final zero-gradient stats even when not resampling
            remaining_zero = len(
                _find_zero_gradient_groups(raw["inspect_scores"], num_gen)
            )
            if total_resampled > 0 or remaining_zero > 0:
                logger.info(
                    "[rollout %d] active sampling: resampled %d groups total, "
                    "%d/%d groups still zero-gradient",
                    step_idx,
                    total_resampled,
                    remaining_zero,
                    num_groups,
                )
        elif num_gen > 1 and len(samples) >= num_gen:
            zero_grad = _find_zero_gradient_groups(raw["inspect_scores"], num_gen)
            if zero_grad:
                num_groups = len(samples) // num_gen
                logger.info(
                    "[rollout %d] %d/%d groups have zero gradient (all identical scores)",
                    step_idx,
                    len(zero_grad),
                    num_groups,
                )

        eval_log_path = raw["eval_log_path"]
        env_masks = raw["env_masks"]
        per_sample_scores = raw["inspect_scores"]

        result: dict[str, Any] = {
            "prompt_ids": raw["prompt_ids"],
            "completion_ids": raw["completion_ids"],
            "logprobs": raw["logprobs"],
        }
        if any(0 in mask for mask in env_masks):
            result["env_mask"] = env_masks
        if per_sample_scores:
            result["inspect_scores"] = per_sample_scores

        completions = (
            [
                trainer.processing_class.decode(cids, skip_special_tokens=True)
                for cids in raw["completion_ids"]
            ]
            if hasattr(trainer, "processing_class")
            else None
        )

        log_rollout(
            prompts=prompts,
            scores=per_sample_scores,
            completions=completions,
            log_location=eval_log_path,
        )

        global last_rollout_end_ts
        last_rollout_end_ts = time.perf_counter()
        logger.info(
            "[rollout %d] done in %.1fs → %s",
            step_idx,
            last_rollout_end_ts - t0,
            eval_log_path or "?",
        )
        set_phase("training", step=step_idx)
        return result

    return rollout_func


def _find_zero_gradient_groups(
    scores: list[dict[str, float]],
    num_generations: int,
) -> list[int]:
    """Return indices of prompt groups where all completions have identical scores.

    These groups produce zero advantage and zero gradient — wasted compute.
    """
    num_groups = len(scores) // num_generations
    zero_grad: list[int] = []
    for g in range(num_groups):
        group = scores[g * num_generations : (g + 1) * num_generations]
        if len(group) < 2:
            continue
        first = group[0]
        if all(s == first for s in group[1:]):
            zero_grad.append(g)
    return zero_grad


def _prompts_to_samples(
    prompts: list[Any],
    sample_lookup: dict[str, Sample],
) -> list[Sample]:
    samples = []
    for i, prompt_messages in enumerate(prompts):
        input_messages = []
        target = None
        metadata: dict[str, Any] = {}

        for msg in prompt_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                input_messages.append(ChatMessageSystem(content=content))
            elif role == "user":
                input_messages.append(ChatMessageUser(content=content))
                orig = sample_lookup.get(content)
                if orig is not None:
                    target = orig.target
                    metadata = dict(orig.metadata) if orig.metadata else {}

        samples.append(
            Sample(
                input=input_messages,
                id=str(i),
                target=target or "",
                metadata=metadata,
            )
        )
    return samples


def _collect_turn_data(sample: Any) -> list[dict[str, Any]]:
    """Pull per-turn token data off assistant messages in a finished sample.

    TRLVLLMProvider stashes `{prompt_ids, completion_ids, logprobs}` on each
    assistant message's `metadata["trl_turn"]`. basic_agent appends those
    messages to state.messages, so they survive into the eval log.
    """
    turns: list[dict[str, Any]] = []
    messages = getattr(sample, "messages", None) or []
    for m in messages:
        if isinstance(m, ChatMessageAssistant) and m.metadata:
            turn = m.metadata.get("trl_turn")
            if turn and turn.get("completion_ids"):
                turns.append(turn)
    return turns


def _aggregate_turns(
    turns: list[dict[str, Any]],
) -> tuple[list[int], list[int], list[float], list[int]]:
    """Concatenate per-turn tokens into one contiguous sequence + env_mask.

    For turn n, prompt_ids_{n+1} must extend (prompt_ids_n + completion_ids_n)
    with the tokens the environment injected between turns (tool response +
    any follow-up user message). We extract that delta directly rather than
    re-tokenizing. env_mask is 1 for model-generated tokens, 0 for environment
    tokens; TRL multiplies it against the completion mask during the loss.
    """
    base_prompt: list[int] = list(turns[0]["prompt_ids"])
    completion: list[int] = []
    logprobs: list[float] = []
    mask: list[int] = []

    expected_prefix = list(base_prompt)
    for i, turn in enumerate(turns):
        p_ids = list(turn["prompt_ids"])
        c_ids = list(turn["completion_ids"])
        lps = list(turn["logprobs"])

        if i == 0:
            assert p_ids == base_prompt, "first turn's prompt must equal base prompt"
        else:
            if p_ids[: len(expected_prefix)] != expected_prefix:
                # find first divergence for a more useful error
                div = next(
                    (
                        k
                        for k in range(min(len(expected_prefix), len(p_ids)))
                        if p_ids[k] != expected_prefix[k]
                    ),
                    min(len(expected_prefix), len(p_ids)),
                )
                raise RuntimeError(
                    f"turn {i} prompt_ids do not extend prior prompt+completion; "
                    f"tokenizer/template drift between turns. "
                    f"first divergence at token {div}/{len(expected_prefix)}. "
                    f"expected[{max(0, div - 3)}:{div + 3}]="
                    f"{expected_prefix[max(0, div - 3) : div + 3]}, "
                    f"got[{max(0, div - 3)}:{div + 3}]={p_ids[max(0, div - 3) : div + 3]}."
                )
            delta = p_ids[len(expected_prefix) :]
            completion.extend(delta)
            logprobs.extend([0.0] * len(delta))
            mask.extend([0] * len(delta))

        completion.extend(c_ids)
        logprobs.extend(lps)
        mask.extend([1] * len(c_ids))
        expected_prefix = p_ids + c_ids

    return base_prompt, completion, logprobs, mask


def _score_to_float(score: Score) -> float:
    v = score.value
    if isinstance(v, (int, float)):
        return float(v)
    if v == "C":
        return 1.0
    if v == "I":
        return 0.0
    return 0.0


def inspect_dataset_to_hf(dataset: Dataset, limit: int | None = None) -> HFDataset:
    """Convert an Inspect Dataset to HuggingFace format for TRL's dataloader."""
    dlist = []
    items = list(dataset)
    if limit is not None:
        items = items[:limit]
    for sample in items:
        if isinstance(sample.input, str):
            prompt = [{"role": "user", "content": sample.input}]
        else:
            prompt = [
                {
                    "role": s.role,
                    "content": s.text or s.content
                    if hasattr(s, "text")
                    else str(s.content),
                }
                for s in sample.input
            ]
        dlist.append({"prompt": prompt, "answer": sample.target or ""})
    return HFDataset.from_list(dlist)
