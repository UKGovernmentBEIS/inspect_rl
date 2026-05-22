# 003 — Tier 2 design: freshest-rollout prefetch (single-slot)

The user-facing goal: vLLM never idles, and when the trainer is ready for a new rollout it picks up the freshest available one. Originally framed as "configurable depth N off-policy"; landed as a binary on/off because the depth knob doesn't increase throughput with our single-worker generation pipeline (see below).

## Reduces to a prefetch wrapper (not a trainer swap)

TRL 1.2.0 already implements truncated importance sampling (TIS) for off-policy correction. Walking the loss path in `grpo_trainer.py:2050-2079`:

```python
if self.use_vllm and self.vllm_importance_sampling_correction:
    per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask
    vllm_importance_sampling_ratio = torch.exp(logps_diff)
    # then clamp at vllm_importance_sampling_cap (default 3.0)
```

- `old_per_token_logps` = current-model forward pass over the rollout's tokens (computed at training time).
- `sampling_per_token_logps` = the logprobs our `rollout_func` already returns from vLLM.

So `exp(current_logp - sampling_logp)` IS the off-policy correction. TRL defaults: `vllm_importance_sampling_correction=True`, `mode="sequence_mask"`, `cap=3.0` (`grpo_config.py:792-820`). Already on; no loss-side changes needed.

**This is the discriminator the advisor flagged.** Result: Path B (prefetch wrapper around `rollout_func`) is mathematically correct at any staleness — TIS handles it, capped at `cap=3.0` for tail control. No custom loss, no `AsyncGRPOTrainer` swap.

## Why depth doesn't help with our single-worker design

With one prefetch thread and inspect_ai's process-wide eval_async lock, at most one rollout is being generated at any moment. A queue of depth N>1 just buffers N-1 already-stale results waiting to be popped FIFO. The trainer would always consume the *oldest* one, which is exactly the wrong policy for freshness.

Working through it:
- `T_rollout > T_train` (gen slow, our common case): with depth=1, by the time the trainer finishes a step, the producer is still mid-rollout. The trainer waits for it. Higher depth doesn't help because there's no spare bandwidth to generate more in parallel.
- `T_rollout < T_train` (trainer slow, big-model case the user cares about): with depth=1 FIFO, the producer finishes a rollout in `T_rollout`, writes it to the queue, then **idles** for `T_train - T_rollout` until the trainer pops. Bad — vLLM should be working. With depth=N FIFO, the producer keeps going and fills the queue with N stale rollouts, then idles. The trainer pops the oldest (~(N-1)·T_rollout staleness). Bad in a different way.
- Neither regime makes depth>1 useful for throughput; both make it worse for freshness.

The fix: don't queue. Have the producer overwrite a single slot every time it finishes. Trainer always gets the most-recent completed rollout; older completed-but-unread rollouts get discarded.

## What changes

1. **`FreshestPrefetchRolloutFunc`** — a wrapper around `make_inspect_rollout_func` that owns:
   - A single-slot register (`_latest` + `threading.Event` + lock).
   - A daemon `threading.Thread` running a `while not stopped: generate → overwrite slot` loop.
   - A parallel cursor over the same HF dataset (TRL re-derives prompt text from returned `prompt_ids`).
   - A `discarded` counter — rollouts the producer finished but no consumer popped (good telemetry: non-zero means the producer was outpacing the trainer).

2. **CLI flag** `--off-policy-steps` (default `-1` = auto):
   - `0`: fully sync; skip the wrapper entirely.
   - any other value: enable prefetch. The integer value isn't a depth; it just gates on/off.
   - `-1` (`auto`): run 3 warmup steps synchronously, measure `T_rollout` vs `T_train` from the gap between calls, then either enable prefetch (if `T_rollout > T_train`) or stay synchronous (if the trainer is already the bottleneck and vLLM would idle even with prefetch).

3. **No changes to `inspect_rl_train` signature** beyond an additional kwarg. The wrapper composes; the existing `rollout_func` is the inner generator.

## Prompt selection

The wrapper ignores TRL's per-call `prompts` and iterates its own cursor over the same HF dataset. TRL re-derives prompt text from the returned `prompt_ids` (grpo_trainer.py:2112), so the returned rollout is self-consistent — TRL doesn't compare against what it asked for. Tradeoff: trained-on order differs from TRL's, so per-prompt deterministic replays don't hold. Reward function payloads (`inspect_scores`) align by position with the returned tensors.

## Failure modes to detect

- **Stuck producer** — daemon thread crashes inside `inspect_eval`. The exception is caught + logged in `_producer_loop`, the loop exits, the consumer blocks forever on `_latest_ready.wait()`. Add a timeout on the wait if this turns out to bite in practice; for now we accept it.
- **Drift in IS ratio** — if `vllm_importance_sampling_ratio` saturates `vllm_importance_sampling_cap=3.0` (lots of tokens clamped), staleness exceeds the IS budget. Log the cap-hit fraction per step.
- **Weight-sync race** — TRL pushes weights to vLLM at the end of each step. The producer may be mid-generation when that happens — it'll see a fresher model than its IS denominator assumes. That's exactly what TIS corrects for; `sampling/sampling_logp_difference/mean` reports the magnitude.

## What this does NOT do

- No `AsyncGRPOTrainer` swap. We keep our `rollout_func`, our `trl_turn` metadata, our `env_mask` plumbing.
- No inflight KV-cache update on vLLM (OLMo 3's full async). vLLM still pauses for NCCL weight sync; we just keep generation running outside that window.
- No ESS scaling / OPOB (Tier 3). TIS+cap is the only off-policy correction; variance growth with staleness is on the user.
- No multi-worker generation. The throughput ceiling is `max(T_rollout, T_train)` per step with the current single-thread producer. Removing the `INSPECT_EVAL_LOCK` + raising `max_workers` would unlock more — separate exploration.

## Validation surface

- `math-agent` end-to-end with `--off-policy-steps -1` → confirm `[off-policy] auto …` decision line, `[summary] off-policy: prefetch on · N discarded` summary line.
- Hand-pick a workload where `T_train > T_rollout` (e.g. a much bigger model on a tiny prompt) to verify the producer discards stale rollouts and the consumer never blocks waiting.

## Estimate (actual, post-implementation)

~250 LOC across `prefetch.py` (wrapper + auto calibrator), `display.py` (console messages + summary), `trainer.py` (plumb-through + step-timer aggregation + cleanup hook), `cli.py` + 4 examples (flag), README + this doc. Main complexity ended up being the auto-calibrator + visibility plumbing rather than the prefetcher itself.
