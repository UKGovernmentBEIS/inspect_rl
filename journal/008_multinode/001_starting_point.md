# 001 — Starting point for multinode work

Snapshot of the codebase entering the multinode chapter, written for post-compaction context.

## What just landed (uncommitted to origin)

Two branches, neither pushed yet:

**`feat/tier1-grpo-mods`** — updates existing **PR #19**, 2 new commits on top of `dafb07e`:
- `d3e6217` `fix: resample loop crashes when zero-grad group is non-first`. T1's active-sampling loop in `rollout.py:_run_eval` indexed `samples_by_id` by `str(i)` (0…N-1) which only matches the initial rollout — on resample subsets the original ids (e.g. "4".."7") carry over and the lookup raised `KeyError`. Reproducible on math-agent step 7 with `--resample-rounds 3`.
- `2e8b1ec` `docs: README features section for active sampling + asymmetric clipping`.

**`feat/tier2-off-policy-prefetch`** — stacked on T1, 6 commits, see `journal/007_faster_rl/003_t2_design.md` for the design narrative.
- `4e85664` initial queue+depth scaffold (later replaced).
- `587c49d` shared `INSPECT_EVAL_LOCK` to serialise prefetcher + val-eval callback (inspect_ai enforces process-wide single eval_async; without the lock the final val eval crashes when a prefetch is mid-flight).
- `906c1d1` `--off-policy-steps -1` auto sentinel.
- `aacbcd8` auto-tune as default + `[off-policy]` / `[summary]` console lines + `_StepTimerCallback` aggregates.
- `07308b1` **freshest-rollout redesign**: single-slot producer thread, vLLM never idles, trainer always pops the freshest available rollout. Older completed rollouts get discarded (tracked by `discarded_count`). README mermaid diagram for the system architecture.
- `34c6885` prune completed work from README "Subsequent work".

Validation: math-agent 5-step smoke with the freshest design completed clean — `[off-policy] auto depth=1 (T_rollout=15.2s T_train=2.3s)`, `[summary] 5 steps in 314.6s · rollout avg 16.1s · train avg 0.8s · off-policy: prefetch on`. Per-step rewards are noise on this scale (smoke, not learning verification). The earlier 20-step T1 validation showed correctness 25% → 37.5% on a 32-sample heldout, no instability.

## Outstanding cleanup (small)

- `[step N+k]` overcount: `inspect_rl.display._step_counter` is global and increments on every inner-rollout completion, including background prefetches that finish *after* training ends. Cosmetic only. Real fix: pass a `log=False` flag through `_run_eval` when the prefetcher invokes it, or have the prefetcher use a separate counter. ~10 LOC.
- T2 has not been run for ≥20 steps to verify reward curves match T1 baseline on math-agent. The 5-step smoke just verifies the pipeline runs.

## What multinode actually needs

The README still lists `--num-processes` as a "forward-compatibility stub" and `_apply_train_env` raises `BadParameter` on anything > 1. The pieces:

1. **Rank-gate the rollout.** `rollout_func` and the `_InspectEvalCallback` currently both call `inspect_eval` on every rank — multi-rank would mean every trainer rank independently bombards vLLM. The right shape: only rank 0 generates, then broadcasts `prompt_ids` / `completion_ids` / `logprobs` to other ranks via `accelerate.utils.broadcast_object_list` or similar. TRL's stock GRPO path already does this gating internally; ours doesn't because the `RolloutFunc` extension point isn't rank-aware by default.

2. **vLLM placement decision.** Single-node multi-GPU is straightforward: vLLM on its own GPU(s), trainer accelerate-launched across the rest. Multi-node requires picking: one shared vLLM (TP across one node's GPUs, NCCL weight sync over the cluster fabric to all trainer nodes), or per-node colocated vLLM (`vllm_mode="colocate"`, but currently incompatible with Inspect's async solver chain — see "Where this could be simpler" in the README).

3. **Drop the `_apply_train_env` guard** once 1+2 work, and either remove the `--num-processes` flag or wire it to actually shell out to `accelerate launch`.

4. **Compose with T2's freshest prefetch.** The prefetch thread also needs to be rank-aware: only rank 0 should run the producer; other ranks pop from a broadcast queue. This means making `FreshestPrefetchRolloutFunc` rank-conditional at `__init__`.

5. **A real tutorial in the README.** Once 1-4 work and have been smoke-tested on the Slurm HPC 4-GPU setup, add a "Multi-GPU and multi-node" section before "Writing your own task". The user explicitly asked for this and was OK with me deferring until the wiring exists.

## Recommended first step after compaction

Verify what TRL's existing GRPO multi-rank handling assumes about `rollout_func` — does it already rank-gate the call, or does each rank invoke it? Read `trl/trainer/grpo_trainer.py` around the `_generate` path on a multi-rank run. That single answer determines whether step 1 above is "add `accelerator.is_main_process` guards" or "build a broadcast layer ourselves."

Don't write README docs before the smoke test passes — both the user and I agree on this.
