# 009/002 — H200 8-GPU DDP smoke (1 vLLM + 7 trainer ranks)

First end-to-end of [`examples/configs/hpc2/h200/slurm_8gpu.sbatch`](../../examples/configs/hpc2/h200/slurm_8gpu.sbatch) — the full-node variant that saturates the exclusive 8-GPU H200 reservation. Job 1615870, `tldr` + `Qwen2.5-0.5B-Instruct`, 10 steps requested, W&B on.

W&B run logged but not linked here (sanitised from internal entity).

**Result:** completed 5 steps, then crashed at step 6 with `KeyError: 'inspect_scores'` — see [Bug: prefetch path drops `inspect_scores`](#bug-prefetch-path-drops-inspect_scores) below. Reward signal in the 5 completed steps was real (`rollout/tldr_reward first5=-159.79 → last5=-141.05`, Δ+18.74).

## What was different vs the 2gpu smoke

| | `slurm_2gpu.sbatch` (job 1615712) | `slurm_8gpu.sbatch` (job 1615870) |
|---|---|---|
| Trainer launcher | `irl train --devices 1` (1 OS process) | `accelerate launch --num_processes 7 -m inspect_rl.cli train …` (7 OS processes) |
| Effective batch / step | 4 | 28 (4 per-device × 7 ranks) |
| Queue wait | ~0s | ~30 min (no whole node free; backfilled when a neighbour's job finished) |
| Trainer cold start (`trainer step starting` → `trainer.train() starting`) | ~60s | ~3 min 13s (7 ranks each import torch + load weights through DDP init) |
| Wall time / step | ~8s | ~40s (rollout dominates: 28-sample rollout vs 4) |
| Off-policy mode (`AutoCalibratingRolloutFunc`) | `warmup-only` | flipped to `prefetch on` after step 5 |
| Final state | clean exit, 10/10 steps | crash at step 6, 5/10 steps |

## Bug: prefetch path drops `inspect_scores`

The 2gpu run finished cleanly because the rollout was faster than the trainer (4 samples; trainer never fell behind), so `AutoCalibratingRolloutFunc` stayed in `warmup-only` mode and never enabled prefetch. The 8gpu run has a 7× larger effective batch → rollout is now slower than the trainer step → the calibrator flips to `prefetch on` at the warmup boundary (step 5). The next batch (step 6) comes from the prefetch path, and TRL's `_calculate_rewards` trips on:

```
File ".../trl/trainer/grpo_trainer.py", line 1186, in _calculate_rewards
    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                                ~~~~~~~^^^^^
KeyError: 'inspect_scores'
```

Diagnosis hypothesis: the warmup path adds an `inspect_scores` field to each rollout sample (consumed by our scorer pipeline downstream); the prefetch path constructs batches via a different code path that hasn't been kept in sync with that field. TRL iterates `for key in keys` where `keys` is derived from `inputs[0].keys()` minus a denylist, so once `inspect_scores` is present in any prior batch's input schema and absent in a later one, this `KeyError` is unavoidable.

**Repro shape**: any config where `7 * rollout_t > train_t` (so calibrator turns prefetch on) + multi-rank DDP. Hadn't shown up before because the existing `hpc1/` configs use multi-node trainer + multi-node vLLM where the per-rank rollout cost dominates and the calibrator behaves differently.

**Workaround options to try next:**

1. Pin `off_policy_steps=0` in the rollout function for 8gpu runs (forces warmup-only, no prefetch). Easy, lose the prefetch throughput win.
2. Audit the prefetch path in `src/inspect_rl/core/rollout.py` (or wherever `AutoCalibratingRolloutFunc` builds prefetched batches) and ensure the same `inspect_scores` field is attached.
3. Make the field optional on the reward-func side so its absence doesn't `KeyError` — but TRL's iteration logic is fixed, so this would need to filter the inputs schema before TRL sees it.

Option 2 is the correct fix; option 1 unblocks runs in the meantime.

## Surprise: `trainer.out` is block-buffered, not silent

I'd initially thought `trainer.out` was nearly empty under `accelerate launch` because it had only the boot banner after ~3 min. **Wrong** — it's the awk `stamp` pipe block-buffering. Every line in trainer.out from this run has the timestamp `[14:50:40]` (the moment the trainer process exited and stdout drained), even though events spanned 14:46–14:50. ~38 KB landed in one flush.

**Practical impact for monitoring 8gpu runs is the same** but the reason is different:

- `tail -F trainer.out` shows nothing live — the pipe is block-buffered, not multiplexed.
- `tail -F train.log` is the live source of truth — rank 0's `inspect_rl` structured INFO log uses `logging.FileHandler` which flushes per record.
- The watchdog in `master.out` checks `trainer.out` mtime, so it will keep warning "idle Ns" even while the run is healthy. Treat watchdog as informational for `slurm_8gpu.sbatch`.

Fix worth trying: pipe trainer stdout through `stdbuf -oL -eL` before `stamp`, or replace the `awk` stamp with `ts` which line-buffers natively.

## Things that worked first try

- `accelerate launch` with `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7` and `--num_processes 7` came up cleanly. No accelerate config file — defaults are fine for single-node DDP.
- Rank gating worked as expected (per [`008_multinode/002_rank_gating.md`](../008_multinode/002_rank_gating.md)) — only rank 0 issued `/generate/` and `/update_named_param/` to vLLM; the other 6 ranks participated in NCCL allreduces only.
- W&B init: only rank 0 logged in. `--wandb` flag and the `WANDB=1` env gate from [`001_debugging_practices.md`](001_debugging_practices.md) propagated through `accelerate launch` without further fuss.

## Queue / cluster notes

`(Resources)` reason at submit — no whole node free across the `general` partition (23 mix / 6 alloc / 1 drain). Slurm's backfill predicted start ~30 min out, then ran at the actual backfill slot when a neighbour's 24h job finished. `overflow` was tested with `sbatch --test-only` and predicted **later** start (~44 min) plus 227 pending jobs ahead — `general` was correctly the right partition. Higher QoS wouldn't have helped: already on `high`, and the blocker was "no eligible whole node" not "lower-priority jobs ahead".

## Fix landed: `off_policy_steps` now defaults to 0 (synchronous)

Disabled prefetch by default across all CLI/example entrypoints + `inspect_rl_train()`. The prefetcher (`FreshestPrefetchRolloutFunc` / `AutoCalibratingRolloutFunc` in `src/inspect_rl/perf/prefetch.py`) is opt-in via `--off-policy-steps N` (N>0 or -1) until the multi-rank instability is fixed. Single-rank (hpc1/1node trainer, hpc2/h200 2gpu) is unaffected throughput-wise; multi-rank loses the prefetch throughput win, which is fine for now since we have nothing in production yet.

## Open follow-ups (next prefetch sitting)

The disable-by-default is a workaround, not a fix. Two layers still needed before re-enabling:

### Layer 2 — defensive assertions in `_RankGatedRolloutFunc`

The current slice in `src/inspect_rl/core/rollout.py:100-110` only fires when `len(value) == num_processes * local_size`; on mismatch it falls through silently, and TRL's `_calculate_rewards` then KeyErrors 1000 lines later with no breadcrumb. Replace the conditional with a hard assert that names the offending key and the expected/actual lengths. Turns this entire failure class into a one-line traceback.

### Layer 3 — pass `inspect_scores` out-of-band

Root cause is TRL's index-based merge of `extra_fields` into `inputs` ([grpo_trainer.py:2115-2122](../../.venv/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py)). It assumes the rollout result's per-sample fields line up 1:1 with TRL's dataloader batch. Under prefetch they don't — the prefetcher uses its own `_next_batch()`, so completions, prompt_ids, and inspect_scores all correspond to a *different* set of prompts than what TRL pulled from its dataloader.

Even when lengths happen to match, the score-to-completion correspondence is wrong; we just don't see it because the per-step reward log doesn't surface the mismatch. So "fix the lengths" is not enough — the whole approach of merging `inspect_scores` via TRL's `inputs` is brittle.

Two candidate redesigns:

- **(a)** Stash `inspect_scores` on `trainer._inspect_scores_latest` from inside `_RankGatedRolloutFunc.__call__`, drop the key from the rollout result dict (so TRL never tries to merge it), and have `_make_scores_reward_func` read from `kwargs["trainer_state"]._inspect_scores_latest`. ~15 lines, removes the entire failure class. Preferred.
- **(b)** Audit the prefetcher to make sure it returns rollouts whose prompts correspond to TRL's `inputs`. Probably means abandoning `_next_batch()` and prefetching against TRL's dataloader order, which is a bigger refactor and re-introduces lock-step with the trainer. Not worth it.

Acceptance criteria when prefetch is re-enabled by default:

1. 8gpu DDP run (this config) completes ≥20 steps under `--off-policy-steps -1` without `KeyError`.
2. `[summary]` line shows `off-policy: prefetch on`.
3. Final val accuracy is monotonically related to baseline (we're not silently training on mismatched completions).
4. Test added that fakes a prefetcher returning a length-mismatched payload and asserts the failure is the named-key assertion from Layer 2, not a downstream KeyError.

## Old follow-ups

- **Fix the prefetch path** (`KeyError: 'inspect_scores'`). Priority: blocks any 8gpu DDP run that exceeds warmup. — Mitigated by disable-default; full fix is Layers 2 + 3 above.
- Replace the `awk` stamp pipe with `ts` or wrap stdout in `stdbuf -oL` so trainer.out streams live under accelerate.
- Try `Qwen2.5-3B` on this topology — 0.5B with 7-rank DDP and 28-sample effective batch is bandwidth-bound on the rollout side, not throughput-bound on the trainer side, so the actual win-per-step is small. 3B+ is where DDP starts to pay back.
- Consider FSDP/ZeRO-3 once we step up to ≥7B; the current `slurm_8gpu.sbatch` defaults to plain DDP, which means a full optimizer-state copy per rank.
