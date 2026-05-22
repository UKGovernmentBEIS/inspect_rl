# 002 — Single-node multi-GPU via accelerate launch

What landed on `feat/multinode` to make `accelerate launch --num_processes N -m inspect_rl.cli train …` actually work, plus the smoke-test evidence.

## What the wiring does

TRL's `_generate` calls `self.rollout_func(prompts, self)` on **every rank** — there's no rank guard in the extension point itself. Stock TRL's vLLM path (`trl.generation.vllm_generation.generate`) handles the multi-rank case internally by `gather_object(prompts) → main-only generate → broadcast_object_list → slice[process_index * len(prompts):...]`. We mirror that pattern as a wrapper around our rollout chain (`rollout.make_rank_gated_rollout_func` / `_RankGatedRolloutFunc`).

Layering matters. The wrapper is the **outermost** thing — so when there's a prefetcher inside, only rank 0's producer thread spawns (because non-main ranks never reach the inner call inside the wrapper). The prefetcher's `_next_batch` now takes `world_size` and generates `world_size * batch_size` prompts per slot so each rank's slice is well-formed after broadcast.

Other pieces that needed rank-gating:
- `_check_vllm_server` (main only — vLLM is shared, one probe is enough).
- `create_run_dir` / `_prepare_resume` (rank 0 picks the timestamped path; broadcast to others so all ranks write into the same dir). Without this, ranks would race on timestamps and write into distinct dirs.
- Val eval callback (`is_world_process_zero` guard — already there from before).
- `_apply_train_env` now redirects `--num-processes>1` to `accelerate launch -m inspect_rl.cli …` instead of erroring as "not wired yet".

The accelerator object lives on `trainer.accelerator` at rollout-time; we use that for `is_main_process` / `process_index` / `num_processes`. For setup work that happens *before* `GRPOTrainer.__init__` (run-dir broadcast), we use `accelerate.PartialState()` — same env-var-driven backend, so the values match what TRL's accelerator will see.

## Smoke tests on this node (1 vllm GPU + N trainer GPUs)

```bash
# vLLM on GPU 0 (already running from earlier session, Qwen2.5-3B)
CUDA_VISIBLE_DEVICES=1,2 uv run accelerate launch --num_processes 2 \
    -m inspect_rl.cli train math-agent \
    --max-steps 2 --off-policy-steps 0 --eval-steps 100 --save-steps 100 --resample-rounds 0
```

→ `[summary] 2 steps in 217.1s · off-policy: sync`. Both ranks loaded the model on their own GPU (logical cuda:0 → physical GPU 1; logical cuda:1 → physical GPU 2), both wrote to the same timestamped dir, rollout 1 produced 16 samples (2 × batch_size 8), step 2 saved a checkpoint cleanly. Sync path with `--off-policy-steps 0` confirms the gather → main-only eval → broadcast → slice loop end-to-end.

3-rank prefetch test (`--num_processes 3 --off-policy-steps 1`, GPUs 1+2+3):

- First attempt at `bs=8` OOM'd at step 3 (not step 1, not step 2). The pattern "step 1 fit, step 3 didn't" is memory *growth*, not a flat budget overrun — Adam's fp32 master + m + v state for 3B params (~24 GiB) materializes lazily on the first optimizer step, not at construction. So baseline-then-step-1 fit, step-2 was halfway through, step-3 ran out. Architectural mechanism is fine.
- Re-ran with `--per-device-train-batch-size 4 --num-generations 4`: 4 steps in 242.4s, prefetch on, checkpoint saved, reward signal climbing 0.17 → 0.42 → 0.42 → 0.58 on `correctness`. End-to-end win.

## Concrete gotchas I hit

- **Two concurrent runs collide.** `on_train_end` runs an extra val eval (~90s on math-agent), so the trainer process holds memory long after the final `[step N]` log line. If you stack runs without waiting, the next one OOMs on whichever GPUs the previous trainer was using. Pattern: wait for the `[summary]` line, not the step count, before starting the next.
- **vLLM weight sync** under DDP works without us doing anything — TRL's `update_named_param` is already main-only for `vllm_mode="server"`, so all trainer ranks call `sync_weights` but only rank 0 pushes parameters via NCCL to the vLLM server's communicator. We don't need to touch it.
- **train.log gets each line N times.** Every rank's logger writes to the same file. Cosmetic. Could fix by gating the FileHandler to rank 0 (`if not is_main: remove handler`), but lower-priority than the smoke tests.

## What's still open

- The full 4-GPU smoke (1 vllm + 3 trainer with prefetcher) needs to land cleanly before I touch the README tutorial. As of writing the re-run is in flight.
- Multi-node (across two Slurm HPC nodes via Slurm) is the user-deferred next step — the wiring for it is identical (accelerate handles `--num_machines>1`); the user will smoke-test on Slurm later.
- `--num-processes` on plain `irl train` still errors with a redirect message. The supported path is `accelerate launch -m inspect_rl.cli train …`. We chose not to re-exec inside the CLI — clearer to let the user own the launcher. Worth a README note once the smoke passes.
