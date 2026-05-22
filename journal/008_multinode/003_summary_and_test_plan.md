# 003 — Multi-GPU work: what shipped + how to test it

Summary of [PR #20](https://github.com/AI-Safety-Institute/inspect-rl/pull/20) (branch `feat/multinode`), and the smoke-test recipe for verifying it on a real Slurm cluster — `001_starting_point.md` listed the goals; this is the wrap-up.

## What shipped

- **Rank-gated rollout** (`rollout._RankGatedRolloutFunc`). Wraps the rollout chain so only rank 0 runs `inspect_eval` against the shared vLLM server. Mirrors stock TRL's `vllm_generation.generate` pattern: `gather_object(prompts)` → main-only inner call → `broadcast_object_list` → each rank slices `[process_index * local_size : ...]`. No-op when `world_size == 1`.
- **Prefetcher rank-aware.** `FreshestPrefetchRolloutFunc` / `AutoCalibratingRolloutFunc` take `world_size` and produce `world_size * batch_size` prompts per slot. Producer thread only spawns on rank 0 (because non-main ranks never reach the inner call inside the wrapper). Other ranks block on the per-step broadcast.
- **Setup gates.** vLLM health probe, run-dir creation, resume IO, file logging, and end-of-run summary are all rank-0-only. Run-dir path is broadcast so every rank writes into the same timestamped directory.
- **CLI redirect.** `irl train --num-processes N` (N>1) now points the user at `accelerate launch -m inspect_rl.cli train …` rather than silently re-exec'ing.
- **README + examples.** New "Multi-GPU and multi-node" section pinned to `--per-device-train-batch-size 4 --num-generations 4` (default `bs=8` OOMs at DDP step ≈ 3 once Adam materializes its fp32 master/m/v). `examples/configs/` has `accelerate_4gpu_singlenode.yaml`, `accelerate_8gpu_2node.yaml`, and a Slurm `slurm_2node_8gpu.sbatch` template.

## What does NOT need to change for larger setups

- **Multi-GPU vLLM.** `trl vllm-serve --tensor_parallel_size N` shards one model across N GPUs behind a single URL. `--data_parallel_size M` adds M replicas for throughput. Our trainer talks to one URL either way — the rank-gated wrapper means there's exactly one client per step regardless of `--num_processes`.
- **One vLLM client.** A reasonable worry on the second look at the design is "won't N trainer ranks each spam vLLM?" — they won't. Only rank 0 issues requests. Pulling that into a separate rollout *service* (out-of-process) is a real refactor for marginal gain right now — see Q3 in the PR thread for the trade-offs.

## How to test it

### Single node (done; this PR's evidence)

GH200 dev node, vLLM on GPU 0 (Qwen2.5-3B), trainer on GPUs 1+/2/3:

```bash
# Reset any stale NCCL communicator from a previous run.
curl -sf -X POST http://localhost:8000/close_communicator/ -d '{}'

# 2-rank sync (verifies gather/broadcast end-to-end)
CUDA_VISIBLE_DEVICES=1,2 uv run accelerate launch --num_processes 2 \
    -m inspect_rl.cli train math-agent \
    --max-steps 2 --off-policy-steps 0 \
    --eval-steps 100 --save-steps 100 --resample-rounds 0

# 3-rank with prefetch
CUDA_VISIBLE_DEVICES=1,2,3 uv run accelerate launch --num_processes 3 \
    --main_process_port 29501 \
    -m inspect_rl.cli train math-agent \
    --max-steps 4 --off-policy-steps 1 \
    --per-device-train-batch-size 4 --num-generations 4 \
    --eval-steps 100 --save-steps 100 --resample-rounds 0
```

Pass criteria seen in `train.log` (rank 0 only — file logging is now gated):

- `[step N/M]` lines appear with non-zero scores and increasing.
- `[summary] N steps in …s · off-policy: <sync|prefetch on>` at the end, no exception.
- `train.log` is *not* duplicated per rank (regression check on the logging-gate change).
- Sample count per `[rollout] starting:` equals `world_size * per_device_train_batch_size`.

Don't stack runs back-to-back: `on_train_end` fires a final val eval (~90s) that keeps GPU memory pinned long after the last `[step N]` line.

### Cluster (this is what's outstanding)

As of `sinfo` this turn: `workq` has 6 fully-idle nodes (24 GH200s). Plenty for the 2-node smoke. Reservation, queue load, and project allocation will gate when you can actually grab two.

The template encodes Layout A — vLLM with `--tensor-parallel-size 4` on node 0, 4-rank trainer on node 1. Submit:

```bash
sbatch examples/configs/slurm_2node_8gpu.sbatch
# overrides via --export, e.g.:
sbatch --export=MODEL=Qwen/Qwen2.5-7B-Instruct,EXAMPLE=math-agent \
    examples/configs/slurm_2node_8gpu.sbatch
```

The sbatch waits for vLLM `/health/` on the trainer node before launching accelerate, and splits stdout into `vllm-<jobid>.out` and `trainer-<jobid>.out`.

Pass criteria on Slurm (in addition to the single-node ones):

- `vllm-<jobid>.out` shows the model loaded and `INFO: Application startup complete.`
- `trainer-<jobid>.out` shows `=== trainer.train() starting` with `max_steps=…` from one rank, then `[step N/M]` lines.
- The `Logging to …` path resolves to a shared filesystem location (e.g. `$SCRATCH/outputs/…`) so it's reachable from your login node.
- Cancelling the sbatch (`scancel <jobid>`) tears both pieces down.

### Things that have not been verified yet on Slurm

- Multi-node trainer (Layout B). Wiring is in `accelerate_8gpu_2node.yaml` but no sbatch template — needs `MAIN_PROCESS_IP` derived from `$SLURM_JOB_NODELIST` and `--machine_rank` set per-node. Worth doing once Layout A passes.
- Recovery semantics if vLLM crashes between steps. Single-node test shows TRL hangs ~5 min then raises `DistStoreError`; behaviour on a remote vLLM hasn't been exercised.
- The freshest-prefetch broadcast overhead on a real cluster fabric. Per-step broadcast is ~MB-scale (token IDs + logprobs for `world_size * batch_size` samples); negligible over NVLink, possibly visible over slow inter-node.

## Reference

- PR: <https://github.com/AI-Safety-Institute/inspect-rl/pull/20>
- Implementation notes: [`002_rank_gating.md`](002_rank_gating.md)
- Launch templates: [`examples/configs/`](../../examples/configs/)
