# `examples/configs/` — multi-node launch templates

Worked-example Slurm + accelerate configurations for running the trainer at
different scales with vLLM serving on its own GPU(s). Each scale lives in its
own subdirectory with the sbatch entry point alongside the accelerate config
it loads — so you never have to chase config dependencies across the tree.

## Layout

```
examples/configs/
├── README.md                 (this file)
├── 1node/                    1 trainer node, separate persistent vLLM job
│   ├── slurm_trainer.sbatch    accelerate-launched trainer
│   ├── slurm_vllm.sbatch       persistent vLLM server (run separately)
│   └── accelerate.yaml         4-rank DDP on a single machine
├── 2node/                    canonical small multi-node config
│   ├── slurm.sbatch            combined: 1 vllm node (TP=1) + 1 trainer node
│   └── accelerate.yaml         4-rank DDP on the trainer node
├── 4node/                    canonical production multi-node config
│   ├── slurm.sbatch            combined: 1 vllm node (TP=4) + 3 trainer nodes
│   ├── accelerate.yaml         12-rank FSDP SHARD_GRAD_OP (default)
│   ├── accelerate_zero2.yaml   12-rank DeepSpeed ZeRO-3 alternative (override via env)
│   └── deepspeed.json          DeepSpeed settings for the ZeRO-3 alternative
└── 8node/                    72B-scale config
    ├── slurm.sbatch            combined: 1 vllm node (TP=4) + 7 trainer nodes
    ├── accelerate_zero3.yaml   28-rank DeepSpeed ZeRO-3 (default)
    └── deepspeed.json          DeepSpeed ZeRO-3 settings
```

## Which config?

| Scale | Topology | vLLM | Trainer | Notes |
|---|---|---|---|---|
| `1node/` | 2 separate Slurm jobs | 1 node, any TP | 1 node × 4 GPUs DDP | Persistent vLLM across multiple trainer iterations. Only safe with a 1-node trainer — multi-node trainer + separate jobs trips a CXI VNI bug. |
| `2node/` | 1 Slurm job, 2 nodes | 1 GPU, TP=1 | 1 node × 4 GPUs DDP | Smallest multi-node config. Smoke-tested end-to-end. |
| `4node/` | 1 Slurm job, 4 nodes | 4 GPUs, TP=4 | 3 nodes × 4 GPUs = 12-rank FSDP | Production shape for 7-8B-class models. Default uses PyTorch FSDP1 SHARD_GRAD_OP (no DeepSpeed dep); set `ACCELERATE_CONFIG=examples/configs/4node/accelerate_zero2.yaml` to swap in DeepSpeed ZeRO-3. |
| `8node/` | 1 Slurm job, 8 nodes | 4 GPUs, TP=4 | 7 nodes × 4 GPUs = 28-rank ZeRO-3 | Sized for Qwen2.5-72B. Jump here when ZeRO-3 per-rank memory under 12 ranks exceeds ~50 GiB; 32B still fits under `4node/`. |

`2node/`, `4node/`, and `8node/` run vLLM and the trainer as **concurrent srun
steps within one Slurm job**. On Slingshot/CXI Slurm HPC clusters (e.g. Cray
EX) this is required so both share the job's VNI namespace — full diagnosis
in [`journal/008_multinode/006_vllm_worker_internal_error.md`](../../journal/008_multinode/006_vllm_worker_internal_error.md).
They also force `NCCL_NET=Socket` to bypass `aws-ofi-nccl`, which trips
`VNI_NOT_FOUND` with ≥2 trainer peers.

## Quick start

```bash
mkdir -p slurm_logs  # one-time, sbatch --output writes here

# 2-node, 0.5B smoke test:
env -u MODEL sbatch \
    --export=ALL,NCCL_NET=Socket,MODEL=Qwen/Qwen2.5-0.5B-Instruct,EXAMPLE=tldr,MAX_STEPS=2 \
    examples/configs/2node/slurm.sbatch

# 4-node, 7B production-shape:
env -u MODEL sbatch \
    --export=ALL,NCCL_NET=Socket,MODEL=Qwen/Qwen2.5-7B-Instruct,EXAMPLE=math-agent,MAX_STEPS=200 \
    examples/configs/4node/slurm.sbatch

# 8-node, 72B:
env -u MODEL sbatch \
    --export=ALL,NCCL_NET=Socket,MODEL=Qwen/Qwen2.5-72B-Instruct,EXAMPLE=math-agent,MAX_STEPS=3 \
    examples/configs/8node/slurm.sbatch

# 1-node trainer with a persistent vLLM (run vLLM first, then trainer):
sbatch examples/configs/1node/slurm_vllm.sbatch
VLLM_URL=$(cat slurm_logs/latest_vllm_url.txt)
env -u MODEL sbatch --export=ALL,VLLM_URL=$VLLM_URL,MAX_STEPS=2 \
    examples/configs/1node/slurm_trainer.sbatch
```

Each run writes per-rank logs to `slurm_logs/<datetime>_<jobid>/{master,vllm,trainer-N}.out`.
Symlinks `slurm_logs/latest_combined_*` (or `latest_{vllm,trainer}_*` for `1node/`)
always point at the most recent run for easy `tail -F`.

## Debug mode

`2node/`, `4node/`, and `8node/` accept `IRL_DEBUG=1` to switch on verbose tracing:

```bash
env -u MODEL sbatch \
    --export=ALL,IRL_DEBUG=1,NCCL_NET=Socket,MODEL=...,EXAMPLE=tldr,MAX_STEPS=2 \
    examples/configs/2node/slurm.sbatch
```

`IRL_DEBUG=1` enables on both vLLM and trainer:

- `NCCL_DEBUG=TRACE`, `NCCL_DEBUG_SUBSYS=ALL` — per-channel NCCL state
- `FI_LOG_LEVEL=debug`, `FI_LOG_PROV=cxi` — libfabric/CXI provider trace
- `TORCH_DISTRIBUTED_DEBUG=DETAIL`, `TORCH_NCCL_TRACE_BUFFER_SIZE=65536`
- A py-spy sidecar that dumps live stacks every 30s to `pyspy-<host>.txt`

Trace output is ~1k libfabric lines per minute per process; leave off for production.

## Caveats

- Targets an aarch64 Slurm HPC cluster (`workq` partition, `--gpus-per-node=4`). On other Slurm clusters, adjust the `#SBATCH` block.
- All sbatches include commented-out `module load` lines for cluster NCCL and aws-ofi-nccl modules (required for cross-node NCCL on some HPC clusters — uncomment and replace with your cluster's module names). They also set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` (compute nodes typically have no HF Hub access — pre-populate `$HF_HOME` from a login node).
- `env -u MODEL` strips a stray `MODEL` from the parent shell so `--export=ALL,MODEL=…` actually takes effect — without it the harness's `MODEL` clobbers your override.
- Only trainer rank 0 issues requests to vLLM. `--num_processes N` shows up as one vLLM client, not N. See [`journal/008_multinode/002_rank_gating.md`](../../journal/008_multinode/002_rank_gating.md).
- `NCCL_NET=Socket` is a workaround — the intra-trainer allreduces also go over TCP instead of Slingshot RDMA. Acceptable for 0.5–7B; a known bottleneck at 13B+ and especially under ZeRO-3 / FSDP at the `8node/` scale (param gather/reduce-scatter on every step). Re-enabling RDMA needs an upstream fix in `aws-ofi-nccl`.
