# `examples/configs/` — Slurm launch templates

Worked Slurm + accelerate configurations for running the trainer at different scales with vLLM on its own GPU(s). Two cluster families, each with its own README:

```
examples/configs/
├── README.md          (this file — index + hpc1 cluster notes)
├── hpc1/              Slingshot/CXI Slurm HPC (aarch64 NVHPC + GH200/H100)
│   ├── 1node/         1 trainer node + separate persistent vLLM job
│   ├── 2node/         combined sbatch: 1 vllm node (TP=1) + 1 trainer node
│   ├── 4node/         combined sbatch: 1 vllm node (TP=4) + 3 trainer nodes
│   └── 8node/         combined sbatch: 1 vllm node (TP=4) + 7 trainer nodes (72B-scale)
└── hpc2/              Shared multi-tenant H200 Slurm cluster
    └── h200/          single-node smoke (2 GPUs) or full-node DDP (8 GPUs)
                       — see hpc2/h200/README.md for sequence + log table
```

## Which config?

| Scale | Topology | vLLM | Trainer | Notes |
|---|---|---|---|---|
| [`hpc2/h200/`](hpc2/h200/) | 2 srun steps in 1 exclusive node | 1 GPU, TP=1 | 1 GPU (`slurm_2gpu.sbatch`) or 7-rank DDP (`slurm_8gpu.sbatch`) | H200 single-node. Encodes `--partition=general --qos=high` and assumes `ConstrainDevices=no`. See [`hpc2/h200/README.md`](hpc2/h200/) for process model + sequence diagram. |
| [`hpc1/1node/`](hpc1/1node/) | 2 separate Slurm jobs | 1 node, any TP | 1 node × 4 GPUs DDP | Persistent vLLM across multiple trainer iterations. Only safe with a 1-node trainer — multi-node trainer + separate jobs trips a CXI VNI bug. |
| [`hpc1/2node/`](hpc1/2node/) | 1 Slurm job, 2 nodes | 1 GPU, TP=1 | 1 node × 4 GPUs DDP | Smallest multi-node config. Smoke-tested end-to-end. |
| [`hpc1/4node/`](hpc1/4node/) | 1 Slurm job, 4 nodes | 4 GPUs, TP=4 | 3 nodes × 4 GPUs = 12-rank FSDP | Production shape for 7-8B-class models. Default uses PyTorch FSDP1 SHARD_GRAD_OP; set `ACCELERATE_CONFIG=examples/configs/hpc1/4node/accelerate_zero2.yaml` to swap in DeepSpeed ZeRO-3. |
| [`hpc1/8node/`](hpc1/8node/) | 1 Slurm job, 8 nodes | 4 GPUs, TP=4 | 7 nodes × 4 GPUs = 28-rank ZeRO-3 | Sized for Qwen2.5-72B. Jump here when ZeRO-3 per-rank memory under 12 ranks exceeds ~50 GiB; 32B still fits under `4node/`. |

---

## hpc1 cluster notes

The rest of this file is hpc1-specific (Slingshot/CXI HPC). hpc2 specifics live in [`hpc2/h200/README.md`](hpc2/h200/).

`hpc1/2node/`, `hpc1/4node/`, and `hpc1/8node/` run vLLM and the trainer as **concurrent srun steps within one Slurm job**. On Slingshot/CXI clusters (e.g. Cray EX) this is required so both share the job's VNI namespace — full diagnosis in [`journal/008_multinode/006_vllm_worker_internal_error.md`](../../journal/008_multinode/006_vllm_worker_internal_error.md). They also force `NCCL_NET=Socket` to bypass `aws-ofi-nccl`, which trips `VNI_NOT_FOUND` with ≥2 trainer peers.

### Quick start (hpc1)

```bash
mkdir -p slurm_logs  # one-time, sbatch --output writes here

# 2-node, 0.5B smoke test:
env -u MODEL sbatch \
    --export=ALL,NCCL_NET=Socket,MODEL=Qwen/Qwen2.5-0.5B-Instruct,EXAMPLE=tldr,MAX_STEPS=2 \
    examples/configs/hpc1/2node/slurm.sbatch

# 4-node, 7B production-shape:
env -u MODEL sbatch \
    --export=ALL,NCCL_NET=Socket,MODEL=Qwen/Qwen2.5-7B-Instruct,EXAMPLE=math-agent,MAX_STEPS=200 \
    examples/configs/hpc1/4node/slurm.sbatch

# 8-node, 72B:
env -u MODEL sbatch \
    --export=ALL,NCCL_NET=Socket,MODEL=Qwen/Qwen2.5-72B-Instruct,EXAMPLE=math-agent,MAX_STEPS=3 \
    examples/configs/hpc1/8node/slurm.sbatch

# 1-node trainer with a persistent vLLM (run vLLM first, then trainer):
sbatch examples/configs/hpc1/1node/slurm_vllm.sbatch
VLLM_URL=$(cat slurm_logs/latest_vllm_url.txt)
env -u MODEL sbatch --export=ALL,VLLM_URL=$VLLM_URL,MAX_STEPS=2 \
    examples/configs/hpc1/1node/slurm_trainer.sbatch
```

Each run writes per-rank logs to `slurm_logs/<datetime>_<jobid>/{master,vllm,trainer-N}.out`. Symlinks `slurm_logs/latest_combined_*` (or `latest_{vllm,trainer}_*` for `1node/`) always point at the most recent run for easy `tail -F`.

### Debug mode

`hpc1/2node/`, `hpc1/4node/`, and `hpc1/8node/` accept `IRL_DEBUG=1` to switch on verbose tracing:

```bash
env -u MODEL sbatch \
    --export=ALL,IRL_DEBUG=1,NCCL_NET=Socket,MODEL=...,EXAMPLE=tldr,MAX_STEPS=2 \
    examples/configs/hpc1/2node/slurm.sbatch
```

`IRL_DEBUG=1` enables on both vLLM and trainer:

- `NCCL_DEBUG=TRACE`, `NCCL_DEBUG_SUBSYS=ALL` — per-channel NCCL state
- `FI_LOG_LEVEL=debug`, `FI_LOG_PROV=cxi` — libfabric/CXI provider trace
- `TORCH_DISTRIBUTED_DEBUG=DETAIL`, `TORCH_NCCL_TRACE_BUFFER_SIZE=65536`
- A py-spy sidecar that dumps live stacks every 30s to `pyspy-<host>.txt`

Trace output is ~1k libfabric lines per minute per process; leave off for production.

### Caveats (hpc1)

- Targets an aarch64 Slurm HPC cluster (`workq` partition, `--gpus-per-node=4`). On other Slurm clusters, adjust the `#SBATCH` block.
- All sbatches include commented-out `module load` lines for cluster NCCL and aws-ofi-nccl modules (required for cross-node NCCL on some HPC clusters — uncomment and replace with your cluster's module names). They also set `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` (compute nodes typically have no HF Hub access — pre-populate `$HF_HOME` from a login node).
- `env -u MODEL` strips a stray `MODEL` from the parent shell so `--export=ALL,MODEL=…` actually takes effect — without it the harness's `MODEL` clobbers your override.
- Only trainer rank 0 issues requests to vLLM. `--num_processes N` shows up as one vLLM client, not N. See [`journal/008_multinode/002_rank_gating.md`](../../journal/008_multinode/002_rank_gating.md).
- `NCCL_NET=Socket` is a workaround — the intra-trainer allreduces also go over TCP instead of Slingshot RDMA. Acceptable for 0.5–7B; a known bottleneck at 13B+ and especially under ZeRO-3 / FSDP at the `hpc1/8node/` scale (param gather/reduce-scatter on every step). Re-enabling RDMA needs an upstream fix in `aws-ofi-nccl`.
