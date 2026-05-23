# 04 — Known issues

*Last updated: 2026-05-13*

## Requires ≥ 2 GPUs — trainer and vLLM cannot share a device

`vllm_mode="server"` opens an NCCL communicator between the trainer rank and the vLLM-server rank for weight sync, and NCCL refuses to bind two distinct ranks to the same physical device. Attempting to colocate (`CUDA_VISIBLE_DEVICES=0` on both processes) fails with `RuntimeError: Attempting to use the same CUDA device ... for multiple distinct roles/ranks within the same communicator`. Even tiny models (gemma-3-270m) need the trainer on a different GPU from `irl serve`. `vllm_mode="colocate"` sidesteps this in principle but doesn't currently play well with Inspect's async solver chain (see [`03_internals.md`](03_internals.md) "Where this could be simpler").

## `uv sync` fails on torch wheel with `UV_EXCLUDE_NEWER` set

If `UV_EXCLUDE_NEWER` is set to a cutoff date, `uv sync` may fail to lock/install torch with e.g. `torch-2.7.0+cu128-cp313-cp313t-manylinux_2_28_x86_64.whl is missing an upload date, but user provided: 2026-04`. PyTorch's extra index doesn't publish upload dates on its wheels, so `uv` can't compare them against the cutoff and refuses the wheel. Workaround: `unset UV_EXCLUDE_NEWER` before `uv sync`.

## `uv sync` partial torch install (hardlink fallback)

When `uv`'s cache and the venv are on different filesystems, hardlink creation falls back to copy and a few torch sibling files (`torchgen/`, internal `torch._higher_order_ops/`) can be dropped. The symptom is `ModuleNotFoundError: No module named 'torchgen'` on `import torch`. Fix:

```bash
uv sync --link-mode=copy --reinstall-package torch
```

Do **not** `pip install torchgen` from PyPI — that's an unrelated name-squat package containing GAN/VAE code.

## `inspect_eval` intermittently hangs mid-training

`inspect_eval()` occasionally wedges during a rollout or validation eval — GPU utilization drops to 0%, vLLM responds fine to health checks and direct `/generate/` calls, but the `inspect_ai` async sample runner is stuck. Observed at steps 20 and 49 across different runs. Root cause is unknown (inside `inspect_ai`'s async scheduler). Mitigation: keep `save_steps` small to limit data loss, then resume from the latest checkpoint.

## Stale NCCL communicator after trainer restart

If a training run is killed without clean shutdown (e.g. Jupyter kernel restart), vLLM's NCCL communicator survives and the next `init_communicator` call hangs for ~5 minutes before failing with `DistStoreError`. Always close the communicator between runs:

```bash
curl -sf -X POST http://localhost:8000/close_communicator/ -d '{}'
```

The `just train` recipe and `irl train` entry point do this automatically.

## Gemma-3 compatibility

Gemma-3 models require a `token_type_ids` kwarg that TRL's `GRPOTrainer` never passes, causing a `ValueError` during training. Text-only models (Qwen, Llama) are unaffected. Additionally, vLLM 0.15.0 hits an import error with Gemma-3's multimodal processor (`ReasoningEffort` from `mistral_common`). Workaround: use Qwen or Llama models, or upgrade to vLLM >= 0.17.

## CUDA out of memory

If you hit OOM errors, run `uvx nvitop` to see per-GPU memory usage and check the process list at the bottom of the UI for zombie processes from previous runs. Kill any stale processes to reclaim memory.

## Batch size vs. unique prompts per step

With default settings (`per_device_train_batch_size=8`, `num_generations=8`), TRL's `RepeatSampler` yields 8 copies of the same prompt per step — meaning only `max_steps` unique prompts are seen across training. Increase `per_device_train_batch_size` relative to `num_generations` for better prompt coverage, keeping GPU memory in mind.

## Multi-node specific

See [`journal/008_multinode/004_cluster_smoke.md`](../journal/008_multinode/004_cluster_smoke.md) for a validated Slurm HPC setup and the known gotchas there (harness `MODEL` env clobber, compute-node HF offline mode, `brics/nccl` + `brics/aws-ofi-nccl` modules, deepspeed `torch_adam` to avoid FusedAdam JIT-compile).

## Multi-node trainer + separate vLLM Slurm job → `VNI_NOT_FOUND`

On Slingshot/CXI Slurm HPC clusters (e.g. Cray EX), if vLLM is submitted as one Slurm job and a **multi-node** trainer is submitted as a separate job, the trainer's NCCL weight-sync `init_communicator` AllReduce fails with:

```
NCCL WARN NET/OFI Request ... completed with error. RC: 107.
Error: Inappropriate ioctl for device.
```

Underlying cause (visible with `FI_LOG_LEVEL=info FI_LOG_PROV=cxi`):

```
cxi:av:cxip_av_insert_addr(): nid=0x... inserted multiple times
cxi:ep_data:cxip_report_send_completion(): error (err: 107, VNI_NOT_FOUND)
```

Each Slurm job is allocated its own CXI **Virtual Network Identifier** (VNI). When the multi-node trainer's intra-DDP NCCL group has already opened CXI endpoints in its VNI, it can't add vLLM's NIC (which lives in a different job's VNI namespace) — the SEND is rejected at the kernel driver. 1-node trainer works because its intra-DDP NCCL is intra-node (NVLink, no CXI EPs opened).

**Fix:** combine TWO knobs:
1. Run vLLM and the trainer as concurrent `srun` steps inside a **single** sbatch job (same VNI namespace).
2. Set `NCCL_NET=Socket` to bypass aws-ofi-nccl entirely — its CXI provider advertises `pid=0` for every peer address, so the libfabric AV table collapses entries with ≥2 trainer nodes and SEND fails with `VNI_NOT_FOUND`. Forcing TCP socket transport sidesteps the plugin.

Worked template: [`examples/configs/hpc1/2node/slurm.sbatch`](../examples/configs/hpc1/2node/slurm.sbatch). Submit with:

```bash
env -u MODEL sbatch \
    --export=ALL,NCCL_NET=Socket,MODEL=Qwen/Qwen2.5-0.5B-Instruct,EXAMPLE=tldr,MAX_STEPS=2 \
    examples/configs/hpc1/2node/slurm.sbatch
```

This was validated end-to-end (2-node trainer + 1-node vLLM, 2 training steps in 1m46s) — job 202638, 2026-05-13.

Cost: TCP socket transport is slower than Slingshot RDMA. For 0.5B it's negligible; for 7B+ the per-step weight broadcast becomes the bottleneck. The intra-trainer NCCL group also uses Socket (NCCL_NET is per-process global), so DDP allreduces are likewise on TCP. Re-enabling RDMA needs an upstream fix in aws-ofi-nccl or its CXI provider config — track via your HPC vendor's support channel.

Full diagnosis: [`journal/008_multinode/006_vllm_worker_internal_error.md`](../journal/008_multinode/006_vllm_worker_internal_error.md).

### Debug mode for NCCL / libfabric / CXI issues

The combined sbatch takes `IRL_DEBUG=1` to switch on verbose logging that surfaces issues like the one above:

```bash
env -u MODEL sbatch --export=ALL,IRL_DEBUG=1,MODEL=...,EXAMPLE=tldr,MAX_STEPS=2 \
    examples/configs/hpc1/2node/slurm.sbatch
```

`IRL_DEBUG=1` enables:
- `NCCL_DEBUG=INFO` with `NCCL_DEBUG_SUBSYS=INIT,COLL,NET,ENV` (NCCL channel/transport selection, bootstrap, errors)
- `FI_LOG_LEVEL=info` with `FI_LOG_PROV=cxi` (libfabric/CXI provider trace — needed for things like the VNI bug above)

Output lands in `slurm_logs/<stamp>_<jobid>/vllm.out` and `slurm_logs/<stamp>_<jobid>/trainer-*.out`. The trace is large (~1k libfabric lines per minute per process), so leave debug mode off for production runs.
