# 004 — Cluster smoke: 2-node and 4-node 8B

Cluster validation of the design from [003](003_summary_and_test_plan.md). The Layout A (2-node) plumbing passes end-to-end; Layout B (4-node, 8B via ZeRO-2) was added in the same session.

## 2-node, 3B model (Layout A) — PASS

Job 202547 on `workq` (nodes `<redacted-nid>` vLLM + `<redacted-nid>` trainer).

```
[summary] 2 steps in 311.9s · rollout avg 64.5s · train avg 1.1s · off-policy: sync
[eval 2] eval/correctness/accuracy=0.1875 · uses_calculator=0.7812 · valid_submit=0.4375 · 75.9s
```

Pass criteria from 003 all hit:

- vLLM `Application startup complete.` on node 0.
- Trainer `[step 1/2]` / `[step 2/2]` lines emitted from rank 0; cross-node `POST /init_communicator/` + per-tensor `POST /update_named_param/` calls all returned 200.
- `Logging to <redacted-scratch>/outputs/math_agent/2026-05-13-15-00-10/` — shared Lustre, visible from login + both compute nodes.
- Final val eval ran; checkpoint saved; clean exit (`COMPLETED` 0:0, 6m58s).

## 4-node, Qwen2.5-7B (Layout B with ZeRO-2) — wired, smoke FAILED

Added [`slurm_4node_16gpu.sbatch`](../../examples/configs/slurm_4node_16gpu.sbatch) + [`accelerate_3node_12gpu_zero2.yaml`](../../examples/configs/accelerate_3node_12gpu_zero2.yaml) + [`deepspeed_zero2.json`](../../examples/configs/deepspeed_zero2.json). The plumbing reaches cross-node NCCL `init_communicator` to vLLM (job 202560 confirmed) but **hangs inside `GRPOTrainer.__init__` during DeepSpeed engine init** — `ps` on a trainer node shows all 4 ranks in `Ssl` (sleeping on collective) with no compile activity. Cancelled at ~9 min. Next steps for whoever picks it up:

- `NCCL_DEBUG=INFO` + `NCCL_DEBUG_SUBSYS=ALL` in the trainer srun body to surface which collective is stuck.
- Alternative: swap DeepSpeed for FSDP (built into torch, no external CUDA ops). Would replace `deepspeed_config` with `fsdp_config` in the accelerate YAML.
- Verify the OFI fabric module versions match what vLLM uses (the vLLM↔trainer NCCL group works; only the *intra-trainer* ZeRO-2 group hangs).

Topology when it works:

- Node 0 (4 GPUs): vLLM `--tensor-parallel-size 4`, one URL.
- Nodes 1-3 (12 GPUs): trainer DDP + DeepSpeed ZeRO-2 (12 ranks), optimizer state sharded.

ZeRO-2 is mandatory at this size — a 7B model with Adam fp32 master+m+v in plain DDP would need ~120–160 GiB per GH200; ZeRO-2 cuts that to ~30 GiB per rank by sharding the optimizer state across the 12 ranks. (See README "Subsequent work / Scaling".)

The sbatch fans the trainer out with one `srun --nodes=3 --ntasks=3 --ntasks-per-node=1` step. `$SLURM_NODEID` (0..2 within the step) is read inside the bash body as `--machine_rank`, and `$HEAD_TRAINER` (the first trainer node's hostname) is passed as `--main_process_ip`. Compute-node DNS resolves hostnames directly, so no `getent hosts` plumbing was needed.

```bash
sbatch examples/configs/slurm_4node_16gpu.sbatch
# overrides:
sbatch --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct,MAX_STEPS=2 \
    examples/configs/slurm_4node_16gpu.sbatch
```

## Gotchas hit (worth memorising)

These all bit during this session — none are in 003 because none had been seen yet.

1. **Harness env clobbers sbatch defaults.** The Claude Code harness exports `MODEL=<assistant-model-id>`. `sbatch --export=ALL` propagates it into the job, clobbering any script with `${MODEL:-…}`. Job 202543 died in 24 s because vLLM tried to load `claude-opus-4-7`. Fix: `env -u MODEL sbatch --export=ALL,…` (don't use `--export=NONE` — it strips PATH and `execve(bash)` fails on the compute node).

2. **Compute nodes have no/unreliable HF hub access.** The 2-node smoke failed on rank 0's *ref_model* load (the second `from_pretrained` call inside `GRPOTrainer.__init__`) with `OSError: ... pytorch_model.bin or model.safetensors`. The cache had everything; the policy model loaded fine. Concurrent rank access + flaky hub lookups → transient None from `cached_file`. Fix: `export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1` inside both srun bash bodies. Pre-populate the cache from a login node first.

3. **Cross-node NCCL needs the OFI modules loaded *inside the srun shell*.** The cluster NCCL and aws-ofi-nccl modules don't propagate from the login session into the compute-node srun subshell. Without them, NCCL falls back to sockets (slow at best, hangs `init_communicator` at worst). Adding the `module load` lines into both srun bash bodies was the fix.

4. **`vllm_base_url` was rollout-only.** TRL's `GRPOTrainer` opens its own vLLM client for the NCCL weight-sync handshake using `GRPOConfig.vllm_server_host` / `vllm_server_port`, which every example hardcoded to `localhost`/`8000`. Fatal for any remote-vLLM topology. Fixed centrally in `src/inspect_rl/trainer.py:135` with a 4-line `urlparse(vllm_base_url)` override — examples' hardcoded values are now overwritten right after the run-dir broadcast, so callers don't have to remember.

5. **`uv sync` can leave torch partially installed.** On this filesystem (Lustre `$SCRATCH` for the uv cache, NFS-style `$HOME` for the venv), uv's hardlink fallback can drop files. `import torch` failed with `ModuleNotFoundError: No module named 'torchgen'`, then after copying torchgen, `torch._higher_order_ops`. Fix: `uv sync --reinstall-package torch --link-mode=copy`. (Do *not* `pip install torchgen` — there's a name-squatted PyPI package by that name containing unrelated GAN/VAE code.)

6. **deepspeed reads `CUDA_HOME` at import time.** `inspect_rl/__init__.py` auto-detects it before the deepspeed import in the normal CLI path, but the sbatch sets `CUDA_HOME=${CUDA_HOME:-/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda}` explicitly in each srun shell to make the chain robust.

7. **deepspeed JIT-compiles FusedAdam unless told otherwise.** Compute nodes here lack a new enough GCC + ninja to build the op, and the failure path inside deepspeed deadlocks ranks rather than raising clean. Mitigations baked into the 4-node example: `"optimizer.type": "AdamW", "torch_adam": true, "adam_w_mode": true` in `deepspeed_zero2.json`; `DS_BUILD_OPS=0`, `DS_SKIP_CUDA_CHECK=1` in the srun env. (Did not fully resolve the hang — see the unresolved 4-node section above.)

## Smoke recipe

```bash
# From a clean checkout on Slurm HPC login node:
unset UV_EXCLUDE_NEWER
uv sync --link-mode=copy
mkdir -p slurm_logs   # one-time; sbatch --output needs it pre-existing

# Pre-populate HF cache (login node only — compute nodes can't reach the hub).
.venv/bin/python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    m='Qwen/Qwen2.5-7B-Instruct'; AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m)"

# 2-node smoke (3B):
env -u MODEL sbatch --export=ALL,MAX_STEPS=2,OFF_POLICY_STEPS=0,ENFORCE_EAGER=1 \
    examples/configs/slurm_2node_8gpu.sbatch

# 4-node smoke (7B + ZeRO-2):
env -u MODEL sbatch --export=ALL,MAX_STEPS=2,OFF_POLICY_STEPS=0,ENFORCE_EAGER=1 \
    examples/configs/slurm_4node_16gpu.sbatch
```

Logs land under `slurm_logs/<datetime>_<jobid>/{master.out, vllm.out, trainer*.out}`. Tail `vllm.out` and `trainer*.out`. Pass when `[summary] 2 steps in …` is in the trainer log and exit is `COMPLETED 0:0`.

**Sensitive-data note.** Avoid `sbatch --export=ALL` on its own when your parent shell has secrets (API keys, AWS Secrets Manager ARNs, etc) — deepspeed's multi-node launcher serialises the propagated env into `$PWD/.deepspeed_env` on disk so remote ranks inherit it, which on a shared filesystem is leakage. The repo gitignores the file, but it still sits on Lustre. Either submit with `--export=NONE,PATH=...,VAR=...` listing only what's needed, or run under an env with the secrets unset.
