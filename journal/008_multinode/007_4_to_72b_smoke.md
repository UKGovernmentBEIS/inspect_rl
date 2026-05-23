# 007 — Landing math-agent across Qwen2.5 {7B → 72B}

## TL;DR

Working 4-node + 8-node configs end-to-end on Slurm HPC for Qwen2.5
{7B, 14B, 32B, 72B}. PR #23 lands as the single point fix, replacing
the FSDP path with DeepSpeed ZeRO-3 for ≥7B. Smoke matrix proves the
networking, vLLM serve, and trainer gradient/weight-sync paths all
hold up at full size.

## Smoke matrix

| Model | Topology | Steps | Weight-sync / step | Result |
|---|---|---|---|---|
| 7B  | 4-node, 12-rank ZeRO-3 | 10 (then unrelated hang) | ~90 s  | heldout correctness 0.19 → **0.78** |
| 14B | 4-node, 12-rank ZeRO-3 | 3 / 3 | ~177 s | heldout correctness 0.34 → 0.53 |
| 32B | 4-node, 12-rank ZeRO-3 | 3 / 3 | ~415 s | rollout correctness 0.71 → 1.00 |
| 72B | 8-node, 28-rank ZeRO-3 | 3 / 3 | ~930 s | rollout correctness 0.29 → 0.64 |

Wall-clock for one full step scales roughly linearly with model size:
**90 s / 177 s / 415 s / 930 s** at 7B / 14B / 32B / 72B — i.e. the
ZeRO-3 weight gather + vLLM push dominates and is bottlenecked by
parameter count.

## What was broken

Six independent bugs, each surfaced as the previous one was fixed. In
encounter order:

1. **Ref-model `device_map` mismatch under FSDP.** TRL's
   `create_model_from_path` defaults `device_map="auto"` and TRL only
   force-overrides that for `MULTI_GPU`/`DEEPSPEED`, never FSDP. The
   ref model was auto-sharded across the rank's visible GPUs, then
   `prepare_fsdp(device_id=cuda:0)` crashed with
   `Inconsistent compute device and device_id: cuda:1 vs cuda:0`.
2. **FSDP1 + reentrant gradient checkpointing.** Each checkpointed
   segment ran as if it were the FSDP root, so on the next outer
   forward FSDP's `_lazy_init` found nested instances with
   `_is_root=True` and asserted.
3. **FSDP1 multi-process-per-node device confusion.** With 4 ranks on
   a trainer node and `CUDA_VISIBLE_DEVICES=0,1,2,3`, TRL's slim
   `prepare_fsdp` helper didn't pin ranks to their own GPU and the
   ref model copies all piled onto `cuda:0` (~22 GiB × 4 = 88 GiB).
   Not chased further — switched to DS ZeRO-3 instead, which has
   never had this class of bug on this stack.
4. **DS torch_adam in wrong slot.** `torch_adam: true` was at the top
   of the `optimizer` block, but DS reads it from `optimizer.params`
   (`engine.py:1660`). Without it, DS fell through to FusedAdam and
   JIT-compiled a CUDA op that fails on aarch64.
5. **AdamW fp32 state under ZeRO-2.** Even with the correct
   `torch_adam`, `torch.optim.AdamW` allocates fp32 m + v for the
   *full unsharded* param set on every rank, because ZeRO-2 only
   shards grads + master weights. 7B × 8 bytes × 2 = 112 GB / rank,
   OOM before step 1. ZeRO-3 fixes this — local optimizer only sees
   its 1/N shard. The accelerate config also needs
   `zero3_init_flag: true` so the model is constructed already-sharded.
6. **Fp32 model load.** Even after sharding, policy + ref each
   loaded in fp32 (28 GB for 7B → 56 GB / rank baseline). Setting
   `model_init_kwargs={"device_map": None, "dtype": "bfloat16"}` halves
   to 14 + 14 GB / rank and the run fits.

## Tooling cleanup discovered along the way

Three deployment-specific env vars must be cleared before any
`from inspect_ai` import or the run aborts at startup when
`<redacted: telemetry/auth provider package>` isn't installed:

- `INSPECT_TELEMETRY` (already popped — routes logs to CloudWatch,
  PutLogEvents 408s spam 50-line tracebacks per sample)
- `INSPECT_API_KEY_OVERRIDE` (key-rotation provider lookup)
- `INSPECT_REQUIRED_HOOKS` (hard-fails the startup hook check)

`<redacted: telemetry/auth provider package>` itself is now removed from
the optional extra — its telemetry tried to call AWS from compute nodes
and dumped ~hundreds of botocore tracebacks per step. Re-add once it
gains an offline mode.

## Open follow-ups (not in this PR)

1. The 7B run hung at step 11 inside `inspect_eval` after the
   step-10 checkpoint. Looks like the agent enters a long-running
   parse-error retry loop when the policy produces malformed
   `<tool_call>` blocks. Independent of the FSDP/DS work. Plan: lower
   `message_limit` + add a per-sample timeout to the rollout's
   `inspect_eval` call.
2. Lift `model_init_kwargs={"device_map": None, "dtype": "bfloat16"}`
   into `inspect_rl/trainer.py` defaults for any distributed run, so
   other examples get it for free.
3. `examples/configs/hpc1/4node/accelerate_zero2.yaml` is a misnomer (it
   runs ZeRO-3). Rename in a follow-up.
4. Filing an upstream issue against TRL: extend the
   `MULTI_GPU / DEEPSPEED → device_map=None` override
   (`trl/trainer/grpo_trainer.py:296-298`) to cover FSDP too.
