# Rearchitecture: Inspect-owned rollouts for agentic RL

## Context

inspect-rl currently has a single-turn reward shim (`scorer_to_reward_func` in `grpo.py`) that fakes a `TaskState` from a raw completion string. The solver chain never runs — no tools, no sandbox, no multi-turn. This makes agentic evals (sandboxed Linux environments, tool use) impossible.

The target architecture (documented in `~/scratch/learning/2026-04-18-agent-security/journal/005_rl_with_inspect/`) flips ownership: **Inspect owns the rollout**, TRL becomes a gradient engine consuming pre-computed token IDs, logprobs, and scores. Any Inspect eval — including multi-turn agentic tasks with k8s sandboxes — becomes a valid RL reward signal.

**Goal:** Rewrite inspect-rl's core to implement this architecture. Verify on a single 4×GH200 node with a training run that completes in ~1 hour.

## Architecture

```
Trainer process (4× GH200)          vLLM server process (separate)
┌─────────────────────────┐         ┌──────────────────────────┐
│ InspectGRPOTrainer      │         │ vllm_serve.py            │
│   ↓                     │  HTTP   │   /chat/  → token data   │
│ rollout_func(prompts)───┼────────▶│   /health/               │
│   ↓ eval_async(task)    │         │   weight sync endpoints  │
│   ↓ extract scores +    │  NCCL   │                          │
│     token data          │────────▶│   receive updated weights│
│   ↓                     │         └──────────────────────────┘
│ GRPO loss + update      │
└─────────────────────────┘
```

## Files to create/modify

### New files (core)

1. **`src/inspect_rl/trl_vllm_provider.py`** — Custom Inspect `ModelAPI`
   - `@modelapi(name="trl-vllm")` registration
   - `generate()` → enqueues to batch thread → POST to `/chat/` → returns `ModelOutput` with `metadata["trl_completion_data"]`
   - Auto-batching via background thread (collect requests within timeout window, single HTTP call)
   - Message conversion: `ChatMessage` → TRL dict format

2. **`src/inspect_rl/rollout.py`** — Rollout function factory
   - `make_inspect_rollout_func(task) -> RolloutFunc`
   - Builds content→Sample lookup tables at creation time
   - Each call: reconstruct Samples from prompts, `task_with(task, dataset=MemoryDataset(...), epochs=G)`, `eval_async()`, extract token data + scores, reorder by sample ID and epoch
   - `_inspect_dataset_to_hf(dataset)` — convert Inspect Dataset to HF format for TRL's dataloader

3. **`src/inspect_rl/trainer.py`** — Trainer subclass + entry point
   - `InspectGRPOTrainer(GRPOTrainer)` — override `_generate_single_turn` to preserve structured prompts via `gather_object()`
   - `_make_scores_reward_func(scorer_name)` — looks up pre-computed scores from `kwargs["inspect_scores"]`
   - `inspect_rl_train(task, model, config, vllm_server_host, peft_config=None)` — entry point wiring everything together

4. **`src/inspect_rl/vllm_serve.py`** — Minimal vLLM server
   - FastAPI app with `/chat/` (batched conversations → `{prompt_ids, completion_ids, logprobs}`)
   - `/health/` endpoint
   - `/init_communicator/`, `/update_named_param/` for NCCL weight sync
   - `WeightSyncWorkerExtension` for vLLM workers
   - CLI entry point: `python -m inspect_rl.vllm_serve --model ... --tensor_parallel_size 4`

5. **`src/inspect_rl/example/gsm8k_v2.py`** — Test example using new architecture
   - GSM8K with xmlcount + correctness scorers (from journal)
   - Uses `generate()` solver so the full Inspect pipeline runs
   - `inspect_rl_train()` call with LoRA config for 8B model

6. **`configs/gsm8k_grpo.yaml`** — GRPOConfig for the test run

### Files to modify

7. **`src/inspect_rl/__init__.py`** — Export `inspect_rl_train`, register provider
8. **`src/inspect_rl/grpo.py`** — Keep for backwards compat but mark deprecated; new code uses `trainer.py`
9. **`pyproject.toml`** — Add `deepspeed>=0.18.5`, `peft`, `httpx` deps
10. **`justfile`** — Add `train` and `serve` recipes

## Implementation order

### Step 1: vLLM server (`vllm_serve.py`)
Write the minimal server first — it's the dependency everything else talks to. Start with `/chat/` and `/health/`. Weight sync endpoints can come next (TRL's built-in NCCL sync should work with vLLM 0.15's `worker_extension_cls`).

Verify: `python -m inspect_rl.vllm_serve --model meta-llama/Llama-3.1-8B-Instruct --tensor_parallel_size 4` starts and `/health/` responds.

### Step 2: TRL vLLM provider (`trl_vllm_provider.py`)
Custom Inspect ModelAPI that talks to the server from Step 1. The auto-batching thread is important — Inspect issues many concurrent `generate()` calls during `eval_async`.

Verify: In debug notebook, instantiate provider and call `generate()` against running vLLM server.

### Step 3: Rollout function (`rollout.py`)
The core glue. `make_inspect_rollout_func(task)` returns a callable matching TRL's `RolloutFunc` signature.

Key detail: TRL 0.29's `_generate_single_turn` now takes `(prompt_ids, images, multimodal_fields)` not message dicts. The rollout function receives `list[str]` prompts from TRL. We need to handle this — the `InspectGRPOTrainer` must stash structured prompts before the base class tokenizes them.

### Step 4: Trainer + entry point (`trainer.py`)
`InspectGRPOTrainer` subclass + `inspect_rl_train()`. Wire rollout func, reward funcs, dataset conversion.

Verify: Instantiate trainer, check it creates correctly.

### Step 5: GSM8K example + config
Write the example task and YAML config. Wire up the launch flow:
```bash
# Terminal 1: vLLM server
just serve

# Terminal 2: Training
just train
```

### Step 6: End-to-end test
Run the full training loop. Target: Llama-3.1-8B-Instruct with LoRA (r=16), GSM8K, ~200 steps, 8 generations per prompt. Should complete in <1 hour on 4×GH200.

## Key decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Model | Llama-3.1-8B-Instruct | Fits easily in 4×GH200; well-tested with LoRA |
| Eval | GSM8K with format+correctness scorers | Good RL signal, fast to score, no sandbox needed for first test |
| Training method | LoRA (r=16) | Fast weight sync (~28MB vs 16GB), lower memory |
| Generations per prompt | 8 | Reasonable for group-relative advantages, keeps batch tractable |
| vLLM tensor parallelism | 2 (2 GPUs for vLLM, 2 for trainer) | Balance inference throughput vs training compute |
| Weight sync | Standard NCCL (TRL built-in) | Simplest starting point; batched sync is an optimization for later |

## What we're NOT doing (yet)

- K8s sandbox integration (no sandbox needed for GSM8K — that's a follow-up)
- Multi-node coordination
- Overlap generation / prefetch
- Batched NCCL weight sync optimization
- AsyncCheckpointEvaluator
- The old single-turn shim in `grpo.py` stays untouched (deprecated, not deleted)

## Verification

1. `just lint` passes
2. vLLM server starts and responds to `/health/`
3. Provider can generate completions against running server (debug notebook cell)
4. Full training loop runs: `just train` completes ~200 steps
5. W&B shows reward curves trending upward
6. Model outputs show improving format compliance + answer accuracy
