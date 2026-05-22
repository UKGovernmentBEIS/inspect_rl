# Testing the rearchitecture

## Status

End-to-end smoke test passed on 2026-04-21 — 5 GRPO steps with gemma-3-4b-it, LoRA r=16, full pipeline (rollout → Inspect scoring → reward merge → GRPO loss → LoRA update → NCCL weight sync to vLLM). See `debug.ipynb`.

### What's done
- `src/inspect_rl/trl_vllm_provider.py` — custom Inspect ModelAPI with auto-batching, talks to TRL's `/chat/` endpoint
- `src/inspect_rl/rollout.py` — bridges Inspect eval → TRL's RolloutFunc interface (Jupyter-safe: thread fallback when a loop is already running)
- `src/inspect_rl/trainer.py` — `inspect_rl_train()` entry point, wires GRPOTrainer with Inspect rollout
- `src/inspect_rl/example/gsm8k_v2.py` — GSM8K with xmlcount + correctness scorers
- `justfile` — `serve` and `train` recipes

### Environment gotchas (Slurm HPC)
- **`CUDA_HOME` must be set before importing torch.** DeepSpeed (pulled in transitively by `accelerate`) probes `torch.utils.cpp_extension.CUDA_HOME` at import and fails with `MissingCUDAException`. No `nvcc` is on the default PATH here — use `export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6` (or load the `cudatoolkit/24.11_12.6` module). The `-laio` / `-lcufile` linker warnings from DeepSpeed's op builder are harmless.
- **Single-GPU training for now.** With >1 GPU visible, HF Trainer spreads params across devices but TRL's NCCL weight-sync (`VLLMClient.update_named_param` → `PyNcclCommunicator.broadcast`) asserts the tensor is on the same device as the communicator (`cuda:0`). Set `CUDA_VISIBLE_DEVICES=2` (one GPU) for the trainer; the vLLM server keeps GPUs 0,1 with tp=2.

### Known issues
- **vLLM 0.15.0 + transformers 5.5.x**: Gemma 3 multimodal processor hits `ImportError: cannot import name 'ReasoningEffort' from 'mistral_common'`. Stick with transformers 5.0.0 or text-only models (Llama).
- **vLLM 0.15.0 doesn't support Gemma 4** (`gemma4` model type not in registry). Would need vLLM ≥0.17 or so.
- **TRL warns about vLLM 0.15.0** (officially supports up to 0.12.0). This is fine for our use case since we use a custom `rollout_func` and bypass TRL's built-in vLLM generation path.
- **Gemma-3 requires `token_type_ids` when training.** `Gemma3Model.forward` raises `ValueError: token_type_ids is required as a model input when training` because the bidirectional-image-mask path needs it. TRL's GRPOTrainer never passes one. Monkey-patch `Gemma3Model.forward` to default `token_type_ids=torch.zeros_like(input_ids)` before constructing the trainer (see cell 3 of `debug.ipynb`). Text-only models don't need this.
- **Stale weight-update group on kernel restart.** If a previous trainer instance dies without running its `atexit` hook (common on Jupyter kernel shutdown or mid-cell exceptions), the vLLM server still holds the NCCL communicator from that trainer. The next `init_communicator/` call then fails with `Weight update group already initialized` and the vLLM workers crash, wedging the whole server. Mitigation: `POST /close_communicator/` on the vLLM server before each training run. `debug.ipynb` cell 3 does this; production script runs are fine because the process exits cleanly.

## How to test

### Prerequisites
- 4× GH200 (or any 4-GPU node with ≥40GB per GPU)
- `uv sync` completed
- Model weights cached: `meta-llama/Llama-3.1-8B-Instruct`

### Step 1: Start vLLM server (terminal 1)

```bash
just serve
```

This runs `trl vllm-serve` on GPUs 0-1 with tensor_parallel_size=2. Wait until the health endpoint responds:

```bash
curl http://localhost:8000/health/
```

Startup takes 1-3 minutes (CUDA context + weight loading).

### Step 2: Run training (terminal 2)

```bash
just train
```

This runs `python -m inspect_rl.example.gsm8k_v2 train` on GPUs 2-3. It will:
1. Load GSM8K dataset (7473 samples, capped to 5000)
2. Create GRPOTrainer with LoRA (r=16) and our custom rollout_func
3. Each step: rollout_func runs Inspect eval → generates via TRLVLLMProvider → scores with xmlcount + correctness scorers → returns token data + scores to TRL
4. TRL computes GRPO loss and updates LoRA weights

### Step 3: Verify

- **W&B**: should show `reward/inspect_xmlcount_scorer` and `reward/inspect_correctness_scorer` trending up
- **Logs**: each step should show Inspect eval completing, then TRL loss computation
- **Target**: ~200 steps, should complete in <1 hour

### Customizing the run

```bash
# Different model
just serve model="some/other-model"
CUDA_VISIBLE_DEVICES=2,3 uv run python -m inspect_rl.example.gsm8k_v2 train --model some/other-model

# Fewer steps for a quick smoke test
CUDA_VISIBLE_DEVICES=2,3 uv run python -m inspect_rl.example.gsm8k_v2 train --max_steps 5

# Different batch size / generations
CUDA_VISIBLE_DEVICES=2,3 uv run python -m inspect_rl.example.gsm8k_v2 train \
    --per_device_train_batch_size 2 \
    --num_generations 4
```

## Architecture (data flow)

```
GRPOTrainer.train()
  └─ _generate() calls rollout_func(prompts, trainer)
       └─ rollout.py: _rollout_async()
            ├─ Convert TRL prompts → Inspect Samples (with targets from lookup)
            ├─ get_model("trl-vllm/...") → TRLVLLMProvider
            ├─ eval_async(task_with(task, dataset=samples))
            │    └─ Inspect solver chain runs (generate() → TRLVLLMProvider)
            │         └─ TRLVLLMProvider._execute_batch()
            │              └─ POST /chat/ → vLLM server
            │              └─ Returns ModelOutput with metadata["trl_completion_data"]
            │    └─ Inspect scorers run (xmlcount, correctness)
            ├─ Extract prompt_ids, completion_ids, logprobs from metadata
            ├─ Extract per-sample scores into list[dict[str, float]]
            └─ Return {prompt_ids, completion_ids, logprobs, inspect_scores}

  └─ TRL merges inspect_scores into reward_kwargs (per-sample dicts)
  └─ reward_func reads kwargs["inspect_scores"][i][scorer_name]
  └─ TRL computes GRPO advantages + policy gradient loss
  └─ Backprop through LoRA weights
  └─ NCCL weight sync to vLLM server
```

## Key design decisions (from implementation, differs from plan.md)

- **No custom vLLM server** — TRL's built-in `trl vllm-serve` already has `/chat/`, `/health/`, and NCCL weight sync
- **No GRPOTrainer subclass** — TRL 0.29's `_generate()` passes structured prompts directly to `rollout_func` when set
- **Per-sample score dicts** — `rollout_func` returns `inspect_scores: list[dict[str, float]]` (not `dict[str, list[float]]`) because TRL's `extra_fields` merge does `inp[key] = values[i]` which requires a list
