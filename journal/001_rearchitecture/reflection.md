# Reflection: Inspect-owned rollout rearchitecture

## What we built

Three core files that flip inspect-rl's architecture from "TRL generates, Inspect scores" to "Inspect owns the full rollout, TRL just does gradients":

- **`trl_vllm_provider.py`** — Custom Inspect `ModelAPI` (`@modelapi("trl-vllm")`) that talks to TRL's vLLM `/chat/` endpoint. Auto-batches concurrent `generate()` calls from Inspect's eval loop into single HTTP requests. Captures prompt_ids, completion_ids, and logprobs in `ModelOutput.metadata["trl_completion_data"]`.

- **`rollout.py`** — Factory that returns a TRL `RolloutFunc`. Each call reconstructs Inspect `Sample`s from TRL's prompt dicts (restoring targets via content lookup), runs `eval_async` with the full solver chain + scorers, then extracts token-level data and per-sample scores. Returns the dict TRL expects.

- **`trainer.py`** — `inspect_rl_train()` wires a stock `GRPOTrainer` with the rollout func and reward functions that read pre-computed scores from `kwargs["inspect_scores"]`.

Plus a GSM8K example with xmlcount + correctness scorers, and justfile recipes for the two-terminal workflow.

## What we learned from TRL's internals

The plan originally called for a `GRPOTrainer` subclass and a custom vLLM server. Reading TRL 0.29's source revealed neither was needed:

- **`rollout_func` gets structured prompts directly** (line 1563 of `grpo_trainer.py`). When `rollout_func` is set, TRL skips tokenization and passes the raw message lists through. No subclass needed.

- **TRL's built-in `trl vllm-serve`** already provides `/chat/`, `/health/`, and NCCL weight sync. No custom server needed.

- **Extra fields flow to reward functions** via `extra_fields` merge (line 1869-1874). But the merge does `inp[key] = values[i]` — so extra fields must be a **list of per-sample values**, not a dict of lists. This was a bug we caught by reading the source before running.

## What blocked the end-to-end test

We never got to a successful training run. The blockers were all version compatibility:

1. **`python -m trl` doesn't work** — TRL's CLI is `trl vllm-serve`, not `python -m trl vllm-serve`. Cost ~5 minutes.

2. **vLLM 0.15.0 doesn't support Gemma 4** — `gemma4` model type not in the model registry. Would need vLLM ~0.17+.

3. **Gemma 3 + transformers 5.5.4 + vLLM 0.15.0** — Gemma 3's multimodal processor triggers a lazy import of `mistral_common.ReasoningEffort` which doesn't exist in the installed `mistral_common` version. The error is deep in vLLM's worker initialization path.

4. **transformers override pin** — `pyproject.toml` has `override-dependencies = ["transformers==5.0.0"]` which conflicts with wanting newer transformers for newer models. This creates a tension: vLLM pins old transformers, new models need new transformers.

The user resolved this by bumping vLLM to 0.19.0 and updating the wheel index URL. That should unblock Gemma 4 support.

## What's left

- Run `uv lock && uv sync` with the new vLLM 0.19.0 + verify it resolves cleanly
- Start the server, confirm health endpoint responds
- Run training for a few steps, verify the full data flow works
- Check W&B for reward curves

## Design decisions worth revisiting

- **`asyncio.run()` in rollout_func**: TRL calls rollout_func synchronously from its training loop. We use `asyncio.run(_rollout_async(...))` to bridge into Inspect's async eval. This works but creates a new event loop each call. If TRL ever calls from an async context, this will break. Could use `nest_asyncio` or check for a running loop.

- **Sample lookup by content string**: We match TRL's prompts back to original Inspect samples by user message content. This is fragile if two samples have identical user messages but different targets. A hash-based or ID-based approach would be more robust.

- **Single eval per rollout call**: Each rollout call runs one `eval_async`. For agentic tasks with sandboxes, this means spinning up sandboxes on every training step. Batching multiple steps' worth of evals or keeping sandboxes warm would help throughput.
