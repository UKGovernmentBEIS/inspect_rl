# inspect_rl.core

The GRPO training loop and its non-optional pieces. Remove any of these and
training stops working.

| Module | Role |
|---|---|
| `trainer.py` | `inspect_rl_train()` — wires task, rollout, vLLM weight-sync, GRPOTrainer. |
| `rollout.py` | Adapter from Inspect's eval pipeline to TRL's `RolloutFunc` (token IDs + logprobs + scores). |
| `trl_vllm_provider.py` | Inspect `@modelapi("trl-vllm")` provider — generation goes through TRL's vLLM server. |

**Does not belong here:** stateless helpers (logging, output-dir resolution,
CUDA bootstrap) → `inspect_rl.util`. Optional performance wrappers around the
rollout path → `inspect_rl.perf`.
