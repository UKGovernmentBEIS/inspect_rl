# inspect_rl.util

Cross-cutting helpers used by `core` (and consumers) — stateless, no GRPO
knowledge. Safe to import from anywhere.

| Module | Role |
|---|---|
| `display.py` | Compact `[step N] …` progress lines for console/notebook/wandb. |
| `heartbeat.py` | Daemon thread that logs `[heartbeat] step=… phase=…` every N seconds for hang detection. |
| `logging_config.py` | One-shot setup of the `inspect_rl` logger namespace. |
| `run_dir.py` | Resolve an output root from env (`INSPECT_RL_OUTPUT_ROOT`, `$SCRATCH`, …) and create the run dir layout. |
| `_cuda.py` | Autodetect `CUDA_HOME` from NVHPC before torch caches it at import time. |

**Does not belong here:** anything that participates in a training step
(rollout, scoring, weight sync) → `inspect_rl.core`. Anything that changes
performance characteristics → `inspect_rl.perf`.
