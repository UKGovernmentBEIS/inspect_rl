# example-pkg

Tiny downstream package that depends on `inspect-rl`. Kept in the workspace so a single `uv sync` from the repo root installs both, letting us verify packaging end-to-end without publishing anywhere.

```bash
uv sync                             # from repo root
uv run example-pkg                  # prints a sanity summary
uv run --directory examples/example_pkg irl --help  # inspect-rl CLI works from here too
```

## Smoke run (`example-tldr`) — requires ≥ 2 GPUs

`example-tldr` runs a 5-step TL;DR training loop end-to-end on `google/gemma-3-270m-it`. **Two GPUs are required** even though the model easily fits on one: TRL's weight-sync path opens an NCCL communicator that spans the vLLM-server rank and the trainer rank, and NCCL refuses to bind two distinct ranks to the same physical device. `irl serve` owns GPU 0; the script pins the trainer to GPU 1.

```bash
# Terminal 1 — vLLM on GPU 0, eager mode for fast startup
CUDA_VISIBLE_DEVICES=0 uv run irl serve google/gemma-3-270m-it --enforce-eager

# Terminal 2 — trainer on GPU 1 (the script sets this internally)
uv run example-tldr
```

`--enforce-eager` skips vLLM's CUDA graph capture (~30-60s off startup). Drop the flag for throughput-sensitive real training.

The script sets `CUDA_VISIBLE_DEVICES=0` internally before torch loads, so no extra env var is needed in terminal 2.

To mimic a real external consumer, point `inspect-rl` at a git source instead of the workspace — see the commented block in `pyproject.toml`.
