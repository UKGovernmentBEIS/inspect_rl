# 01 — Features

*Last updated: 2026-05-13*

What `inspect-rl` does on top of TRL's stock `GRPOTrainer`.

## Resume from checkpoint

Pass a previous run directory to pick up where you left off. Optimizer, scheduler, and step counter are restored; eval logs past the checkpoint step are cleaned up to avoid stale results; and the W&B run ID is read from `manifest.json` so metrics continue on the same chart.

```python
inspect_rl_train(task=task, model="my-model", grpo_config=config, resume_from="outputs/math_agent/2026-04-21-16-19-56")
```

## Multi-turn agent support with gradient masking

Inspect's solver chain can run multi-turn agents (`basic_agent`, `react`) with tool calls, sandboxed environments, and retries. The rollout aggregator concatenates all turns into a single token-ID sequence and builds an `env_mask` that zeroes out environment tokens (tool responses, system messages) so only policy-generated tokens contribute to the GRPO gradient.

## Client-side tokenization (BPE drift prevention)

The custom vLLM provider talks to `/generate/` with pre-tokenized prompts instead of `/chat/`. This avoids BPE re-encoding drift — where `detokenize → re-tokenize` produces different token IDs — which would break the per-token importance ratio alignment that GRPO requires. Per-turn token data is stashed on message metadata so it survives through arbitrary solver chains.

## Periodic validation eval

A callback runs `inspect_eval` on a held-out sample set every `eval_steps` steps, logs metrics to W&B and `train.log`, and writes structured `.eval` logs to `eval_logs/NNN_val/`. Validation evals are non-fatal (`fail_on_error=False`) so a single bad sample won't crash training.

## Off-policy prefetch (auto by default)

The trainer is idle while vLLM generates and vice versa. `FreshestPrefetchRolloutFunc` breaks that lock-step by running a background thread that continuously regenerates rollouts and keeps only the freshest one in a single slot. When the trainer is ready for the next step, it grabs whatever the producer most recently finished; older completed rollouts are discarded. vLLM stays busy whenever the trainer is busy — and the trainer never gets handed a stale FIFO leftover.

Off-policy correction comes for free: TRL's `GRPOTrainer` already applies truncated importance sampling on `(current_logp − sampling_logp)` capped at `vllm_importance_sampling_cap=3.0`, so the staleness budget is automatic.

The `--off-policy-steps` flag is effectively a switch:

```bash
irl train math-agent                         # default: auto-decide
irl train math-agent --off-policy-steps 0    # disable, fully synchronous
irl train math-agent --off-policy-steps 1    # force prefetch on
```

In auto mode (`-1`, the default) the trainer runs three warmup steps synchronously while measuring rollout time and weight-update time, then enables prefetch if `T_rollout > T_train` — or stays synchronous if the trainer is already the bottleneck (because vLLM would be forced to idle anyway). The decision is announced on the console:

```text
[off-policy] auto depth=1 (T_rollout=12.4s T_train=5.6s)
```

A one-line summary is emitted when the run finishes:

```text
[summary] 20 steps in 184.2s · rollout avg 12.4s · train avg 5.6s · off-policy: prefetch on · 7 stale rollouts discarded
```

The discarded count is the number of rollouts the producer finished but the trainer never popped (a newer one overwrote them). A non-zero count means the producer was outpacing the trainer — exactly the regime the freshest-only design is meant for.

The wrapper ignores TRL's per-call `prompts` and iterates its own cursor over the same HF dataset — TRL re-derives prompt text from the returned `prompt_ids` so this is self-consistent, but the trained-on order differs from TRL's, so per-prompt deterministic replays are not preserved. Background reading in [`journal/007_faster_rl/003_t2_design.md`](../journal/007_faster_rl/003_t2_design.md).

## GRPO algorithmic optimisations (OLMo 3 / DAPO)

Two small additions on top of TRL's `GRPOTrainer` reduce wasted gradient steps and let the policy take larger positive updates. Both are on by default in the bundled examples; see [`journal/007_faster_rl/`](../journal/007_faster_rl/) for the research notes.

**Active sampling (zero-gradient resampling).** GRPO's advantage is computed per group of `num_generations` completions. When all completions in a group receive identical scores the advantage is zero — the prompt contributes nothing to the gradient and the inference cost is wasted. After the initial rollout the trainer detects such groups (`rollout.py:_find_zero_gradient_groups`) and re-runs `inspect_eval` over just those prompts, up to `max_resample_rounds` times. Even with `--resample-rounds 0` the detection still logs zero-gradient groups so you can see how much compute is being wasted before turning resampling on. OLMo 3 reports ~4× fewer wasted steps from this alone; the win is largest on easy datasets or late in training when the model gets most prompts entirely right or entirely wrong.

```bash
irl train math-agent --resample-rounds 3      # default for the examples
irl train math-agent --resample-rounds 0      # off, but still logs the wastage
```

**Asymmetric clipping.** Each example's `GRPOConfig` sets `epsilon_high=0.28` (TRL default is `0.2`, symmetric with `epsilon`). The upper PPO clip bound is widened ~40% so the policy can take larger steps toward high-reward completions; the lower bound stays at `0.2` to constrain regressions. Standard OLMo 3 / DAPO convention — best understood as the policy being allowed to move faster in the right direction while still being held back from moving fast in the wrong one.

**Token-level loss** is already the TRL 1.2.0 default (`loss_type="dapo"`) so no change was needed. It normalises by total active tokens in the batch rather than per-sequence, which removes the length bias that would otherwise reward shorter completions disproportionately.

## Structured logging

Two log layers keep things readable: `train.log` captures the trainer at INFO+ (step timings, eval summaries, errors) while per-step `eval.log` files capture `inspect_ai` at INFO for that eval only. Outside of eval windows, `inspect_ai` is pinned to WARNING to avoid flooding the trainer log.

## Output directory resolution

Run artifacts (checkpoints, eval logs, `train.log`) land under `outputs/<example>/<timestamp>/` by default. Relative output paths resolve in this order:

1. `$INSPECT_RL_OUTPUT_ROOT` if set
2. `$SCRATCH` if set (common on Slurm HPC)
3. The nearest enclosing `pyproject.toml` walking up from cwd
4. cwd

Pass an absolute path to `output_dir=` (or the `--output-dir` CLI flag) to bypass the lookup entirely.

## CUDA_HOME auto-detect

`vllm`, `deepspeed`, and `torch` JIT extensions all read `$CUDA_HOME` at torch import time and cache the result — setting it later is too late. `inspect_rl/__init__.py` runs the bootstrap on first import of the package (before anything pulls in torch), so the same auto-detect works whether you reach the package through the `irl` CLI or via `from inspect_rl import inspect_rl_train`.

On NVHPC-based Slurm HPC nodes the probe checks `/opt/nvidia/hpc_sdk/Linux_*/<release>/cuda/` then `/usr/local/cuda`, picking the first directory with a real `bin/nvcc`. The chosen path is logged to stderr as `[inspect_rl] CUDA_HOME=… (auto-detected)`. To override, export `CUDA_HOME` **before** importing the package (or invoking `irl`); a post-import override is silently ignored by torch.

Package import stays light — torch only loads when you actually touch `inspect_rl.inspect_rl_train` or run a training subcommand. `irl --help` and `import inspect_rl` take ~0.3 s.

## Device selection and distributed training

Single-process today: one Python process owns the policy + optimizer, and a separate vLLM server handles generation. `--devices` masks `CUDA_VISIBLE_DEVICES` for the trainer process.

For multi-GPU and multi-node, launch the CLI through `accelerate launch` — see the [Multi-GPU and multi-node](../README.md#multi-gpu-and-multi-node) section in the README. Plain `irl train --num-processes N` is intentionally rejected with a redirect message; we don't re-exec inside the CLI.
