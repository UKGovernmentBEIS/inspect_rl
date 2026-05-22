# Inspect RL

**Reinforcement finetuning using [Inspect](https://inspect.aisi.org.uk/) tasks as the reward signal.**

_Disclaimer (May 2026): This is experimental research code and we have not committed to significant maintenance or further development_

Inspect owns the full rollout — solvers, tools, sandboxed environments, multi-turn conversations — while [TRL](https://huggingface.co/docs/trl/index)'s GRPO computes policy gradients. Any Inspect task you can evaluate, you can train on.

## A note on training with evaluation tasks

We use [Inspect](https://inspect.aisi.org.uk/) extensively for evaluating language models, and found it natural to reuse the same task infrastructure for RL training. We want to be explicit about a trade-off this creates: **training on evaluation tasks reduces the informativeness of those same evaluations as a measure of a model's general behaviour**. A model trained with a particular Inspect task as its reward signal will score higher on that task, but the score will no longer reliably reflect out-of-distribution capability.

This tool is best suited for tasks written specifically for training (not re-used from a held-out eval suite), or for research where the distinction between train and eval tasks is carefully managed. If you are using Inspect evals as part of a safety or capability assessment, we recommend keeping those tasks strictly held out from the training pipeline.

## Requirements

- **≥ 2 GPUs.** The trainer and the vLLM server must run on different physical devices. TRL's weight-sync path opens an NCCL communicator spanning both ranks and NCCL refuses to bind two distinct ranks to the same GPU.
- **Linux, CUDA.** Developed on an aarch64 Slurm HPC cluster (NVHPC + H100); should also work on ordinary x86_64 Linux + NVIDIA. macOS and Windows are not supported.
- **AWS on-demand instances** worth trying: `g6.12xlarge`, `g5.12xlarge`, `g4dn.12xlarge`, `g4ad.8xlarge`, `g5a.16xlarge`. T4-based (g4dn) and g5 instances did not work in testing.

## Quick start (from a clone)

```bash
uv sync
```

Two terminals (or `tmux` panes). One for vLLM, one for the trainer. They communicate over HTTP + NCCL.

```bash
# Terminal 1 — vLLM server (GPU 0)
CUDA_VISIBLE_DEVICES=0 uv run irl serve google/gemma-3-270m-it

# Terminal 2 — GRPO training (GPU 1)
CUDA_VISIBLE_DEVICES=1 uv run irl train tldr
```

No need to restart vLLM between training runs — TRL re-syncs weights at the start of each. Only restart if you change models.

## Install as a dependency

```bash
uv add --dev "inspect-rl @ git+ssh://git@github.com/AI-Safety-Institute/inspect-rl.git"
# pin to a ref:
uv add --dev "inspect-rl @ git+ssh://git@github.com/AI-Safety-Institute/inspect-rl.git@main"
```

This installs the `irl` console script and exposes `from inspect_rl import inspect_rl_train`.

A minimal downstream consumer (pyproject + a script that defines its own Task/scorer/GRPOConfig and calls `inspect_rl_train()`) lives in [`examples/example_pkg/`](examples/example_pkg/). `uv run example-tldr` runs a 5-step training loop end-to-end.

## CLI (`irl`)

```bash
irl serve [MODEL]              # start a vLLM inference server via `trl vllm-serve`
irl train tldr          [...]  # single-turn TL;DR
irl train magic-number  [...]  # multi-turn agent sanity test
irl train math-agent    [...]  # multi-turn tool-calling agent on GSM8K
irl train gsm8k         [...]  # single-turn XML output with LoRA
```

`irl serve` takes `--devices`, `--tensor-parallel-size`, `--gpu-memory-utilization`, `--max-model-len`. `irl train <example> --help` lists per-example kwargs (`model`, `output_dir`, `max_steps`, batch size, learning rate, wandb, …). `--help` does not load torch, so it's fast.

```bash
irl serve google/gemma-3-270m-it --devices 0
irl train --devices 1 tldr --max-steps 50
```

## Examples

| Example | Task | Model | What it shows |
|---------|------|-------|---------------|
| [`tldr`](src/inspect_rl/example/tldr.py) | TL;DR summarisation | `gemma-3-270m-it` | Simplest possible setup — single-turn, one scorer |
| [`magic_number`](src/inspect_rl/example/magic_number.py) | Guess a fixed digit | `Qwen2.5-0.5B` | Multi-turn agent sanity test (`react` solver) |
| [`math_agent`](src/inspect_rl/example/math_agent.py) | Math with tools | `Qwen2.5-3B` | Multi-turn tool-calling agent with multiple scorers |
| [`gsm8k`](src/inspect_rl/example/gsm8k.py) | GSM8K, XML output | `Qwen2.5-3B` (LoRA) | Single-turn structured output, PEFT |

```bash
# math-agent is the canonical example for multi-turn + tools.
irl serve Qwen/Qwen2.5-3B-Instruct
irl train math-agent
```

## Multi-GPU and multi-node

The trainer is a normal `accelerate`-launched process — pair it with one vLLM server on a separate GPU. With N trainer ranks you'll need N+1 GPUs total.

```bash
# Terminal 1 — vLLM on GPU 0
CUDA_VISIBLE_DEVICES=0 uv run irl serve Qwen/Qwen2.5-3B-Instruct

# Terminal 2 — 3 trainer ranks across GPUs 1, 2, 3 (single node)
CUDA_VISIBLE_DEVICES=1,2,3 uv run accelerate launch --num_processes 3 \
    -m inspect_rl.cli train math-agent \
    --per-device-train-batch-size 4 --num-generations 4
```

**Multi-node.** Standard `accelerate launch` cluster flags (`--num_machines`, `--machine_rank`, `--main_process_ip`). One vLLM server is shared across all trainer nodes — point them at it with `--vllm-base-url http://<vllm-host>:8000`; `inspect_rl_train()` derives `GRPOConfig.vllm_server_host`/`port` from the same URL so TRL's NCCL weight-sync goes to the right place.

Worked Slurm templates by scale in [`examples/configs/`](examples/configs/):

| Scale | Config | vLLM | Trainer |
|---|---|---|---|
| 1-node trainer + separate vLLM | [`1node/`](examples/configs/1node/) | 1 node, any TP | 1 node × 4 GPUs DDP |
| 2-node combined | [`2node/`](examples/configs/2node/) | 1 GPU, TP=1 | 1 node × 4 GPUs DDP |
| 4-node combined | [`4node/`](examples/configs/4node/) | 4 GPUs, TP=4 | 3 nodes × 4 GPUs = 12-rank ZeRO-2 |

Multi-node + separate vLLM on Slingshot/CXI Slurm HPC clusters (e.g. Cray EX) needs both vLLM and trainer in the **same Slurm job** plus `NCCL_NET=Socket` — full diagnosis in [`docs/04_known-issues.md`](docs/04_known-issues.md) and [`journal/008_multinode/006_vllm_worker_internal_error.md`](journal/008_multinode/006_vllm_worker_internal_error.md). The `2node/` and `4node/` sbatches handle both automatically.

Gotchas (full list in [`docs/04_known-issues.md`](docs/04_known-issues.md)):

- `irl train --num-processes N …` is rejected with a redirect — multi-process runs go via `accelerate launch -m inspect_rl.cli train …`.
- `train.log` lives only on rank 0; stdout is per-rank under `accelerate launch`.
- DDP gives each rank a full copy of the policy + optimizer state. A 3B at bf16 with Adam fp32 master+m+v ends up ~50-70 GiB per GPU. For ≥7B, shard with ZeRO/FSDP — see [`docs/05_roadmap.md`](docs/05_roadmap.md).

## Resume from checkpoint

Every `irl train` example takes `--resume <run-dir>` and picks up where it stopped — optimizer, scheduler, step counter, and wandb run are all restored. Eval logs after the checkpoint step are trimmed so charts don't see stale results.

```bash
# Original run wrote checkpoints/checkpoint-30/ then died at step 33.
irl train math-agent --resume outputs/math_agent/2026-05-15-12-40-05
```

What gets restored:
- Latest `checkpoints/checkpoint-N/` (optimizer state, scheduler, RNG, step N).
- Step counter — training continues at step N+1 until `--max-steps`.
- W&B run via the `wandb_run_id` stored in `manifest.json`. `WANDB_RESUME=must` is used (so a deleted cloud run fails loudly rather than silently splitting metrics), and the first heartbeat after `wandb.init()` flips the server-side state from crashed/cancelled back to running.
- Eval logs (`eval_logs/NNN_*` for `NNN > N`) and post-checkpoint val evals are removed.

What does *not* get restored:
- vLLM state. Restart `irl serve` with the same model before resuming; TRL re-syncs weights at step 1.

## Heartbeat

The trainer's main process emits one log line every 30 seconds while training runs:

```
[heartbeat] step=21 phase=eval elapsed_in_phase=125s
```

Phases cycle through `rollout`, `training`, `eval`, `shutdown`. A monitoring agent (or `tail -f train.log`) can treat absent heartbeats for more than ~90 s as evidence the trainer is wedged — typically inside `inspect_eval`'s parse-error retry loop on bad tool-call JSON, or a stalled NCCL weight sync.

## Writing your own task

```python
from inspect_ai import Task
from inspect_ai.dataset import hf_dataset
from inspect_ai.solver import generate, TaskState
from inspect_ai.scorer import scorer, Score, Scorer, Target, accuracy
from trl import GRPOConfig
from inspect_rl import inspect_rl_train

@scorer(metrics=[accuracy()])
def my_reward() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        # Full access to conversation history, tool calls, etc.
        return Score(value=1.0 if "correct" in state.output.completion else 0.0)
    return score

task = Task(
    dataset=hf_dataset("my-org/my-dataset", split="train", ...),
    solver=[generate()],
    scorer=[my_reward()],
)

grpo_config = GRPOConfig(
    output_dir="outputs/my_task",
    use_vllm=True,
    vllm_mode="server",
    max_steps=200,
    bf16=True,
    report_to="wandb",
)

inspect_rl_train(task=task, model="my-model", grpo_config=grpo_config)
```

## Slurm HPC setup

Developed against an aarch64 (NVHPC + GH200/H100) Slurm HPC cluster. The trainer picks up `$SCRATCH` automatically so training artifacts land on fast scratch storage.

Endpoint config (W&B host, EKS cluster, k8s namespace) is read from environment variables — copy [`.env.example`](.env.example) to `.env` and fill in your values. Then:

```bash
cp .env.example .env && $EDITOR .env
set -a; source .env; set +a

# NVHPC toolchain — adjust path for your install. Auto-detected by the
# package when unset (see docs/01_features.md).
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6

uv sync
just setup-slurm-hpc                 # kubectl + EKS kubeconfig (arm64 kubectl)
uv run wandb login --host "$WANDB_BASE_URL"
```

Slurm + multi-node specifics in [`journal/008_multinode/004_cluster_smoke.md`](journal/008_multinode/004_cluster_smoke.md).

## Further reading

Numbered in suggested reading order:

- [`docs/01_features.md`](docs/01_features.md) — Resume, prefetcher, GRPO mods (OLMo 3 / DAPO), masking, output dir resolution, `CUDA_HOME` auto-detect.
- [`docs/02_artifacts.md`](docs/02_artifacts.md) — Output directory layout, log file conventions.
- [`docs/03_internals.md`](docs/03_internals.md) — System architecture diagram, `RolloutFunc`, why token IDs not text, the custom vLLM provider, "where this could be simpler".
- [`docs/04_known-issues.md`](docs/04_known-issues.md) — Known issues and workarounds (GPU sharing, stale NCCL, OOM, Gemma-3, …).
- [`docs/05_roadmap.md`](docs/05_roadmap.md) — Open threads: scaling to bigger models, inflight weight updates (OlmoRL/VCPO), curriculum learning.
- [`journal/`](journal/) — Project journal, one-min reads, one folder per workstream.

## Dev

```bash
just lint           # ruff format + check
just test           # pytest
just smoke          # 2-GPU end-to-end: vLLM on GPU 0, 1 trainer step on GPU 1 (~3 min)
```

`just smoke` is the lightest possible end-to-end check — it boots vLLM with
`gemma-3-270m-it`, runs a single GRPO step on the TL;DR task, and tears
vLLM down. Pass GPU indices if 0/1 are busy: `just smoke 2 3`.

## Coding agent / Claude Code guidance

- Docstrings short and to the point; prefer inline comments close to usage.
- Use typehints, but `Any` is fine when stubs are more trouble than they're worth.
- `uv` for everything — `uv run` and `uv add`.
- If you have github mcp available, default to that over gh cli
- Lint with `just lint`. Run `just test` before pushing.
- Project journal lives in `./journal/` — entries should be readable in 1 min, skimmable in 10 s.
- Use the `nb` MCP server to interface with jupyter notebooks (e.g. `journal/001_rearchitecture/debug.ipynb`). It auto-starts a kernel from the project venv.
- For interactive python (reading eval logs, debugging libraries) prefer notebooks — they avoid re-starting a python process each cell. Create them in journal directories.
- Bias towards imports at top of file (unless very optional or slow and rarely needed).
- If you're in a tmux pane, sibling panes are useful for managing vllm and running debug training runs.

### Research workflow (RL / training)

- Skim recent journal/ entries for context on where we are at with development
- **Always eval on a fixed N-sample heldout set across steps.** Per-step rollouts sample from a rotating pool and cannot distinguish learning from problem-difficulty noise. Before concluding anything from a curve, check `n`, `uniq_targets`, and whether the scorer read from `tool_calls` or free completion text.
- **Scorers that fall back to regex over free text create reward shortcuts** — the policy will satisfy them without using tools. Walk `state.messages[].tool_calls` instead.
- **Training stdout is a flood.** The per-step `inspect_eval` in `rollout.py` is the worst offender; it overwhelms the Jupyter IOPub socket and the MCP client, which then blocks the publisher and hangs training. Fixes: set rollout `inspect_eval` to `display="none"`, `GRPOConfig(disable_tqdm=True)`. The compact `[step N] …` line from `inspect_rl/util/display.py` is enough visibility.
- **Monitor by polling the filesystem**, not by streaming training stdout. `val_eval_logs/` and `eval_logs/` grow as the run progresses; size + count is a reliable liveness + progress signal.
- **vLLM's NCCL communicator survives across trainer restarts.** Always `curl -sf -X POST http://localhost:8000/close_communicator/ -d '{}'` between runs. Switching model requires killing vLLM entirely.
- **Bump model size before tuning rewards.** For POCs, one jump (0.5B→3B) beats ten reward-weight changes. Qwen2.5-3B on GSM8K is the clean-curve threshold; 0.5B/1.5B are unstable without SFT warmup or curriculum.
- **KL regularisation (`beta`) matters.** `beta=0` is the TRL default and drifts policies into noise; `beta=0.01` is weak for narrow per-step signals; `beta=0.05` is a reasonable floor when you can only afford 1–2 prompts per grad step.

Reference links:

- <https://inspect.aisi.org.uk/llms.txt>
- <https://huggingface.co/docs/trl/index>
- <https://docs.vllm.ai/en/stable/>
- <https://inspect.aisi.org.uk/evals/>
- <https://just.systems/man/en/>
