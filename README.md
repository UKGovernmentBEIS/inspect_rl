# Inspect RL

_Disclaimer (May 2026): This is experimental research code and we have not committed to significant maintenance or further development._

**An integration between [Inspect](https://inspect.aisi.org.uk/) (many evals, transcript viewer, sandboxed environments) and [TRL](https://huggingface.co/docs/trl/index) (GRPO policy-gradient training) — any Inspect task you can evaluate, you can train on.**

## A note on training with evaluation tasks

We (UK AISI) use Inspect extensively for evaluating language models, and found it natural to reuse the same task infrastructure for RL training. **Training on evaluation tasks reduces the informativeness of those same evaluations as a measure of a model's general behaviour** — a model trained against an Inspect task as its reward signal will score higher on that task, but the score will no longer reliably reflect out-of-distribution capability. Best suited for tasks written for training, or for research where train/eval separation is carefully managed. If you're using Inspect evals as part of a safety or capability assessment, keep them strictly held out.

## Requirements

- **≥ 2 GPUs**, Linux, CUDA. Trainer and vLLM server must run on different physical devices (TRL opens an NCCL communicator across both and NCCL refuses to bind two ranks to the same GPU).
- Developed on aarch64 Slurm HPC (NVHPC + H100/H200) and shared H200 nodes; also works on ordinary x86_64 + NVIDIA. macOS and Windows unsupported.
- AWS instances worth trying: `g6.12xlarge`, `g5.12xlarge`. T4-based (`g4dn`) did not work in testing.

---

## A. Run the examples

### Quick start (single node, ≥ 2 GPUs)

```bash
uv sync
```

At minimum you need 2 Nvidia gpus to test this (H100 or better) - bear in mind if you are sharing a node you may not be allocated devices 0 and 1!

Two terminals or `tmux` panes — vLLM and trainer talk over HTTP + NCCL:

```bash
# Terminal 1 — vLLM on GPU 0
CUDA_VISIBLE_DEVICES=0 uv run irl serve google/gemma-3-270m-it

# Terminal 2 — GRPO training on GPU 1
CUDA_VISIBLE_DEVICES=1 uv run irl train tldr
```

No need to restart vLLM between training runs — TRL re-syncs weights at the start of each. Only restart on model change.

### Examples

| Example | Task | Model | What it shows |
|---------|------|-------|---------------|
| [`tldr`](src/inspect_rl/example/tldr.py) | TL;DR summarisation | `gemma-3-270m-it` | Simplest setup — single-turn, one scorer |
| [`magic_number`](src/inspect_rl/example/magic_number.py) | Guess a fixed digit | `Qwen2.5-0.5B` | Multi-turn agent sanity test (`react` solver) |
| [`math_agent`](src/inspect_rl/example/math_agent.py) | Math with tools | `Qwen2.5-3B` | Multi-turn tool-calling, multiple scorers (canonical) |
| [`gsm8k`](src/inspect_rl/example/gsm8k.py) | GSM8K, XML output | `Qwen2.5-3B` (LoRA) | Single-turn structured output, PEFT |

Run any of them with `irl train <name>`. CLI reference: `irl serve --help`, `irl train <example> --help` (fast — doesn't load torch).

### Scaling up: more GPUs, multi-node, Slurm

For N trainer ranks on one node (N+1 GPUs total): `accelerate launch --num_processes N -m inspect_rl.cli train …` after starting vLLM on a separate GPU. For multi-node or Slurm, use the worked sbatch templates in [`examples/configs/`](examples/configs/) — `hpc1/` for 1–8 node Slingshot/CXI clusters, `hpc2/h200/` for a single multi-GPU H200 node with split jobs. That README has a sequence diagram, per-config table, and per-cluster caveats.

---

## B. Use in your own code (experimental)

### Install as a dependency

```bash
uv add --dev "inspect-rl @ git+ssh://git@github.com/alex-treebeard/inspect-rl.git"
# pin to a ref:
uv add --dev "inspect-rl @ git+ssh://git@github.com/alex-treebeard/inspect-rl.git@main"
```

Installs the `irl` console script and exposes `from inspect_rl import inspect_rl_train`.

### Define a task and train

Define an Inspect `Task` (dataset + solver + scorer) plus a TRL `GRPOConfig`, then call `inspect_rl_train(task=..., model=..., grpo_config=...)`. The scorer has full access to conversation history and tool calls via `TaskState`:

```python
from inspect_ai import Task
from inspect_ai.scorer import scorer, Score, Scorer, Target, accuracy
from inspect_ai.solver import TaskState, generate
from trl import GRPOConfig
from inspect_rl import inspect_rl_train

@scorer(metrics=[accuracy()])
def my_reward() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=1.0 if "correct" in state.output.completion else 0.0)
    return score

inspect_rl_train(
    task=Task(dataset=..., solver=[generate()], scorer=[my_reward()]),
    model="my-model",
    grpo_config=GRPOConfig(output_dir="outputs/my_task", use_vllm=True,
                           vllm_mode="server", max_steps=200, bf16=True),
)
```

A minimal downstream consumer (pyproject + script + `uv run example-tldr`) lives in [`examples/example_pkg/`](examples/example_pkg/). The built-in [`src/inspect_rl/example/`](src/inspect_rl/example/) tasks are also worth reading for solver/scorer patterns.

---

## Operating a run

- **Resume** — `irl train <example> --resume <run-dir>` restores optimizer/scheduler/RNG/step + wandb run; eval logs after the checkpoint are trimmed. vLLM state is **not** restored (restart `irl serve` first). Details: [`docs/01_features.md`](docs/01_features.md).
- **Heartbeat** — trainer emits `[heartbeat] step=N phase=… elapsed_in_phase=Xs` every 30 s; absent for >90 s means wedged (usually an `inspect_eval` parse-error retry loop or a stalled NCCL weight sync). Phase reference: [`docs/02_artifacts.md`](docs/02_artifacts.md).
- **Slurm HPC setup** — copy [`.env.example`](.env.example) → `.env`, fill in W&B / EKS / k8s vars, `set -a; source .env; set +a`, then `uv sync && just setup-slurm-hpc && uv run wandb login --host "$WANDB_BASE_URL"`. `$SCRATCH` and `CUDA_HOME` are auto-detected.
- **Logs** — `irl_output_slurm/latest/{master,vllm,trainer}.out` is the live triplet to tail; `train.log` is the post-hoc structured log.

## Further reading

- [`docs/01_features.md`](docs/01_features.md) — Resume, prefetcher, GRPO mods (OLMo 3 / DAPO), masking, `CUDA_HOME` auto-detect.
- [`docs/02_artifacts.md`](docs/02_artifacts.md) — Output directory layout, log file conventions.
- [`docs/03_internals.md`](docs/03_internals.md) — System architecture, `RolloutFunc`, why token IDs not text, custom vLLM provider.
- [`docs/04_known-issues.md`](docs/04_known-issues.md) — GPU sharing, stale NCCL, OOM, Gemma-3, …
- [`docs/05_roadmap.md`](docs/05_roadmap.md) — Bigger models, inflight weight updates, curriculum.
- [`journal/`](journal/) — Per-workstream notes; 1-min reads.
- Upstream: [Inspect](https://inspect.aisi.org.uk/) · [TRL](https://huggingface.co/docs/trl/index) · [vLLM](https://docs.vllm.ai/en/stable/)

## Dev

```bash
just lint    # ruff format + check
just test    # pytest
just smoke   # 2-GPU end-to-end (~3 min); `just smoke 2 3` if GPUs 0/1 are busy
```

### Coding agent / Claude Code guidance

- Docstrings short; prefer inline comments close to usage.
- Typehints, but `Any` is fine when stubs aren't worth it.
- `uv` for everything — `uv run` / `uv add`.
- Prefer github MCP over `gh` CLI when available.
- `just lint` before pushing; `just test` if you touched code.
- Journal entries in `./journal/` should be 1-min reads / 10-s skims.
- Use the `nb` MCP server for notebooks (auto-starts kernel from project venv). Prefer notebooks for interactive python (eval-log inspection, library debugging) over re-spawning python.
- Imports at top of file unless very optional / slow.
- If you're in a tmux pane, sibling panes are useful for vllm + debug training runs.
- **On shared Slurm clusters, never hard-code `CUDA_VISIBLE_DEVICES`** to an absolute index — some clusters don't cgroup-constrain device visibility, so `nvidia-smi` shows every GPU and guessing will OOM a neighbour. `irl serve/train --devices 0` already means "the first GPU Slurm gave us". See [`examples/configs/hpc2/h200/`](examples/configs/hpc2/h200/) for the split-job pattern.
- Don't leak infra specifics (node names, IPs, filesystem paths, partition/QoS) into the repo without checking. You can refer to clusters by hpc1, hpc2 etc. and public product names sparingly though

### Research workflow (RL / training)

- Skim recent `journal/` entries before starting.
- **Always eval on a fixed N-sample heldout set across steps.** Per-step rollouts sample from a rotating pool and can't separate learning from problem-difficulty noise. Before concluding anything from a curve, check `n`, `uniq_targets`, and whether the scorer read `tool_calls` or free completion text.
- **Scorers that fall back to regex over free text create reward shortcuts** — the policy satisfies them without using tools. Walk `state.messages[].tool_calls` instead.
- **Training stdout is a flood.** Per-step `inspect_eval` in `rollout.py` overwhelms the Jupyter IOPub / MCP client and hangs training. Use `display="none"` + `GRPOConfig(disable_tqdm=True)`. The compact `[step N]` line from `inspect_rl/util/display.py` is enough.
- **Monitor by polling the filesystem**, not by streaming stdout - can be helpful to give user a play-by-play on training runs as it's a lot of files to track
- **vLLM's NCCL communicator survives across trainer restarts.** Always `curl -sf -X POST http://localhost:8000/close_communicator/ -d '{}'` between runs; switching model needs killing vLLM entirely.
- **Bump model size before tuning rewards.** Qwen2.5-3B on GSM8K is the clean-curve threshold; 0.5B/1.5B are unstable without SFT warmup or curriculum.
- **KL regularisation matters.** `beta=0` (TRL default) drifts to noise; `beta=0.01` is weak for narrow per-step signals; `beta=0.05` is a reasonable floor when you can only afford 1–2 prompts per grad step.
