"""Minimal TL;DR training run, written as a downstream consumer would write it.

Mirrors `inspect_rl.example.tldr` — fresh Task, scorer, GRPOConfig — but
builds everything from the public `inspect_rl` / `inspect_ai` / `trl`
surfaces without reaching into the examples package. If this script runs,
the package works as a dependency.

**Requires ≥ 2 GPUs.** TRL's vLLM server + trainer initialise a shared
NCCL communicator for weight sync, and NCCL refuses to bind two distinct
ranks to the same physical device — so even though gemma-3-270m fits on
one card memory-wise, you need a separate device for the trainer. This
script pins the trainer to device 1; start `irl serve` on device 0.

Prereq — run in a separate shell first:

```bash
CUDA_VISIBLE_DEVICES=0 irl serve google/gemma-3-270m-it --enforce-eager
```

`--enforce-eager` skips vLLM's CUDA graph capture and shaves ~30-60s
off startup. Leave it off for real training, but for a 5-step smoke run
the per-token speed hit is irrelevant.

Then run this script:

```bash
uv run example-tldr
```
"""

from __future__ import annotations

import os

# Pin the trainer to device 1 BEFORE any torch / inspect_rl import — the
# inspect_rl bootstrap then runs against this process env. Explicit `=`
# (not setdefault) because ambient shells often pre-set
# CUDA_VISIBLE_DEVICES to something broader; this script wants device 1
# specifically so `irl serve` can own device 0 (TRL's NCCL communicator
# rejects colocation on the same physical GPU).
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# ---------------------------------------------------------------------------
# Scorer + dataset definitions at module level.
#
# inspect_rl.trainer._extract_scorer_names uses __qualname__ split on
# ".<locals>" to derive reward-func names — a scorer nested inside main()
# would collapse to "main" and the reward lookup would miss, yielding zero
# rewards. Module-level defs sidestep this.
# ---------------------------------------------------------------------------

from inspect_ai.dataset import Sample  # noqa: E402
from inspect_ai.model import ChatMessageSystem, ChatMessageUser  # noqa: E402
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer  # noqa: E402
from inspect_ai.solver import TaskState  # noqa: E402

INSTRUCTION = (
    "**You are a helpful assistant. Your task is to summarize the "
    "following text in a concise manner**"
)


@scorer(metrics=[accuracy()])
def tldr_reward(ideal: int = 100) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        comp = state.output.completion
        if not comp:
            return Score(value=-1000.0)
        length_penalty = abs(len(comp) - ideal)
        first_person_bonus = 100.0 if comp.startswith("I") else 0.0
        return Score(value=first_person_bonus - length_penalty)

    return score


def _record_to_sample(record: dict) -> Sample:
    return Sample(
        input=[
            ChatMessageSystem(content=INSTRUCTION),
            ChatMessageUser(content=record["prompt"]),
        ],
        target=record["completion"],
    )


def main() -> None:
    # Heavy imports live inside main() so `python -c 'import example_pkg'`
    # stays light — torch only loads when the script actually runs.
    from inspect_ai import Task
    from inspect_ai.dataset import hf_dataset
    from inspect_ai.solver import generate
    from trl import GRPOConfig

    from inspect_rl import inspect_rl_train

    task = Task(
        dataset=hf_dataset(
            "trl-lib/tldr",
            split="train",
            sample_fields=_record_to_sample,
            auto_id=True,
        ),
        solver=[generate()],
        scorer=[tldr_reward()],
    )

    grpo_config = GRPOConfig(
        output_dir="outputs/example_pkg_tldr",
        max_steps=5,
        per_device_train_batch_size=4,
        num_generations=4,
        max_completion_length=256,
        temperature=1.0,
        learning_rate=1e-5,
        warmup_steps=2,
        bf16=True,
        fp16=False,
        report_to="none",
        save_steps=1000,  # skip checkpointing for this smoke run
        logging_steps=1,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host="localhost",
        vllm_server_port=8000,
        reward_weights=[1.0],
    )

    print(
        f"[example-tldr] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"CUDA_HOME={os.environ.get('CUDA_HOME')}"
    )

    inspect_rl_train(
        task=task,
        model="google/gemma-3-270m-it",
        grpo_config=grpo_config,
        dataset_limit=32,
    )
