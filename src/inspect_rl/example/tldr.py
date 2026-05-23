"""TLDR summarization — simplest end-to-end example.

Reward: length penalty toward an ideal char count + bonus for first-person lead.
Easy to move the needle on, useful as a quick smoke test.

Usage:
    # Terminal 1 — start vLLM server
    irl serve google/gemma-3-270m-it

    # Terminal 2 — train
    irl train tldr
"""

from __future__ import annotations

from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate
from peft import LoraConfig, TaskType
from trl import GRPOConfig

from inspect_rl import inspect_rl_train

INSTRUCTION = "**You are a helpful assistant. Your task is to summarize the following text in a concise manner**"


@scorer(metrics=[accuracy()])
def tldr_reward(ideal: int = 100, minl: int = 0) -> Scorer:
    """Reward summaries near `ideal` chars long, bonus for starting with 'I'."""

    async def score(state: TaskState, target: Target) -> Score:
        comp = state.output.completion
        if len(comp) <= minl:
            return Score(value=-1000.0)
        length_penalty = abs(len(comp) - ideal)
        first_person_bonus = 100.0 if comp.startswith("I") else 0.0
        return Score(value=first_person_bonus - length_penalty)

    return score


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=[
            ChatMessageSystem(content=INSTRUCTION),
            ChatMessageUser(content=record["prompt"]),
        ],
        target=record["completion"],
    )


def get_task(split: str = "train") -> Task:
    return Task(
        dataset=hf_dataset(
            "trl-lib/tldr",
            split=split,
            sample_fields=record_to_sample,
            auto_id=True,
        ),
        solver=[generate()],
        scorer=[tldr_reward()],
    )


def train(
    model: str = "google/gemma-3-270m-it",
    vllm_base_url: str = "http://localhost:8000",
    output_dir: str = "irl_output",
    max_steps: int = 200,
    per_device_train_batch_size: int = 8,
    num_generations: int = 8,
    dataset_limit: int = 1000,
    max_completion_length: int = 512,
    learning_rate: float = 1e-5,
    eval_steps: int = 20,
    eval_limit: int = 32,
    resample_rounds: int = 3,
    off_policy_steps: int = 0,
    resume: str | None = None,
    wandb: bool = False,
    verbose: bool = False,
) -> None:
    import inspect_rl.util.display

    inspect_rl.util.display.verbose = verbose

    task = get_task(split="train")
    eval_task = get_task(split="test")

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        temperature=1.0,
        learning_rate=learning_rate,
        warmup_steps=10,
        bf16=True,
        fp16=False,
        report_to="wandb" if wandb else "none",
        save_steps=50,
        logging_steps=1,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host="localhost",
        vllm_server_port=8000,
        epsilon_high=0.28,
        reward_weights=[1.0],
    )

    peft_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=32,
        lora_dropout=0.0,
    )

    inspect_rl_train(
        task=task,
        model=model,
        grpo_config=grpo_config,
        vllm_base_url=vllm_base_url,
        peft_config=peft_config,
        dataset_limit=dataset_limit,
        eval_task=eval_task,
        eval_steps=eval_steps,
        eval_limit=eval_limit,
        resume_from=resume,
        max_resample_rounds=resample_rounds,
        off_policy_steps=off_policy_steps,
    )
