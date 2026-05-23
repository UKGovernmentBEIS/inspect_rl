"""GSM8K math reasoning — LoRA example with structured output.

Reward: XML format compliance + answer correctness. Trains with LoRA
on Llama-3.1-8B-Instruct to produce <reasoning>...</reasoning><answer>...</answer> blocks.

Usage:
    # Terminal 1 — start vLLM server
    irl serve meta-llama/Llama-3.1-8B-Instruct

    # Terminal 2 — train
    irl train gsm8k
"""

from __future__ import annotations

import re
from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate
from peft import LoraConfig, TaskType
from trl import GRPOConfig

from inspect_rl import inspect_rl_train

SYSTEM_PROMPT = """You are given a math problem.
Think about the problem and provide your working out.
Place it between <reasoning> and </reasoning>.
Then, provide your final answer between <answer> and </answer>.

Example:
What is 1 + 1 + 1?
<reasoning>
1 + 1 = 2, 2 + 1 = 3
</reasoning>
<answer>
3
</answer>"""


@scorer(metrics=[accuracy()])
def xmlcount_scorer() -> Scorer:
    """Reward correct XML formatting."""

    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion
        value = 0.0
        if text.count("<reasoning>\n") == 1:
            value += 0.25
        if text.count("\n</reasoning>\n") == 1:
            value += 0.25
        if text.count("\n<answer>\n") == 1:
            value += 0.25
            trailing = text.split("</answer>")[-1] if "</answer>" in text else ""
            value -= len(trailing.strip()) * 0.001
        if text.count("\n</answer>") == 1:
            value += 0.25
        return Score(value=max(value, 0.0))

    return score


@scorer(metrics=[accuracy()])
def correctness_scorer() -> Scorer:
    """Reward correct final answers."""

    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match and match.group(1).strip() == target.text.strip():
            return Score(value=4.0, explanation="Correct")
        return Score(value=0.0, explanation="Wrong")

    return score


def record_to_sample(record: dict[str, Any]) -> Sample:
    delim = "####"
    answer = record["answer"].split(delim)
    target = answer.pop().strip()
    return Sample(
        input=[
            ChatMessageSystem(content=SYSTEM_PROMPT),
            ChatMessageUser(content=record["question"]),
        ],
        target=target,
    )


def get_task(split: str = "train") -> Task:
    return Task(
        dataset=hf_dataset(
            "openai/gsm8k",
            data_dir="main",
            split=split,
            sample_fields=record_to_sample,
            auto_id=True,
        ),
        solver=[generate()],
        scorer=[xmlcount_scorer(), correctness_scorer()],
    )


def train(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    vllm_base_url: str = "http://localhost:8000",
    output_dir: str = "irl_output",
    max_steps: int = 200,
    per_device_train_batch_size: int = 16,
    num_generations: int = 8,
    dataset_limit: int = 5000,
    eval_steps: int = 5,
    eval_limit: int = 50,
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
        max_completion_length=4096,
        temperature=1.0,
        learning_rate=1e-5,
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
        reward_weights=[2.0, 3.0],
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
