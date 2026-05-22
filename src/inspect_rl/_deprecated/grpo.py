from typing import Any
from inspect_ai.dataset import Dataset
from datasets import Dataset as HFDataset
from trl import GRPOConfig, GRPOTrainer

import asyncio
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ModelName,
    ModelOutput,
    get_model,
)
from inspect_ai.solver import TaskState

from inspect_ai.scorer import Scorer, Target

from transformers import AutoModelForCausalLM
from trl.trainer.grpo_trainer import RewardFunc
import os
import wandb
from rich.console import Console
from rich.table import Table


def inspect_rl(
    hf_model: str,
    dataset: Dataset,
    scorers: list[Scorer],
    limit: int = 10,
    wandb_enabled: bool = True,
    output_dir: str = "outputs",
):
    dlist = []
    for sample in dataset[-limit:]:
        if isinstance(sample.input, str):
            prompt = {
                "role": "user",
                "content": sample.input,
            }
        else:
            prompt = [
                {
                    "role": s.role,
                    "content": s.content,
                }
                for s in sample.input
            ]
        dlist.append(
            {
                "prompt": prompt,
                "answer": sample.target,
            }
        )

    hf_dataset = HFDataset.from_list(dlist)

    _scorers = [scorer_to_reward_func(s, wandb_enabled) for s in scorers]

    grpo(
        hf_model,
        hf_dataset,
        _scorers,
        wandb_enabled=wandb_enabled,
        output_dir=output_dir,
    )


def grpo(
    hf_model: str,
    dataset: HFDataset,
    scorers: list[RewardFunc],
    wandb_enabled: bool = True,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    output_dir: str = "grpo_outputs",
):
    # Initialize wandb if project name is provided
    if wandb_enabled:
        import wandb

        if wandb_project is None:
            if "JUPYTERHUB_USER" in os.environ:
                wandb_project = f"inspect-rl-{os.environ['JUPYTERHUB_USER']}"
            elif "USER" in os.environ:
                user = os.environ["USER"].replace(".", "-").lower()
                wandb_project = f"inspect-rl-{user}"
            else:
                raise ValueError(
                    "wandb_project must be provided if wandb_enabled is True and not running in a known environment."
                )

        if wandb_entity is None:
            wandb_entity = "research-unit"

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config={
                "model": hf_model,
            },
        )

    # following the huggingface docs
    training_args = GRPOConfig(
        output_dir=output_dir,
        report_to="wandb",
        save_steps=50,
        max_steps=200,
        bf16=True,
        fp16=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=scorers,
        args=training_args,
        train_dataset=dataset,
    )
    train_result = trainer.train()

    if wandb_enabled and training_args.report_to == "wandb":
        metrics = train_result.metrics
        wandb.log(metrics)
        wandb.finish()


def scorer_to_reward_func(scorer: Scorer, wandb_enabled: bool) -> RewardFunc:
    def reward_func(
        prompts: list[Any], completions: list[Any], answer: list[str], *args, **kwargs
    ) -> list[float]:
        wandb_table = wandb.Table(
            columns=["prompt", "target response", "model response", "score"]
        )
        scores = []

        step = kwargs["trainer_state"].global_step

        table = Table(title=f"Responses and Scores for step {step}")
        table.add_column("prompt", justify="left", style="cyan")
        table.add_column("target response", justify="right", style="green")
        table.add_column("model response", justify="left", style="magenta")
        table.add_column("score", justify="right", style="yellow")

        for index, (prompt, completion, _answer) in enumerate(
            zip(prompts, completions, answer)
        ):
            response = completion[0]["content"]
            target = Target(_answer)

            state = TaskState(
                model=ModelName(get_model("mockllm/model")),
                sample_id=0,
                epoch=0,
                input=prompt,
                messages=[],
                target=target,
                choices=None,
                output=ModelOutput(
                    choices=[
                        ChatCompletionChoice(
                            message=ChatMessageAssistant(
                                content=response,
                            )
                        )
                    ]
                ),
                message_limit=None,
                token_limit=None,
                completed=False,
                metadata={},
            )
            score = asyncio.run(scorer(state, target))
            scores.append(score.value)

            if index < 1:
                try:
                    table.add_row(
                        str(prompt[-1]["content"][:500]),
                        str(_answer),
                        str(response[:500]),
                        str(score.value),
                    )
                    wandb_table.add_data(
                        str(prompt[-1]["content"][:500]),
                        str(_answer),
                        str(response[:500]),
                        score.value,
                    )
                except Exception as e:
                    print(f"Error adding row to table: {e}")
                    # Fallback to string representation

        console = Console()
        if step % 10 == 0:
            console.print(table)
            if wandb_enabled:
                wandb.log({f"scores_step_{step}": wandb_table})
                print(
                    f'See table wandb by querying for table runs.summary["scores_step_{step}"]'
                )
        return scores

    return reward_func  # type: ignore
