# %%
from typing import Any
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_rl.grpo import inspect_rl


# %%
@scorer(metrics=[accuracy()])
def tldr_reward(ideal: int = 100, minl: int = 0) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        comp = state.output.completion
        if len(state.output.completion) <= minl:
            return Score(value=-1000)

        length_penalty = abs(len(state.output.completion) - ideal)
        first_person_bonus = 100 if comp.startswith("I") else 0
        return Score(value=first_person_bonus - length_penalty)

    return score


INSTRUCTION = "**You are a helpful assistant. Your task is to summarize the following text in a concise manner**"


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=[
            ChatMessageSystem(content=INSTRUCTION),
            ChatMessageUser(content=record["prompt"]),
        ],
        target=record["completion"],
    )


task = Task(
    scorer=[tldr_reward()],
    dataset=hf_dataset(
        "trl-lib/tldr",
        split="train",
        sample_fields=record_to_sample,
        auto_id=True,
    ),
)

# %% eval


def _eval(
    hf_model: str = "google/gemma-3-270m-it", local_model: str | None = None
) -> None:
    if local_model:
        model = get_model(
            "hf/local",
            model_path=local_model,
        )
        print("Using local model:", local_model)
    else:
        model = get_model(
            f"hf/{hf_model}",
        )
        print("Using HF model:", hf_model)
    eval(task, limit=10, epochs=5, token_limit=1024, model=model)


# %% train


def _train(wandb: bool = True, hf_model: str = "google/gemma-3-270m-it") -> None:
    assert task.scorer is not None

    inspect_rl(
        hf_model=hf_model,
        dataset=task.dataset,
        scorers=task.scorer,
        limit=1000,
        wandb_enabled=wandb,
        output_dir="outputs/tldr",
    )


# %%
