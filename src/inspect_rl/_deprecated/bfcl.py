# %%
"""WIP probs harder than gsm8k"""

import json
from typing import Any
from inspect_ai.model import ChatMessageUser, get_model
from inspect_evals.bfcl.bfcl import (
    bfcl_scorer,
    tool_call_to_string,
    DATASET_PATH,
    bfcl_solver,
    parse_target,
)


from inspect_ai import Task, dataset, eval, task

from inspect_rl.grpo import inspect_rl

from inspect_ai.model import ChatMessageAssistant
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import (
    Solver,
    TaskState,
)

# %%


@scorer([accuracy()])
def bfcl_reward_scorer() -> Scorer:
    # TODO shape the reward
    # Reward constants for different outcomes
    PERFECT_MATCH_REWARD = 1.0  # Function and arguments both correct
    CORRECT_FUNCTION_REWARD = 0.7  # Right function, wrong arguments
    CORRECT_ARGUMENTS_REWARD = 0.4  # Right arguments, wrong function
    WRONG_TOOL_CALL_COUNT_REWARD = 0.3  # Has tool calls but wrong number
    STRUCTURED_ATTEMPT_REWARD = 0.2  # Wrong function and arguments, but valid structure
    NO_TOOL_CALLS_REWARD = 0.2  # No tool calls found
    WRONG_MESSAGE_COUNT_REWARD = 0.1  # Wrong number of assistant messages
    NO_RESPONSE_REWARD = 0.0  # No assistant message at all

    async def score(state: TaskState, target: Target) -> Score:
        assistant_messages = [
            m for m in state.messages if isinstance(m, ChatMessageAssistant)
        ]

        # No assistant message - complete failure
        if len(assistant_messages) == 0:
            return Score(value=NO_RESPONSE_REWARD, answer="No assistant message")

        # Wrong number of messages - partial penalty
        elif len(assistant_messages) != 1:
            return Score(
                value=WRONG_MESSAGE_COUNT_REWARD,
                answer=f"Expected just 1 assistant message, got {len(assistant_messages)}",
            )

        message = assistant_messages[0]
        tool_calls = message.tool_calls

        # No tool calls or wrong number - partial penalty
        if tool_calls is None:
            return Score(value=NO_TOOL_CALLS_REWARD, answer="No tool calls found")
        elif len(tool_calls) != 1:
            return Score(
                value=WRONG_TOOL_CALL_COUNT_REWARD,
                answer=f"Expected 1 tool call, got {len(tool_calls)}",
            )

        target_obj = state.metadata["target_obj"]
        tool_call = tool_calls[0]

        args_identical = tool_call.arguments == target_obj["arguments"]
        function_identical = tool_call.function == target_obj["function"]

        print(
            f"args: {tool_call.arguments} == {target_obj['arguments']}\nfunction: {tool_call.function} == {target_obj['function']}"
        )

        # Calculate reward based on what's correct
        if function_identical and args_identical:
            # Perfect match
            reward = PERFECT_MATCH_REWARD
        elif function_identical:
            # Right function, wrong args - good partial credit
            reward = CORRECT_FUNCTION_REWARD
        elif args_identical:
            # Right args, wrong function - some partial credit
            reward = CORRECT_ARGUMENTS_REWARD
        else:
            # Both wrong but structured correctly - minimal credit
            reward = STRUCTURED_ATTEMPT_REWARD

        tool_call_string = tool_call_to_string(tool_call.function, tool_call.arguments)
        return Score(value=reward, answer=repr(tool_call_string))

    return score


# %%

SYSTEM_MESSAGE = """Your outputs must be in the format \\n[get_weather(city='San Francisco', metric='celsius')]\\n"""


def record_to_sample(record: dict[str, Any]) -> dataset.Sample:
    assert len(record["question"]) == 1
    assert len(record["ground_truth"]) == 1
    target = record["ground_truth"][0]

    parsed_target = parse_target(target)
    formatted_target = tool_call_to_string(
        parsed_target["function"], parsed_target["arguments"]
    )

    # the dataset contains tuples and lists, to simplify comparing these we convert them to lists by running them through json serialization
    jsoned_target = json.loads(json.dumps(parsed_target))
    question = record["question"][0][0]["content"] + "\n\n" + SYSTEM_MESSAGE
    input = [
        # ChatMessageSystem(content=SYSTEM_MESSAGE),
        ChatMessageUser(content=question),
    ]
    return dataset.Sample(
        input=input,
        target=formatted_target,
        metadata={"tools": record["function"], "target_obj": jsoned_target},
    )


@task
def bfcl_rl(solver: Solver | list[Solver] = bfcl_solver()) -> Task:
    ds = dataset.hf_dataset(
        DATASET_PATH,
        split="train",
        sample_fields=record_to_sample,
        # main branch does not load cleanly into an HF dataset so we use a PR branch which fixes it
        # see https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/discussions/15
        revision="1bf8bbc3c0e35d04d00339c223a3fd653aa195ac",
        name="exec_simple",
    )
    return Task(dataset=ds, solver=solver, scorer=bfcl_reward_scorer())


# %%

hf_model = "google/gemma-3-270m-it"


def _eval() -> None:
    bfcl_task = bfcl_rl()

    bfcl_task.scorer = [
        bfcl_scorer(),
        bfcl_reward_scorer(),
    ]
    # CUDA_VISIBLE_DEVICES=3 uv run vllm serve google/gemma-3-270m-it --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser pythonic --chat-template gemma3_pythonic
    model = get_model(
        f"openai/{hf_model}",
        base_url="http://localhost:8000/v1",
        api_key="local",
        # max_tokens=4096,
    )
    # model = "openai/gpt-4o-mini"
    eval(bfcl_task, limit=10, epochs=5, token_limit=1024, model=model)


def _train(wandb: bool = True) -> None:
    inspect_rl(
        hf_model=hf_model,
        dataset=task.dataset,
        scorers=task.scorer,
        limit=1000,
        wandb_enabled=wandb,
        output_dir="outputs/bfcl",
    )


# %%
