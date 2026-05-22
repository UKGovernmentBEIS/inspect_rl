# %%
from typing import Any
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState
from inspect_rl.grpo import inspect_rl
import re

# %%

# Special tags for reasoning and solution
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

# Regex to match reasoning and solution sections
match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

# Regex to extract numerical answers from solution text
match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)


@scorer(metrics=[accuracy()])
def match_format_exactly_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        # Compare state / model output with target
        # to yield a score
        if match_format.search(state.output.completion) is not None:
            return Score(value=3.0, explanation="Output format is correct.")
        return Score(value=0.0, explanation="Output format is incorrect.")

    return score


@scorer(metrics=[accuracy()])
def match_format_approximately_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion
        score_value = 0
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score_value += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score_value += 0.5 if response.count(reasoning_end) == 1 else -0.5
        score_value += 0.5 if response.count(solution_start) == 1 else -0.5
        score_value += 0.5 if response.count(solution_end) == 1 else -0.5
        return Score(value=score_value)

    return score


@scorer(metrics=[accuracy()])
def check_answer_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion
        guess = match_format.search(response)
        if guess is None:
            return Score(value=0.0, explanation="No answer found in response.")

        guess = guess.group(1)
        score_value = 0
        # Correct answer gets 3 points!
        if guess == target.text:
            score_value += 3.0
            explanation = "Exact answer match."
        # Match if spaces are seen
        elif guess.strip() == float(target.text):
            score_value += 1.5
            explanation = "Answer matches with spaces."
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(float(target.text))
                if ratio >= 0.9 and ratio <= 1.1:
                    score_value += 0.5
                    explanation = "Answer is within 10% of the correct answer."
                elif ratio >= 0.8 and ratio <= 1.2:
                    score_value += 0.25
                    explanation = "Answer is within 20% of the correct answer."
                else:
                    score_value -= 1.0  # Penalize wrong answers
                    explanation = "Answer is too far from the correct answer."
            except (ValueError, ZeroDivisionError):
                score_value -= 0.5  # Penalize
                explanation = "Error in parsing answer."
        return Score(value=score_value, explanation=explanation)

    return score


@scorer(metrics=[accuracy()])
def check_numbers_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion
        guess = match_numbers.search(response)
        if guess is None:
            return Score(value=0.0, explanation="No number found in response.")

        guess = guess.group(1)
        # Convert to numbers
        try:
            guess_float = float(guess.strip())
            return Score(
                value=1.5 if guess_float == float(target.text) else 0.0,
                explanation="Number matches the target.",
            )
        except ValueError:
            return Score(value=0.0, explanation="Error in parsing number.")

    return score


system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}.

e.g. What is 1 + 1 + 1?

Your working out is: {reasoning_start}1 + 1 = 2, 2 + 1 = 3{reasoning_end}
Your final answer is: {solution_start}3{solution_end}
"""


def record_to_sample(record: dict[str, Any]) -> Sample:
    DELIM = "####"
    input = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": record["question"],
        },
    ]
    answer = record["answer"].split(DELIM)
    target = answer.pop().strip()
    reasoning = DELIM.join(answer)
    return Sample(input=input, target=target, metadata={"reasoning": reasoning.strip()})


task = Task(
    scorer=[
        match_format_exactly_scorer(),
        match_format_approximately_scorer(),
        check_answer_scorer(),
        check_numbers_scorer(),
    ],
    dataset=hf_dataset(
        "openai/gsm8k",
        data_dir="main",
        split="train",
        sample_fields=record_to_sample,
        auto_id=True,
    ),
)

hf_model = "unsloth/gemma-3-1b-it"
# hf_model = "google/gemma-3-270m-it"

# %% eval


def _eval() -> None:
    model = get_model(
        f"openai/{hf_model}",
        base_url="http://localhost:8000/v1",
        api_key="local",
        # max_tokens=4096,
    )
    eval(task, limit=1, epochs=20, token_limit=1024, model=model)


# %% train


def _train(wandb: bool = True) -> None:
    inspect_rl(
        hf_model=hf_model,
        dataset=task.dataset,
        scorers=task.scorer,
        limit=5000,
        wandb_enabled=wandb,
        output_dir="outputs/gsm8k",
    )


# %%
