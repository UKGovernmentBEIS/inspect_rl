"""Magic number — minimal multi-turn agent sanity test.

Every sample asks the same question: "what's the magic number?" The answer is
always `MAGIC_NUMBER` (a fixed single digit). The agent has only a `submit`
tool and can try up to `attempts` times — wrong submissions come back with
"that's not it, try again". With a uniform random policy it's right ~1/10 of
the time per attempt; if training is working at all, the policy should
collapse onto the fixed digit within a handful of steps.

Used to debug the multi-turn agent training path. If this doesn't learn, the
problem is in the pipeline, not in the task difficulty.

Uses the Inspect agent API (`react`) rather than the solver API (`basic_agent`).

Usage:
    irl serve Qwen/Qwen2.5-0.5B-Instruct
    irl train magic-number
"""

from __future__ import annotations

from inspect_ai import Task
from inspect_ai.agent import AgentAttempts, AgentSubmit, react
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState
from trl import GRPOConfig

from inspect_rl import inspect_rl_train

MAGIC_NUMBER = 2

INSTRUCTIONS = (
    "There is a secret magic number between 0 and 9 (inclusive). "
    "Call the submit tool with a single digit to guess it."
)


@scorer(metrics=[accuracy()])
def magic_correctness() -> Scorer:
    """1.0 if the submitted value equals MAGIC_NUMBER, else 0.0.

    Also serves as the react agent's "did I get it right?" signal — react
    uses the first scorer's float value to decide whether to stop early or
    play back the incorrect_message for another attempt. Keep this scorer
    first in the Task's scorer list.
    """

    async def score(state: TaskState, target: Target) -> Score:
        submission = state.output.completion.strip()
        try:
            if int(submission) == int(target.text):
                return Score(value=1.0, explanation="Correct")
        except ValueError:
            pass
        return Score(value=0.0, explanation=f"Wrong: {submission!r}")

    return score


@scorer(metrics=[accuracy()])
def valid_submit() -> Scorer:
    """1.0 if the agent made a well-formed submit call (single digit 0-9), else 0.0.

    Stricter than "any submit tool response exists": we inspect the assistant's
    tool_call arguments and require a parseable 0-9 digit. The looser "any
    submit" version turned into a reward-hacking attractor — once the policy
    learned that *any* submit call clears the floor, it happily submitted
    garbage and training started amplifying whatever verbose JSON-ish text
    happened to correlate with successful rollouts.

    Keep this scorer lightweight compared to correctness (small reward_weight).
    It's a bootstrap for "learn the tool-calling interface", not a thing to
    plateau on.
    """

    async def score(state: TaskState, target: Target) -> Score:
        for msg in state.messages:
            if not isinstance(msg, ChatMessageAssistant) or not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                if tc.function != "submit":
                    continue
                answer = (tc.arguments or {}).get("answer", "")
                try:
                    val = int(str(answer).strip())
                except (ValueError, TypeError):
                    continue
                if 0 <= val <= 9:
                    return Score(value=1.0, explanation=f"Valid submit: {val}")
        return Score(value=0.0, explanation="No valid single-digit submit")

    return score


@scorer(metrics=[accuracy()])
def tool_call_failures() -> Scorer:
    """1.0 if ANY assistant message had a parse error or batched tool calls.

    Parse errors happen when Inspect can't decode the model's tool-call syntax
    (bad JSON, malformed Hermes XML, hallucinated tool name). Batched calls
    happen when the model emits multiple <tool_call> blocks in one message —
    for this task we want exactly one submit per turn.

    Weighted *negatively* in reward_weights so this acts as a penalty. The
    `accuracy` metric on this scorer reports the fraction of rollouts that
    had any failure — useful as a wandb signal for reward-hacking drift.
    """

    async def score(state: TaskState, target: Target) -> Score:
        problems: list[str] = []
        for msg in state.messages:
            if not isinstance(msg, ChatMessageAssistant) or not msg.tool_calls:
                continue
            if len(msg.tool_calls) > 1:
                problems.append(f"batched ({len(msg.tool_calls)} calls)")
            for tc in msg.tool_calls:
                if tc.parse_error:
                    problems.append(f"parse_error: {tc.parse_error[:80]}")
                elif tc.function != "submit":
                    # only tool registered is submit; anything else is hallucinated
                    problems.append(f"unknown tool: {tc.function}")
        if problems:
            return Score(value=1.0, explanation="; ".join(problems[:3]))
        return Score(value=0.0, explanation="clean")

    return score


def _build_dataset(n: int) -> MemoryDataset:
    # Identical samples — we want the policy to memorize one digit, not generalize.
    return MemoryDataset(
        samples=[
            Sample(
                id=str(i),
                input=[ChatMessageUser(content="What is the magic number?")],
                target=str(MAGIC_NUMBER),
            )
            for i in range(n)
        ]
    )


def get_task(n: int = 256) -> Task:
    agent = react(
        prompt=INSTRUCTIONS,
        # answer_only=True: completion is just the last submitted answer, not
        # accumulated with delimiters (so `int(completion)` parses cleanly and
        # AgentAttempts can detect a correct retry).
        # keep_in_messages=True: don't let react strip submit tool_calls from
        # the final state. Without this, our valid_submit / tool_call_failures
        # scorers walk `state.messages` and find nothing — react synthesizes
        # a text-only assistant message in place of the tool call, defeating
        # both the bootstrap credit and the failure penalty.
        submit=AgentSubmit(answer_only=True, keep_in_messages=True),
        attempts=AgentAttempts(
            attempts=3,
            incorrect_message="That is not the magic number. Try a different digit.",
        ),
    )
    return Task(
        dataset=_build_dataset(n),
        solver=agent,
        # correctness first — react's AgentAttempts uses the first scorer to
        # decide early-stop. valid_submit is a small bootstrap credit for
        # emitting a well-formed submit call; tool_call_failures is weighted
        # negatively in reward_weights to penalize bad tool syntax / batched
        # calls.
        scorer=[magic_correctness(), valid_submit(), tool_call_failures()],
        message_limit=20,
    )


def train(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    vllm_base_url: str = "http://localhost:8000",
    output_dir: str = "irl_output",
    max_steps: int = 100,
    per_device_train_batch_size: int = 8,
    num_generations: int = 8,
    dataset_limit: int = 256,
    eval_steps: int = 10,
    eval_limit: int = 16,
    resample_rounds: int = 3,
    off_policy_steps: int = 0,
    resume: str | None = None,
    wandb: bool = True,
    verbose: bool = False,
) -> None:
    import inspect_rl.util.display

    inspect_rl.util.display.verbose = verbose

    task = get_task(n=dataset_limit)
    # Held-out set is a separate batch of the same (degenerate) prompt — at
    # eval temperature=0.0 it cleanly measures whether the policy has
    # collapsed onto MAGIC_NUMBER, distinct from the high-temperature
    # rollouts.
    eval_task = get_task(n=eval_limit)

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        max_completion_length=256,
        temperature=1.0,
        learning_rate=5e-6,
        warmup_steps=20,
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
        reward_weights=[2.5, 0.1, -0.3],
    )

    inspect_rl_train(
        task=task,
        model=model,
        grpo_config=grpo_config,
        vllm_base_url=vllm_base_url,
        dataset_limit=dataset_limit,
        eval_task=eval_task,
        eval_steps=eval_steps,
        eval_limit=eval_limit,
        resume_from=resume,
        max_resample_rounds=resample_rounds,
        off_policy_steps=off_policy_steps,
    )
