"""Multi-turn math agent — tool-use example.

Agent solves GSM8K word problems using a calculator tool and submits a single
numeric answer. Reward shape pushes toward (a) correct final answer, (b) using
the calculator for arithmetic instead of mental math.

Requires a model with tool-calling support in its chat template.

Usage:
    # Terminal 1 — start vLLM server
    irl serve Qwen/Qwen2.5-3B-Instruct

    # Terminal 2 — train
    irl train math-agent
"""

from __future__ import annotations

import re
from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer
from inspect_ai.solver import TaskState, basic_agent, system_message
from inspect_ai.tool import tool
from trl import GRPOConfig

from inspect_rl import inspect_rl_train

# One-shot worked example embedded in the system prompt. Qwen-Instruct
# pattern-matches this pretty reliably — a full demo via message-history
# faked turns is possible but brittle across chat templates.
SYSTEM_PROMPT = """You are a math assistant. Solve word problems step by step.

Tools:
- calculator(expression): evaluates a Python arithmetic expression (e.g. "3 * (4 + 5)"). Use it for every arithmetic step instead of computing in your head.
- submit(answer): submits your final numeric answer as a string.

Keep reasoning short. Call calculator once per arithmetic step. Then submit a single number.

Example:
Problem: A store has 14 eggs. The chicken laid 6 more, then 3 were sold. How many remain?
Step 1: new total → calculator(expression="14 + 6") returns 20
Step 2: after sales → calculator(expression="20 - 3") returns 17
Final: submit(answer="17")

Now solve the user's problem the same way."""


@tool
def calculator():
    async def execute(expression: str) -> str:
        """Evaluate a mathematical expression.

        Args:
            expression: A Python math expression, e.g. "3 * (4 + 5)" or "120 / 8".

        Returns:
            The result as a string.
        """
        try:
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "Error: expression contains invalid characters"
            result = eval(expression)  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    return execute


@scorer(metrics=[accuracy()])
def valid_submit() -> Scorer:
    """1.0 if the agent made a well-formed submit call (parseable numeric arg)."""

    async def score(state: TaskState, target: Target) -> Score:
        for msg in state.messages:
            if not isinstance(msg, ChatMessageAssistant) or not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                if tc.function != "submit":
                    continue
                answer = (tc.arguments or {}).get("answer", "")
                try:
                    float(str(answer).strip())
                except (ValueError, TypeError):
                    continue
                return Score(value=1.0, explanation=f"Valid submit: {answer!r}")
        return Score(value=0.0, explanation="No valid numeric submit")

    return score


@scorer(metrics=[accuracy()])
def uses_calculator() -> Scorer:
    """1.0 if at least one calculator tool_call fired successfully (no parse_error).

    Incentive aligned with goal (b): use calculator instead of doing arithmetic
    in prose. Without this, valid_submit alone lets the policy satisfy the
    bootstrap credit by submitting the answer straight off without tool use,
    which on multi-step problems produces wrong arithmetic.
    """

    async def score(state: TaskState, target: Target) -> Score:
        for msg in state.messages:
            if not isinstance(msg, ChatMessageAssistant) or not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                if tc.function == "calculator" and not tc.parse_error:
                    return Score(value=1.0, explanation="used calculator")
        return Score(value=0.0, explanation="no calculator use")

    return score


@scorer(metrics=[accuracy()])
def tool_call_failures() -> Scorer:
    """1.0 if any assistant tool_call had a parse error or called an unknown tool."""

    async def score(state: TaskState, target: Target) -> Score:
        allowed = {"calculator", "submit"}
        problems: list[str] = []
        for msg in state.messages:
            if not isinstance(msg, ChatMessageAssistant) or not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                if tc.parse_error:
                    problems.append(f"parse_error: {tc.parse_error[:80]}")
                elif tc.function not in allowed:
                    problems.append(f"unknown tool: {tc.function}")
        if problems:
            return Score(value=1.0, explanation="; ".join(problems[:3]))
        return Score(value=0.0, explanation="clean")

    return score


@scorer(metrics=[accuracy()])
def correctness() -> Scorer:
    """1.0 if the last submit tool_call's answer parses to the target numeric."""

    async def score(state: TaskState, target: Target) -> Score:
        last_answer: str | None = None
        for msg in state.messages:
            if not isinstance(msg, ChatMessageAssistant) or not msg.tool_calls:
                continue
            for tc in msg.tool_calls:
                if tc.function == "submit":
                    last_answer = str((tc.arguments or {}).get("answer", "")).strip()

        if last_answer is None:
            return Score(value=0.0, explanation="No submit")

        cleaned = re.sub(r"[^\d.\-]", "", last_answer)
        try:
            if float(cleaned) == float(target.text.strip()):
                return Score(value=1.0, explanation="Correct")
        except ValueError:
            pass
        return Score(value=0.0, explanation=f"Wrong: got {last_answer!r}")

    return score


def record_to_sample(record: dict[str, Any]) -> Sample:
    delim = "####"
    answer = record["answer"].split(delim)
    target = answer.pop().strip()
    return Sample(
        input=[ChatMessageUser(content=record["question"])],
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
        solver=basic_agent(
            init=system_message(SYSTEM_PROMPT),
            tools=[calculator()],
            max_attempts=1,
            message_limit=10,
        ),
        # Order matters: basic_agent uses the first scorer's float value for its
        # early-stop check. Reward weights in train() must match this order.
        scorer=[correctness(), uses_calculator(), valid_submit(), tool_call_failures()],
    )


def train(
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    vllm_base_url: str = "http://localhost:8000",
    output_dir: str = "irl_output",
    max_steps: int = 150,
    per_device_train_batch_size: int = 8,
    num_generations: int = 4,
    dataset_limit: int = 1000,
    eval_steps: int = 10,
    eval_limit: int = 32,
    learning_rate: float = 5e-6,
    beta: float = 0.05,
    save_steps: int = 10,
    resample_rounds: int = 3,
    off_policy_steps: int = 0,
    resume: str | None = None,
    wandb: bool = False,
    verbose: bool = False,
) -> None:
    """Train the math agent.

    Defaults tuned for Qwen2.5-3B on a single GH200. bs=8 num_gen=4 gives
    2 prompts per gradient step. gradient_checkpointing trades ~2x step time
    for ~4x activation-memory savings — needed for 3B at bs=8 seq=1024.
    beta=0.05 prevents the low-entropy mode collapse seen on 0.5B at beta=0.01.
    """
    import inspect_rl.util.display

    inspect_rl.util.display.verbose = verbose

    task = get_task(split="train")
    eval_task = get_task(split="test")

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        max_completion_length=1024,
        temperature=1.0,
        learning_rate=learning_rate,
        warmup_steps=10,
        bf16=True,
        fp16=False,
        report_to="wandb" if wandb else "none",
        save_steps=save_steps,
        save_total_limit=1,
        logging_steps=5,
        disable_tqdm=True,
        gradient_checkpointing=True,
        # FSDP1 + reentrant gradient checkpointing trips
        # "Non-root FSDP instance's `_is_root` should not have been set yet" on
        # the second forward, because the reentrant path runs each checkpointed
        # segment as if it were the FSDP root. Non-reentrant checkpointing
        # routes through autograd without that double-root path.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host="localhost",
        vllm_server_port=8000,
        # TRL's `create_model_from_path` defaults `device_map="auto"` (auto-shards
        # the model across visible GPUs in a single process — breaks FSDP) and
        # `dtype="float32"` (28 GB for a 7B model). We override both:
        #   - device_map=None: model loads on CPU; FSDP/DS places it on the rank's GPU.
        #   - dtype=bfloat16:  policy + ref both load at half size. With bf16 training
        #     enabled (bf16=True above) the fp32 master weights are reconstructed by
        #     the optimizer (or kept in bf16 under DS ZeRO with bf16 enabled).
        model_init_kwargs={"device_map": None, "dtype": "bfloat16"},
        beta=beta,
        # --- OLMo 3 GRPO modifications ---
        # Asymmetric clipping: higher upper bound allows larger positive updates
        # while still constraining negative ones. OLMo 3 / DAPO style.
        epsilon_high=0.28,
        # Token-level loss is already the default (loss_type="dapo").
        reward_weights=[4.0, 0.5, 0.1, -0.5],
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
