"""Custom Inspect ModelAPI that talks to TRL's vLLM /generate/ endpoint.

Uses client-side tokenization so the exact token IDs the policy sampled
survive across multi-turn rollouts (server-side /chat/ re-tokenization
is lossy for BPE — suboptimal splits canonicalize into different tokens).

Captures per-turn token IDs and logprobs on each assistant message's
`metadata["trl_turn"]`, so the rollout can aggregate them across a
multi-turn trajectory. Also applies `tools` to the chat template
client-side and parses Hermes-XML tool calls out of the response via
Inspect's HFHandler.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Thread
from typing import Any

import httpx
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.model._providers.util.hf_handler import HFHandler
from inspect_ai.tool import ToolChoice, ToolInfo
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class _QueueItem:
    messages: list[dict[str, Any]]
    tools: list[ToolInfo]
    prompt_ids: list[int]
    config: GenerateConfig
    future: asyncio.Future[ModelOutput]
    loop: asyncio.AbstractEventLoop
    tool_sig: tuple = field(default_factory=tuple)


@modelapi(name="trl-vllm")
def trl_vllm() -> type[ModelAPI]:
    return TRLVLLMProvider


class TRLVLLMProvider(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str,
        tokenizer: PreTrainedTokenizerBase,
        config: GenerateConfig = GenerateConfig(),
        batch_timeout: float = 0.25,
        max_batch_size: int = 512,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            config=config,
        )
        self.tokenizer = tokenizer
        self.batch_timeout = batch_timeout
        self.max_batch_size = max_batch_size
        self._queue: Queue[_QueueItem] = Queue()
        self._batch_thread: Thread | None = None
        self._shutdown = False
        self._client = httpx.Client(timeout=600.0)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # Inspect's ToolChoice is Literal["auto","any","none"] | ToolFunction —
        # never None. "none" is the default for toolless tasks (e.g. tldr's
        # plain generate()); "auto" is what basic_agent passes. "any" and
        # explicit ToolFunction targets we don't plumb through to vLLM.
        if tool_choice not in ("auto", "none"):
            logger.warning(
                "TRLVLLMProvider ignoring tool_choice=%r; only 'auto' and 'none' are honored.",
                tool_choice,
            )
        self._ensure_batch_thread()
        messages = self._messages_to_dicts(input)
        prompt_ids = self._build_prompt_ids(input, messages, tools)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ModelOutput] = loop.create_future()
        self._queue.put(
            _QueueItem(
                messages=messages,
                tools=list(tools),
                prompt_ids=prompt_ids,
                config=config,
                future=future,
                loop=loop,
                tool_sig=_tool_signature(tools),
            )
        )
        return await future

    async def aclose(self) -> None:
        self._shutdown = True
        if self._batch_thread is not None:
            self._batch_thread.join(timeout=5.0)
        self._client.close()

    # -- batching machinery --

    def _ensure_batch_thread(self) -> None:
        if self._batch_thread is None or not self._batch_thread.is_alive():
            self._shutdown = False
            self._batch_thread = Thread(target=self._process_batches, daemon=True)
            self._batch_thread.start()

    def _process_batches(self) -> None:
        while not self._shutdown:
            try:
                first = self._queue.get(timeout=1.0)
            except Empty:
                continue
            items = [first]
            deadline = time.monotonic() + self.batch_timeout
            while len(items) < self.max_batch_size:
                remaining = max(0, deadline - time.monotonic())
                if remaining <= 0:
                    break
                try:
                    items.append(self._queue.get(timeout=remaining))
                except Empty:
                    break
            try:
                self._execute_batch(items)
            except Exception as exc:
                for item in items:
                    if not item.future.done():
                        item.loop.call_soon_threadsafe(item.future.set_exception, exc)

    def _execute_batch(self, items: list[_QueueItem]) -> None:
        config = items[0].config
        tools = items[0].tools
        # batch must be tool-homogeneous — same task in a GRPO step → same tools.
        # if this ever fires we'd need to split the batch by tool signature.
        first_sig = items[0].tool_sig
        for item in items[1:]:
            if item.tool_sig != first_sig:
                raise RuntimeError(
                    "TRLVLLMProvider batch has heterogeneous tools; "
                    "split batches by tool signature before batching."
                )
        # Use /generate/ with pre-tokenized prompts. This is critical for multi-turn:
        # /chat/ re-tokenizes messages server-side, and BPE re-encoding of the model's
        # raw output is not bijective (suboptimal splits canonicalize into different
        # tokens), which breaks the prompt_{n+1} ⊇ prompt_n + completion_n invariant.
        # Client-side token assembly preserves the exact completion tokens across turns.
        payload: dict[str, Any] = {
            "prompts": [item.prompt_ids for item in items],
            "n": 1,
            "max_tokens": config.max_tokens or 4096,
            "temperature": config.temperature
            if config.temperature is not None
            else 1.0,
            "top_p": config.top_p if config.top_p is not None else 1.0,
            "top_k": config.top_k if config.top_k is not None else -1,
            "logprobs": 0,
        }

        resp = self._client.post(f"{self.base_url}/generate/", json=payload)
        resp.raise_for_status()
        data = resp.json()

        prompt_ids_list = data["prompt_ids"]
        completion_ids_list = data["completion_ids"]
        logprobs_list = data["logprobs"]

        handler = HFHandler(self.model_name) if tools else None

        for i, item in enumerate(items):
            comp_ids = completion_ids_list[i]
            text = self.tokenizer.decode(comp_ids, skip_special_tokens=True)
            # logprobs from TRL server: shape (seq_len, num_logprobs) — take the sampled token.
            per_token_logprobs = logprobs_list[i]
            flat_logprobs = [
                lps[0] if isinstance(lps, list) else lps for lps in per_token_logprobs
            ]

            if handler is not None:
                message = handler.parse_assistant_response(text, item.tools)
            else:
                from inspect_ai.model import ChatMessageAssistant

                message = ChatMessageAssistant(content=text)

            # stash per-turn token data on the message itself so it rides along
            # with basic_agent's state.messages into the eval log for rollout aggregation.
            # `raw_text` is the decoded completion with special tokens stripped; we feed it
            # back verbatim to the server on the next turn so the chat template re-renders
            # identical tokens (avoiding structural tool_calls re-serialization drift).
            message.metadata = {
                "trl_turn": {
                    "prompt_ids": prompt_ids_list[i],
                    "completion_ids": comp_ids,
                    "logprobs": flat_logprobs,
                    "raw_text": text,
                }
            }

            output = ModelOutput(
                model=self.model_name,
                choices=[
                    ChatCompletionChoice(message=message, stop_reason="stop"),
                ],
                usage=ModelUsage(
                    input_tokens=len(prompt_ids_list[i]),
                    output_tokens=len(comp_ids),
                    total_tokens=len(prompt_ids_list[i]) + len(comp_ids),
                ),
                # kept for non-agent single-turn rollouts that read sample.output.metadata.
                metadata={
                    "trl_completion_data": {
                        "prompt_ids": prompt_ids_list[i],
                        "completion_ids": comp_ids,
                        "logprobs": flat_logprobs,
                    }
                },
            )
            item.loop.call_soon_threadsafe(item.future.set_result, output)

    def _build_prompt_ids(
        self,
        input: list[ChatMessage],
        messages: list[dict[str, Any]],
        tools: list[ToolInfo],
    ) -> list[int]:
        """Assemble the next turn's prompt_ids client-side.

        For the first turn we just apply the chat template to the whole
        conversation. For subsequent turns, we splice the last assistant's
        raw `completion_ids` (preserving the exact tokens the policy
        sampled) with a freshly-tokenized environment delta — the tool
        response + next generation prompt — computed by diffing
        rendered-template strings.
        """
        tool_schemas = _to_openai_tools(tools) if tools else None

        # find the most recent assistant message we generated (has trl_turn metadata).
        last_idx = None
        for idx in range(len(input) - 1, -1, -1):
            m = input[idx]
            if m.role == "assistant" and m.metadata and "trl_turn" in m.metadata:
                last_idx = idx
                break

        if last_idx is None:
            # first turn — tokenize the full conversation, let template add tools.
            return self.tokenizer.apply_chat_template(
                messages,
                tools=tool_schemas,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )

        trl_turn = input[last_idx].metadata["trl_turn"]
        prev_prompt_ids: list[int] = list(trl_turn["prompt_ids"])
        prev_completion_ids: list[int] = list(trl_turn["completion_ids"])

        # render the conversation up-to-and-including the last assistant turn, and
        # the full current conversation with a generation prompt. Splice at the
        # assistant's closing eos_token: prev_completion_ids ends exactly at that
        # token (vLLM includes the stop token), so everything after it in the
        # canonical rendering — template separators, tool response, next-turn
        # header — is the environment delta we need to tokenize fresh.
        before_dicts = messages[: last_idx + 1]
        before_text = self.tokenizer.apply_chat_template(
            before_dicts,
            tools=tool_schemas,
            add_generation_prompt=False,
            tokenize=False,
        )
        after_text = self.tokenizer.apply_chat_template(
            messages,
            tools=tool_schemas,
            add_generation_prompt=True,
            tokenize=False,
        )
        eos = self.tokenizer.eos_token
        if eos is None:
            raise RuntimeError("tokenizer has no eos_token; cannot locate splice")
        splice_end = before_text.rindex(eos) + len(eos)
        pre_split = before_text[:splice_end]
        if not after_text.startswith(pre_split):
            raise RuntimeError(
                "chat template rendering is not prefix-stable across turns; "
                "cannot compute environment delta safely."
            )
        delta_text = after_text[len(pre_split) :]
        delta_ids = self.tokenizer.encode(delta_text, add_special_tokens=False)

        return prev_prompt_ids + prev_completion_ids + list(delta_ids)

    @staticmethod
    def _messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        result = []
        for msg in messages:
            d: dict[str, Any] = {"role": msg.role, "content": msg.text or ""}
            if msg.role == "assistant":
                # If we generated this turn, feed back the raw text verbatim — not
                # the structured tool_calls. Re-rendering tool_calls through the
                # chat template produces a canonicalized form that diverges from
                # the model's raw output (whitespace, key order), breaking the
                # prompt_{n+1} ⊇ prompt_n + completion_n invariant the rollout
                # aggregator relies on.
                trl_turn = (
                    (msg.metadata or {}).get("trl_turn") if msg.metadata else None
                )
                if trl_turn and "raw_text" in trl_turn:
                    d["content"] = trl_turn["raw_text"]
                else:
                    tool_calls = getattr(msg, "tool_calls", None)
                    if tool_calls:
                        d["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function,
                                    "arguments": tc.arguments,
                                },
                            }
                            for tc in tool_calls
                        ]
            elif msg.role == "tool":
                tool_call_id = getattr(msg, "tool_call_id", None)
                if tool_call_id:
                    d["tool_call_id"] = tool_call_id
                fn = getattr(msg, "function", None)
                if fn:
                    d["name"] = fn
            result.append(d)
        return result


def _to_openai_tools(tools: list[ToolInfo]) -> list[dict[str, Any]]:
    """Convert Inspect ToolInfos into OpenAI-function-style schemas.

    TRL's vllm-serve forwards `tools` directly to tokenizer.apply_chat_template,
    which Qwen 2.5's built-in template accepts in this nested form.
    """
    return [
        {"type": "function", "function": info.model_dump(exclude_none=True)}
        for info in tools
    ]


def _tool_signature(tools: list[ToolInfo]) -> tuple:
    """Canonical tuple for equality-checking two tool sets within a batch."""
    return tuple(
        (
            t.name,
            t.description,
            json.dumps(t.parameters.model_dump(exclude_none=True), sort_keys=True),
        )
        for t in tools
    )
