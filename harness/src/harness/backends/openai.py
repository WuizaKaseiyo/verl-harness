"""OpenAI-wire backend — covers OpenAI, OpenRouter, DeepSeek, Qwen, vLLM, and any
other endpoint that speaks OpenAI Chat Completions.

Uses the official `openai` SDK with `base_url` + `api_key` overridden per
provider profile. Streaming events are accumulated per tool_call so the loop
sees fully-parsed `tool_use` blocks, matching the Anthropic backend's contract.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import openai

from harness.backends.base import Backend, RawEvent
from harness.backends.translate import (
    finish_reason_to_anthropic,
    messages_anthropic_to_openai,
    tools_anthropic_to_openai,
)
from harness.providers import ResolvedModel


DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT_S = 600.0


class OpenAIBackend(Backend):
    def __init__(
        self,
        *,
        model_id: str,
        api_key: str | None,
        base_url: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.model_id = model_id
        # openai>=1 needs a non-empty api_key; local servers get "dummy".
        # SDK handles retries with respect for Retry-After headers and 5xx
        # transient failures.
        self._client = openai.AsyncOpenAI(
            api_key=api_key or "sk-no-key-needed",
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout_s,
        )

    @classmethod
    def from_resolved(cls, resolved: ResolvedModel) -> "OpenAIBackend":
        assert resolved.provider.wire == "openai", (
            f"OpenAIBackend needs wire=openai, got {resolved.provider.wire!r}"
        )
        return cls(
            model_id=resolved.model_id,
            api_key=resolved.api_key,
            base_url=resolved.provider.base_url,
        )

    async def stream(
        self,
        *,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> AsyncIterator[RawEvent]:
        oai_messages: list[dict[str, Any]] = []
        # Flatten Anthropic-style block list into a single string for the
        # openai wire (no equivalent cache-control today).
        system_str: str
        if isinstance(system, str):
            system_str = system
        else:
            system_str = "\n\n".join(
                block.get("text", "") for block in (system or [])
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if system_str:
            oai_messages.append({"role": "system", "content": system_str})
        oai_messages.extend(messages_anthropic_to_openai(messages))
        oai_tools = tools_anthropic_to_openai(tools) if tools else None

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": oai_messages,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if oai_tools:
            kwargs["tools"] = oai_tools

        # Accumulate streaming state
        tool_calls_state: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: dict[str, int] = {}

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            # Some backends emit a trailing chunk that only carries usage
            if getattr(chunk, "usage", None) is not None:
                usage = {
                    "input_tokens": getattr(chunk.usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(chunk.usage, "completion_tokens", 0) or 0,
                }
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            # Text delta
            content_delta = getattr(delta, "content", None)
            if content_delta:
                yield RawEvent(kind="text_delta", text=content_delta)

            # Tool-call streaming — accumulate by choice index
            tool_call_deltas = getattr(delta, "tool_calls", None) or []
            for tc in tool_call_deltas:
                idx = tc.index
                slot = tool_calls_state.setdefault(
                    idx, {"id": None, "name": None, "arguments": ""}
                )
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        # Emit accumulated tool_use blocks (in call order)
        for idx in sorted(tool_calls_state):
            slot = tool_calls_state[idx]
            try:
                parsed_input = json.loads(slot["arguments"]) if slot["arguments"] else {}
            except json.JSONDecodeError:
                parsed_input = {"__raw_arguments__": slot["arguments"]}
            yield RawEvent(
                kind="tool_use",
                tool_use={
                    "id": slot["id"] or f"call_{idx}",
                    "name": slot["name"] or "",
                    "input": parsed_input,
                },
            )

        yield RawEvent(
            kind="message_stop",
            stop_reason=finish_reason_to_anthropic(finish_reason),
            usage=usage,
        )
