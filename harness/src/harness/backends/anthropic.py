"""Anthropic-native backend — talks to api.anthropic.com (or any endpoint that
speaks Anthropic's Messages API)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anthropic

from harness.backends.base import Backend, RawEvent
from harness.providers import ResolvedModel


DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT_S = 600.0  # 10 min — accommodates 30-40k output tokens


class AnthropicBackend(Backend):
    def __init__(
        self,
        *,
        model_id: str,
        api_key: str,
        base_url: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.model_id = model_id
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            # SDK handles retries with respect for retry-after headers and
            # 5xx transient errors. Max 5 attempts covers rate-limit windows.
            "max_retries": max_retries,
            "timeout": timeout_s,
        }
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)

    @classmethod
    def from_resolved(cls, resolved: ResolvedModel) -> "AnthropicBackend":
        assert resolved.provider.wire == "anthropic", (
            f"AnthropicBackend needs wire=anthropic, got {resolved.provider.wire!r}"
        )
        assert resolved.api_key, "anthropic wire requires an api key"
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
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            # Apply cache_control to the first up-to-4 blocks so consecutive
            # calls in a run reuse them. Anthropic caps cache_control at 4.
            kwargs["system"] = _prepare_system_with_cache(system)
        if tools:
            kwargs["tools"] = tools

        async with self._client.messages.stream(**kwargs) as stream:
            async for text_chunk in stream.text_stream:
                if text_chunk:
                    yield RawEvent(kind="text_delta", text=text_chunk)

            final = await stream.get_final_message()

        for block in final.content:
            if getattr(block, "type", None) == "tool_use":
                yield RawEvent(
                    kind="tool_use",
                    tool_use={
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    },
                )

        usage: dict[str, int] = {}
        if final.usage is not None:
            usage = {
                "input_tokens": final.usage.input_tokens,
                "output_tokens": final.usage.output_tokens,
            }
            # Cache stats — populated when `cache_control` is set
            cr = getattr(final.usage, "cache_read_input_tokens", None)
            if cr is not None:
                usage["cache_read_input_tokens"] = cr
            cw = getattr(final.usage, "cache_creation_input_tokens", None)
            if cw is not None:
                usage["cache_creation_input_tokens"] = cw
        yield RawEvent(
            kind="message_stop",
            stop_reason=final.stop_reason,
            usage=usage,
        )


def _prepare_system_with_cache(
    system: str | list[dict[str, Any]],
) -> list[dict[str, Any]] | str:
    """Convert system to the wire form with cache_control on the first 4 blocks.

    - str → returned as-is (no caching)
    - list of blocks → deep-copied; up to 4 blocks get `cache_control:ephemeral`
    """
    if isinstance(system, str):
        return system
    out: list[dict[str, Any]] = []
    caches_left = 4
    for block in system:
        # copy so we don't mutate the caller's list
        b = dict(block)
        if caches_left > 0 and b.get("type") == "text" and "cache_control" not in b:
            b["cache_control"] = {"type": "ephemeral"}
            caches_left -= 1
        out.append(b)
    return out
