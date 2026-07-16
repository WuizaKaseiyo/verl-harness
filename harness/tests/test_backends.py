"""Backend integration tests.

The Anthropic backend test hits the real API — gated on ANTHROPIC_API_KEY.
Skipped in CI without a key set.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from harness.backends.anthropic import AnthropicBackend
from harness.backends.base import RawEvent
from harness.providers import resolve_model_spec


_HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
_MODEL = os.environ.get("HARNESS_TEST_MODEL", "claude-haiku-4-5-20251001")


@pytest.mark.skipif(not _HAS_KEY, reason="ANTHROPIC_API_KEY not set")
def test_anthropic_hello() -> None:
    resolved = resolve_model_spec(f"anthropic/{_MODEL}")
    backend = AnthropicBackend.from_resolved(resolved)

    async def _run() -> list[RawEvent]:
        events: list[RawEvent] = []
        async for e in backend.stream(
            system="You reply in exactly one word.",
            messages=[{"role": "user", "content": "Say hi."}],
            tools=[],
            max_tokens=32,
        ):
            events.append(e)
        return events

    events = asyncio.run(_run())

    # Sanity: at least one text chunk, exactly one terminal message_stop.
    assert any(e.kind == "text_delta" for e in events)
    assert events[-1].kind == "message_stop"
    assert events[-1].stop_reason in {"end_turn", "max_tokens", "stop_sequence"}
    assert events[-1].usage.get("output_tokens", 0) > 0

    text = "".join(e.text for e in events if e.kind == "text_delta")
    assert text.strip()  # got something


@pytest.mark.skipif(not _HAS_KEY, reason="ANTHROPIC_API_KEY not set")
def test_anthropic_tool_use() -> None:
    """Model asked to call a fake tool — should emit a tool_use event."""
    resolved = resolve_model_spec(f"anthropic/{_MODEL}")
    backend = AnthropicBackend.from_resolved(resolved)

    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a location.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        }
    ]

    async def _run() -> list[RawEvent]:
        events: list[RawEvent] = []
        async for e in backend.stream(
            system="You must call the get_weather tool.",
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in Paris? Use the tool.",
                }
            ],
            tools=tools,
            max_tokens=256,
        ):
            events.append(e)
        return events

    events = asyncio.run(_run())

    tool_events = [e for e in events if e.kind == "tool_use"]
    assert len(tool_events) == 1
    assert tool_events[0].tool_use["name"] == "get_weather"
    assert "location" in tool_events[0].tool_use["input"]
    assert events[-1].stop_reason == "tool_use"
