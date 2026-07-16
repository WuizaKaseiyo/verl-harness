"""Backend abstract base.

Backends stream a `RawEvent` sequence per turn. The tool loop consumes these
events, executes any `tool_use` blocks, appends `tool_result` messages, and
calls back into `Backend.stream(...)` for the next turn.

Internal message + tool-schema shape is **Anthropic-native**. The OpenAI
backend (M3) translates on the wire; every other layer (loop, state driver,
event emitter) assumes Anthropic shape.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RawEvent:
    """One backend event.

    `kind` is exhaustive:
      - "text_delta"  — text.text is a partial assistant text chunk
      - "tool_use"    — the model wants us to call a tool; tool_use = {id, name, input(dict)}
      - "message_stop"— the assistant turn is done; stop_reason + usage populated

    `usage` on message_stop may include:
      - input_tokens         — full input tokens billed
      - output_tokens        — output tokens billed
      - cache_read_input_tokens     — Anthropic: tokens served from cache (10% price)
      - cache_creation_input_tokens — Anthropic: tokens WRITTEN to cache (125% price)
    """

    kind: str
    text: str = ""
    tool_use: dict[str, Any] = field(default_factory=dict)
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class Backend(abc.ABC):
    """One backend instance is bound to one (provider, model_id) pair."""

    model_id: str

    @abc.abstractmethod
    async def stream(
        self,
        *,
        system: str | list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> AsyncIterator[RawEvent]:
        """Stream one assistant turn.

        Args:
          system:   system prompt. Either a plain string OR a list of Anthropic
                    content blocks (each `{"type": "text", "text": "..."}`).
                    Block form allows the anthropic backend to apply
                    `cache_control` for prompt caching; the openai backend flattens.
          messages: Anthropic-native messages list. Each message is
                    `{"role": "user"|"assistant", "content": str | list[block]}`.
                    Blocks are `{"type": "text", "text": ...}`,
                    `{"type": "tool_use", "id": ..., "name": ..., "input": {...}}`,
                    or `{"type": "tool_result", "tool_use_id": ..., "content": ...}`.
          tools:    Anthropic tool schemas: [{"name", "description", "input_schema"}].
          max_tokens: cap on assistant output tokens.

        Yields RawEvent in order. Terminates with exactly one `message_stop`.
        """
        raise NotImplementedError
        # pragma: no cover — abstract
        if False:  # pyright: unreachable
            yield  # type: ignore[unreachable]
