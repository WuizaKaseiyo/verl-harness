"""Tool-use loop.

Runs one Backend turn at a time:

  backend.stream(system, messages, tools)
    → accumulate text + tool_use blocks
    → emit assistant event
    → if control tool called   → halt, return LoopResult(reason="control")
    → if stop_reason=end_turn  → halt, return LoopResult(reason="end_turn")
    → if stop_reason=tool_use  → execute each tool, emit user tool_result, loop
    → if stop_reason=max_tokens → halt with reason="max_tokens"
    → if max_iterations reached → halt with reason="max_iterations"

The `control_tools` set is how the state driver (T6) intercepts special tools
like `transition_to(next_state)` — the loop never calls the registry for them;
it just captures the input and returns.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from harness.backends.base import Backend
from harness.events import EventEmitter, text_block, tool_result_block, tool_use_block
from harness.tools import ToolContext, ToolRegistry


@dataclass
class LoopResult:
    """Outcome of one `run_loop(...)` invocation."""

    reason: str  # "end_turn" | "control" | "max_iterations" | "max_tokens"
    turns: int
    final_text: str
    messages: list[dict[str, Any]]
    usage: dict[str, int] = field(default_factory=dict)
    control_call: dict[str, Any] | None = None  # {"id", "name", "input"}


async def run_loop(
    *,
    backend: Backend,
    system: str | list[dict[str, Any]],
    initial_messages: list[dict[str, Any]],
    tools: ToolRegistry,
    ctx: ToolContext,
    emitter: EventEmitter,
    max_iterations: int = 100,
    max_tokens: int = 4096,
    control_tools: set[str] | None = None,
    control_schemas: list[dict[str, Any]] | None = None,
) -> LoopResult:
    """Drive a tool-use conversation until it halts.

    `control_tools`: names the loop should NOT execute — instead, return
    `LoopResult(reason="control", control_call={...})` as soon as the model
    calls one. Their JSON schemas must be passed via `control_schemas` so the
    backend advertises them to the model.
    """
    control_tools = control_tools or set()
    control_schemas = control_schemas or []
    combined_schemas = tools.schemas() + control_schemas

    messages = [dict(m) for m in initial_messages]  # local copy
    total_usage = {"input_tokens": 0, "output_tokens": 0}
    last_text = ""

    for turn_idx in range(max_iterations):
        text_parts: list[str] = []
        tool_use_blocks: list[dict[str, Any]] = []
        stop_reason: str | None = None
        turn_usage: dict[str, int] = {}

        async for event in backend.stream(
            system=system,
            messages=messages,
            tools=combined_schemas,
            max_tokens=max_tokens,
        ):
            if event.kind == "text_delta":
                text_parts.append(event.text)
            elif event.kind == "tool_use":
                tool_use_blocks.append(
                    tool_use_block(
                        tool_id=event.tool_use["id"],
                        name=event.tool_use["name"],
                        input=event.tool_use["input"],
                    )
                )
            elif event.kind == "message_stop":
                stop_reason = event.stop_reason
                turn_usage = event.usage

        # Emit + record the assistant turn.
        joined = "".join(text_parts)
        content: list[dict[str, Any]] = []
        if joined:
            content.append(text_block(joined))
        content.extend(tool_use_blocks)

        emitter.assistant(
            content=content,
            stop_reason=stop_reason,
            usage=turn_usage,
        )
        total_usage["input_tokens"] += turn_usage.get("input_tokens", 0)
        total_usage["output_tokens"] += turn_usage.get("output_tokens", 0)
        last_text = joined

        messages.append({"role": "assistant", "content": content})

        # Control-tool short-circuit: first matching control call wins.
        for tu in tool_use_blocks:
            if tu["name"] in control_tools:
                return LoopResult(
                    reason="control",
                    turns=turn_idx + 1,
                    final_text=joined,
                    messages=messages,
                    usage=total_usage,
                    control_call={
                        "id": tu["id"],
                        "name": tu["name"],
                        "input": tu["input"],
                    },
                )

        # Terminal stop reasons. Explicit match — `max_tokens` must not be
        # subsumed by the `end_turn` catch-all, or truncated turns get mislabeled.
        if stop_reason == "max_tokens":
            return LoopResult(
                reason="max_tokens",
                turns=turn_idx + 1,
                final_text=joined,
                messages=messages,
                usage=total_usage,
            )
        if not tool_use_blocks:
            # Whatever the reason (end_turn, stop_sequence, refusal, or a nil
            # stop) — no tools to run, we're done.
            return LoopResult(
                reason="end_turn",
                turns=turn_idx + 1,
                final_text=joined,
                messages=messages,
                usage=total_usage,
            )

        # Otherwise: execute the tool_use blocks and feed results back.
        results: list[dict[str, Any]] = []
        for tu in tool_use_blocks:
            content_str, is_error = await asyncio.to_thread(
                tools.execute, tu["name"], tu["input"], ctx
            )
            results.append(
                tool_result_block(
                    tool_use_id=tu["id"],
                    content=content_str,
                    is_error=is_error,
                )
            )
        emitter.user_tool_result(results)
        messages.append({"role": "user", "content": results})

    return LoopResult(
        reason="max_iterations",
        turns=max_iterations,
        final_text=last_text,
        messages=messages,
        usage=total_usage,
    )
