"""Tool-use loop tests using a scripted fake backend.

Live Anthropic integration is exercised in test_state_driver (T6) and the
E2E smoke (T7). Here we cover control flow: end_turn, tool_use round-trip,
parallel tool_use, control-tool short-circuit, max_iterations, error tool.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from harness.backends.base import Backend, RawEvent
from harness.events import EventEmitter
from harness.loop import run_loop
from harness.tools import ToolContext, default_registry


class ScriptedBackend(Backend):
    """Emits pre-scripted events, one script entry per turn."""

    model_id = "scripted-model"

    def __init__(self, script: list[list[RawEvent]]) -> None:
        self._script = script
        self.turn = 0
        self.received_messages: list[list[dict[str, Any]]] = []

    async def stream(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> AsyncIterator[RawEvent]:
        self.received_messages.append(messages)
        if self.turn >= len(self._script):
            raise AssertionError(f"backend script exhausted at turn {self.turn}")
        events = self._script[self.turn]
        self.turn += 1
        for e in events:
            yield e


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ToolContext(cwd=tmp_path, workdir=tmp_path, run_id="T", workspace=ws)


def _emitter() -> tuple[EventEmitter, io.StringIO]:
    sink = io.StringIO()
    return EventEmitter(session_id="s", model="m", cwd="/", sink=sink), sink


# ── happy path ──────────────────────────────────────────────────────────────

def test_single_turn_end_turn(ctx: ToolContext) -> None:
    backend = ScriptedBackend(
        [
            [
                RawEvent(kind="text_delta", text="Hello"),
                RawEvent(kind="text_delta", text=" there"),
                RawEvent(
                    kind="message_stop",
                    stop_reason="end_turn",
                    usage={"input_tokens": 10, "output_tokens": 3},
                ),
            ]
        ]
    )
    emitter, _sink = _emitter()
    result = asyncio.run(
        run_loop(
            backend=backend,
            system="",
            initial_messages=[{"role": "user", "content": "hi"}],
            tools=default_registry(),
            ctx=ctx,
            emitter=emitter,
        )
    )
    assert result.reason == "end_turn"
    assert result.turns == 1
    assert result.final_text == "Hello there"
    assert result.usage["output_tokens"] == 3
    assert len(emitter.history) == 1  # one assistant event, no user event


def test_tool_use_roundtrip(ctx: ToolContext, tmp_path: Path) -> None:
    """turn 1: bash `pwd`  →  turn 2: end_turn with answer text."""
    backend = ScriptedBackend(
        [
            [
                RawEvent(
                    kind="tool_use",
                    tool_use={
                        "id": "t1",
                        "name": "bash",
                        "input": {"command": "pwd"},
                    },
                ),
                RawEvent(kind="message_stop", stop_reason="tool_use"),
            ],
            [
                RawEvent(kind="text_delta", text=f"cwd is {tmp_path}"),
                RawEvent(kind="message_stop", stop_reason="end_turn"),
            ],
        ]
    )
    emitter, _sink = _emitter()
    result = asyncio.run(
        run_loop(
            backend=backend,
            system="",
            initial_messages=[{"role": "user", "content": "cwd?"}],
            tools=default_registry(),
            ctx=ctx,
            emitter=emitter,
        )
    )
    assert result.reason == "end_turn"
    assert result.turns == 2
    assert str(tmp_path) in result.final_text

    # Event trace: assistant(tool_use) → user(tool_result) → assistant(text)
    kinds = [e["type"] for e in emitter.history]
    assert kinds == ["assistant", "user", "assistant"]

    tool_result = emitter.history[1]["message"]["content"][0]
    assert tool_result["tool_use_id"] == "t1"
    assert tool_result["is_error"] is False
    assert str(tmp_path) in tool_result["content"]


def test_parallel_tool_use(ctx: ToolContext, tmp_path: Path) -> None:
    """Two tool_use blocks in one turn — both execute; single user reply carries both results."""
    backend = ScriptedBackend(
        [
            [
                RawEvent(
                    kind="tool_use",
                    tool_use={
                        "id": "t1",
                        "name": "bash",
                        "input": {"command": "printf a"},
                    },
                ),
                RawEvent(
                    kind="tool_use",
                    tool_use={
                        "id": "t2",
                        "name": "bash",
                        "input": {"command": "printf b"},
                    },
                ),
                RawEvent(kind="message_stop", stop_reason="tool_use"),
            ],
            [
                RawEvent(kind="text_delta", text="got both"),
                RawEvent(kind="message_stop", stop_reason="end_turn"),
            ],
        ]
    )
    emitter, _sink = _emitter()
    result = asyncio.run(
        run_loop(
            backend=backend,
            system="",
            initial_messages=[{"role": "user", "content": "run two"}],
            tools=default_registry(),
            ctx=ctx,
            emitter=emitter,
        )
    )
    assert result.reason == "end_turn"

    # user turn should carry BOTH results
    user_event = emitter.history[1]
    assert user_event["type"] == "user"
    results = user_event["message"]["content"]
    assert len(results) == 2
    assert {r["tool_use_id"] for r in results} == {"t1", "t2"}
    assert "a" in results[0]["content"]
    assert "b" in results[1]["content"]


# ── control-tool short-circuit ──────────────────────────────────────────────

def test_control_tool_halts(ctx: ToolContext) -> None:
    """Model calls `transition_to` → loop returns reason=control without executing it."""
    backend = ScriptedBackend(
        [
            [
                RawEvent(kind="text_delta", text="ready to transition."),
                RawEvent(
                    kind="tool_use",
                    tool_use={
                        "id": "c1",
                        "name": "transition_to",
                        "input": {"next_state": "locate_recipe"},
                    },
                ),
                RawEvent(kind="message_stop", stop_reason="tool_use"),
            ],
        ]
    )
    emitter, _sink = _emitter()
    result = asyncio.run(
        run_loop(
            backend=backend,
            system="",
            initial_messages=[{"role": "user", "content": "go"}],
            tools=default_registry(),
            ctx=ctx,
            emitter=emitter,
            control_tools={"transition_to"},
            control_schemas=[
                {
                    "name": "transition_to",
                    "description": "signal readiness for next FSM state",
                    "input_schema": {
                        "type": "object",
                        "properties": {"next_state": {"type": "string"}},
                        "required": ["next_state"],
                    },
                }
            ],
        )
    )
    assert result.reason == "control"
    assert result.control_call is not None
    assert result.control_call["name"] == "transition_to"
    assert result.control_call["input"]["next_state"] == "locate_recipe"

    # No `user` tool_result event should have been emitted — the loop halted first.
    kinds = [e["type"] for e in emitter.history]
    assert kinds == ["assistant"]

    # Control tool schema was advertised to the backend
    assert backend.turn == 1


# ── failure paths ──────────────────────────────────────────────────────────

def test_max_iterations(ctx: ToolContext) -> None:
    """Model calls bash forever — loop stops at max_iterations."""
    forever_script = [
        [
            RawEvent(
                kind="tool_use",
                tool_use={
                    "id": f"t{i}",
                    "name": "bash",
                    "input": {"command": "true"},
                },
            ),
            RawEvent(kind="message_stop", stop_reason="tool_use"),
        ]
        for i in range(10)
    ]
    backend = ScriptedBackend(forever_script)
    emitter, _sink = _emitter()
    result = asyncio.run(
        run_loop(
            backend=backend,
            system="",
            initial_messages=[{"role": "user", "content": "loop"}],
            tools=default_registry(),
            ctx=ctx,
            emitter=emitter,
            max_iterations=3,
        )
    )
    assert result.reason == "max_iterations"
    assert result.turns == 3


def test_unknown_tool_marked_error(ctx: ToolContext) -> None:
    """Model calls a nonexistent tool → tool_result has is_error=True; loop continues."""
    backend = ScriptedBackend(
        [
            [
                RawEvent(
                    kind="tool_use",
                    tool_use={
                        "id": "t1",
                        "name": "doesnt_exist",
                        "input": {},
                    },
                ),
                RawEvent(kind="message_stop", stop_reason="tool_use"),
            ],
            [
                RawEvent(kind="text_delta", text="sorry"),
                RawEvent(kind="message_stop", stop_reason="end_turn"),
            ],
        ]
    )
    emitter, _sink = _emitter()
    result = asyncio.run(
        run_loop(
            backend=backend,
            system="",
            initial_messages=[{"role": "user", "content": "misfire"}],
            tools=default_registry(),
            ctx=ctx,
            emitter=emitter,
        )
    )
    assert result.reason == "end_turn"
    user_event = emitter.history[1]
    tr = user_event["message"]["content"][0]
    assert tr["is_error"] is True
    assert "unknown tool" in tr["content"]


def test_max_tokens_halts(ctx: ToolContext) -> None:
    backend = ScriptedBackend(
        [
            [
                RawEvent(kind="text_delta", text="partial"),
                RawEvent(kind="message_stop", stop_reason="max_tokens"),
            ]
        ]
    )
    emitter, _sink = _emitter()
    result = asyncio.run(
        run_loop(
            backend=backend,
            system="",
            initial_messages=[{"role": "user", "content": "..."}],
            tools=default_registry(),
            ctx=ctx,
            emitter=emitter,
        )
    )
    assert result.reason == "max_tokens"
    assert result.final_text == "partial"


# ── ls-roundtrip: the T5 exit-criteria test ────────────────────────────────

def test_ls_roundtrip_exit_criteria(ctx: ToolContext, tmp_path: Path) -> None:
    """The plan's exit-criteria case — scripted, no live model.

    "run ls and describe" → loop calls bash → returns final assistant text.
    Assert on the 5 event kinds (system:init not in scope of run_loop; the state
    driver in T6 emits it).
    """
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "b.txt").write_text("")

    backend = ScriptedBackend(
        [
            [
                RawEvent(kind="text_delta", text="running ls."),
                RawEvent(
                    kind="tool_use",
                    tool_use={
                        "id": "t1",
                        "name": "bash",
                        "input": {"command": "ls"},
                    },
                ),
                RawEvent(kind="message_stop", stop_reason="tool_use"),
            ],
            [
                RawEvent(kind="text_delta", text="I see a.txt and b.txt."),
                RawEvent(kind="message_stop", stop_reason="end_turn"),
            ],
        ]
    )
    emitter, _sink = _emitter()
    result = asyncio.run(
        run_loop(
            backend=backend,
            system="",
            initial_messages=[{"role": "user", "content": "run ls and describe"}],
            tools=default_registry(),
            ctx=ctx,
            emitter=emitter,
        )
    )
    assert result.reason == "end_turn"
    assert result.turns == 2
    assert "a.txt" in result.final_text or "b.txt" in result.final_text
    # 3 emitter events: assistant(with tool_use) → user(tool_result) → assistant(text)
    assert [e["type"] for e in emitter.history] == ["assistant", "user", "assistant"]
