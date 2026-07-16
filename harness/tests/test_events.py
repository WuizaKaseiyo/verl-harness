"""Verify the stream-json event format matches what the existing dashboard reads.

Reference format lives in web/src/verl_harness_web/submit.py — this test asserts
against the exact fields that parser reads:
  system  → e["type"] == "system" and e["subtype"] == "init"
  assist  → e["type"] == "assistant" and e["message"]["content"] contains text + tool_use blocks
  user    → e["type"] == "user"     and content carries tool_result blocks
  result  → e["type"] == "result"   with is_error field
"""

from __future__ import annotations

import io
import json

from harness.events import (
    EventEmitter,
    text_block,
    tool_result_block,
    tool_use_block,
)


def _lines(sink: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in sink.getvalue().splitlines()]


def _mk() -> tuple[EventEmitter, io.StringIO]:
    sink = io.StringIO()
    return (
        EventEmitter(
            session_id="sess-abc",
            model="anthropic/claude-opus-4-8",
            cwd="/tmp/x",
            sink=sink,
        ),
        sink,
    )


def test_system_init_shape() -> None:
    emitter, sink = _mk()
    emitter.system_init(tools=["bash", "read"])
    events = _lines(sink)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "system"
    assert e["subtype"] == "init"
    assert e["session_id"] == "sess-abc"
    assert e["model"] == "anthropic/claude-opus-4-8"
    assert e["cwd"] == "/tmp/x"
    assert e["tools"] == ["bash", "read"]
    assert e["mcp_servers"] == []


def test_assistant_text_and_tool_use() -> None:
    emitter, sink = _mk()
    emitter.assistant(
        content=[
            text_block("I will run ls."),
            tool_use_block("toolu_01", "bash", {"command": "ls"}),
        ],
        stop_reason="tool_use",
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    events = _lines(sink)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "assistant"
    assert e["session_id"] == "sess-abc"
    msg = e["message"]
    assert msg["role"] == "assistant"
    assert msg["stop_reason"] == "tool_use"
    assert msg["usage"]["output_tokens"] == 5
    assert len(msg["content"]) == 2
    assert msg["content"][0]["type"] == "text"
    assert msg["content"][0]["text"] == "I will run ls."
    assert msg["content"][1]["type"] == "tool_use"
    assert msg["content"][1]["name"] == "bash"
    assert msg["content"][1]["input"] == {"command": "ls"}


def test_user_tool_result_shape() -> None:
    emitter, sink = _mk()
    emitter.user_tool_result(
        [
            tool_result_block("toolu_01", "file listing\n", is_error=False),
            tool_result_block("toolu_02", "boom", is_error=True),
        ]
    )
    events = _lines(sink)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "user"
    content = e["message"]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "tool_result"
    assert content[0]["tool_use_id"] == "toolu_01"
    assert content[0]["is_error"] is False
    assert content[1]["is_error"] is True


def test_result_marks_error() -> None:
    emitter, sink = _mk()
    emitter.result(
        is_error=True,
        subtype="error_during_execution",
        text="boom",
        usage={"input_tokens": 100, "output_tokens": 50},
    )
    events = _lines(sink)
    assert len(events) == 1
    e = events[0]
    assert e["type"] == "result"
    assert e["is_error"] is True
    assert e["subtype"] == "error_during_execution"
    assert e["result"] == "boom"
    assert e["session_id"] == "sess-abc"
    assert e["duration_ms"] >= 0


def test_result_default_success() -> None:
    emitter, sink = _mk()
    emitter.result(is_error=False)
    e = _lines(sink)[0]
    assert e["is_error"] is False
    assert e["subtype"] == "success"


def test_num_turns_counted() -> None:
    emitter, sink = _mk()
    emitter.assistant([text_block("hi")], stop_reason="end_turn")
    emitter.assistant([text_block("bye")], stop_reason="end_turn")
    emitter.result(is_error=False)
    result = _lines(sink)[-1]
    assert result["num_turns"] == 2


def test_history_mirrors_sink() -> None:
    emitter, sink = _mk()
    emitter.system_init(tools=["bash"])
    emitter.assistant([text_block("hi")], stop_reason="end_turn")
    emitter.result(is_error=False)
    assert emitter.history == _lines(sink)


def test_lines_are_newline_delimited_json() -> None:
    emitter, sink = _mk()
    emitter.system_init(tools=["bash"])
    emitter.assistant([text_block("hi")], stop_reason="end_turn")
    emitter.result(is_error=False)
    lines = sink.getvalue().splitlines()
    assert len(lines) == 3
    for line in lines:
        assert line.startswith("{") and line.endswith("}")
        json.loads(line)  # each line individually parseable
