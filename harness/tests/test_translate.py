"""Anthropic ↔ OpenAI translation unit tests.

Covers the message + tool + finish-reason mappings. No live API calls.
"""

from __future__ import annotations

import json

from harness.backends.translate import (
    finish_reason_to_anthropic,
    messages_anthropic_to_openai,
    tools_anthropic_to_openai,
)


# ── tools ───────────────────────────────────────────────────────────────────

def test_tools_rename_input_schema_to_parameters() -> None:
    tools = [
        {
            "name": "bash",
            "description": "run a shell cmd",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        }
    ]
    out = tools_anthropic_to_openai(tools)
    assert out == [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "run a shell cmd",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]


def test_tools_defaults_when_optional_fields_missing() -> None:
    tools = [{"name": "todo"}]
    out = tools_anthropic_to_openai(tools)
    assert out[0]["function"]["description"] == ""
    assert out[0]["function"]["parameters"] == {"type": "object"}


# ── messages ────────────────────────────────────────────────────────────────

def test_user_string_content_passthrough() -> None:
    out = messages_anthropic_to_openai([{"role": "user", "content": "hi"}])
    assert out == [{"role": "user", "content": "hi"}]


def test_assistant_text_only() -> None:
    out = messages_anthropic_to_openai(
        [{"role": "assistant", "content": [{"type": "text", "text": "answer"}]}]
    )
    assert out == [{"role": "assistant", "content": "answer"}]


def test_assistant_tool_use_collapses_to_tool_calls() -> None:
    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "running ls"},
                {
                    "type": "tool_use",
                    "id": "tu1",
                    "name": "bash",
                    "input": {"command": "ls"},
                },
            ],
        }
    ]
    out = messages_anthropic_to_openai(msgs)
    assert len(out) == 1
    entry = out[0]
    assert entry["role"] == "assistant"
    assert entry["content"] == "running ls"
    assert entry["tool_calls"] == [
        {
            "id": "tu1",
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": "ls"})},
        }
    ]


def test_assistant_tool_use_only_content_is_null() -> None:
    msgs = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t", "name": "read", "input": {"file_path": "/x"}},
            ],
        }
    ]
    out = messages_anthropic_to_openai(msgs)
    assert out[0]["content"] is None
    assert out[0]["tool_calls"][0]["function"]["name"] == "read"


def test_tool_results_fan_out_to_tool_role_messages() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "hi\n", "is_error": False},
                {"type": "tool_result", "tool_use_id": "t2", "content": "err", "is_error": True},
            ],
        }
    ]
    out = messages_anthropic_to_openai(msgs)
    assert len(out) == 2
    assert out[0] == {"role": "tool", "tool_call_id": "t1", "content": "hi\n"}
    assert out[1]["role"] == "tool"
    assert out[1]["tool_call_id"] == "t2"
    assert out[1]["content"].startswith("[error]")
    assert "err" in out[1]["content"]


def test_user_text_and_tool_results_mixed() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "here you go"},
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            ],
        }
    ]
    out = messages_anthropic_to_openai(msgs)
    assert out == [
        {"role": "user", "content": "here you go"},
        {"role": "tool", "tool_call_id": "t1", "content": "ok"},
    ]


def test_full_turn_roundtrip_shape() -> None:
    """A realistic 3-message conversation preserving order + roles."""
    msgs = [
        {"role": "user", "content": "list the files"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "sure"},
                {"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a b c"}],
        },
    ]
    out = messages_anthropic_to_openai(msgs)
    roles = [m["role"] for m in out]
    assert roles == ["user", "assistant", "tool"]


# ── finish reasons ──────────────────────────────────────────────────────────

def test_finish_reason_mapping() -> None:
    assert finish_reason_to_anthropic("stop") == "end_turn"
    assert finish_reason_to_anthropic("length") == "max_tokens"
    assert finish_reason_to_anthropic("tool_calls") == "tool_use"
    assert finish_reason_to_anthropic("function_call") == "tool_use"
    assert finish_reason_to_anthropic("content_filter") == "end_turn"
    assert finish_reason_to_anthropic(None) is None
    assert finish_reason_to_anthropic("weird_new_reason") == "end_turn"
