"""Anthropic ↔ OpenAI translation for the openai-wire backend.

Runtime internals speak Anthropic message + tool-schema shape. When we call an
OpenAI-compatible endpoint (OpenAI itself, OpenRouter, DeepSeek, vLLM, …) we
translate on the boundary here.

Only the shape maps we actually need in the tool loop live in this file.
"""

from __future__ import annotations

import json
from typing import Any


# ── tools ──────────────────────────────────────────────────────────────────


def tools_anthropic_to_openai(
    anthropic_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """`{"name","description","input_schema"}` → OpenAI function shape."""
    out: list[dict[str, Any]] = []
    for t in anthropic_tools:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object"}),
                },
            }
        )
    return out


# ── messages ───────────────────────────────────────────────────────────────


def _content_to_str(content: Any) -> str:
    """Anthropic text content: either a string or a list with a single text block."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        return "".join(parts)
    return ""


def messages_anthropic_to_openai(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate the loop's Anthropic-native conversation to OpenAI chat shape.

    Each Anthropic user message with tool_result blocks fans out into one OpenAI
    `role=tool` message per block. Assistant messages with tool_use blocks
    collapse text+tool_use into a single OpenAI assistant message with a
    `tool_calls` array.

    `is_error=true` tool_results are prefixed with "[error]\\n" — OpenAI's
    `role=tool` shape has no is_error field, so the marker gets embedded so the
    model can see it.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
                continue
            # list content — may hold text blocks and/or tool_result blocks
            text_parts: list[str] = []
            tool_msgs: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    body = block.get("content", "")
                    if isinstance(body, list):
                        body = _content_to_str(body)
                    if block.get("is_error"):
                        body = f"[error]\n{body}"
                    tool_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": body,
                        }
                    )
            if text_parts:
                out.append({"role": "user", "content": "".join(text_parts)})
            out.extend(tool_msgs)
            continue

        if role == "assistant":
            if isinstance(content, str):
                out.append({"role": "assistant", "content": content})
                continue
            text_parts_a: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts_a.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block["name"],
                                "arguments": json.dumps(block.get("input") or {}),
                            },
                        }
                    )
            entry: dict[str, Any] = {"role": "assistant"}
            joined = "".join(text_parts_a)
            entry["content"] = joined if joined else None
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
            continue

        # system role is passed via top-level `system` kwarg, not messages
    return out


# ── finish reasons ─────────────────────────────────────────────────────────


def finish_reason_to_anthropic(finish_reason: str | None) -> str | None:
    """Map OpenAI's finish_reason to Anthropic's stop_reason vocabulary."""
    if finish_reason is None:
        return None
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",  # legacy
        "content_filter": "end_turn",
    }
    return mapping.get(finish_reason, "end_turn")
