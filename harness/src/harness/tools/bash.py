"""Bash tool — subprocess with cwd + env + timeout, output truncation."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from harness.tools.base import Tool, ToolContext, ToolError

_DEFAULT_TIMEOUT_MS = 120_000
_MAX_TIMEOUT_MS = 30 * 60 * 1000  # 30 min
_MAX_OUTPUT_BYTES = 100 * 1024  # 100 KB


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit - 200
    tail = 100
    return (
        text[:head]
        + f"\n[... {len(text) - head - tail} bytes truncated ...]\n"
        + text[-tail:]
    )


class BashTool(Tool):
    name = "bash"
    description = (
        "Execute a bash command in a subprocess with the given cwd and env. "
        "Returns combined stdout+stderr; truncated at 100 KB. "
        "Commands must complete within `timeout_ms` (default 120000, max 1800000)."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to run.",
            },
            "timeout_ms": {
                "type": "integer",
                "description": (
                    "Timeout in milliseconds "
                    f"(default {_DEFAULT_TIMEOUT_MS}, max {_MAX_TIMEOUT_MS})."
                ),
            },
            "description": {
                "type": "string",
                "description": "One-line description of what this command does.",
            },
        },
        "required": ["command"],
    }

    def execute(self, input: dict[str, Any], ctx: ToolContext) -> str:
        command = input.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolError("`command` must be a non-empty string")

        timeout_ms = int(input.get("timeout_ms", _DEFAULT_TIMEOUT_MS))
        if timeout_ms < 1 or timeout_ms > _MAX_TIMEOUT_MS:
            raise ToolError(
                f"`timeout_ms` must be in [1, {_MAX_TIMEOUT_MS}], got {timeout_ms}"
            )
        timeout_s = timeout_ms / 1000.0

        env = os.environ.copy()
        env.update(ctx.env)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(ctx.cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            partial = (e.stdout or "") + (e.stderr or "")
            partial = partial if isinstance(partial, str) else partial.decode("utf-8", "replace")
            return _truncate(
                f"[error: command timed out after {timeout_s:.1f}s]\n"
                f"--- partial output ---\n{partial}",
                _MAX_OUTPUT_BYTES,
            )

        parts: list[str] = []
        if proc.stdout:
            parts.append(proc.stdout)
        if proc.stderr:
            parts.append(f"--- stderr ---\n{proc.stderr}")
        if proc.returncode != 0:
            parts.append(f"[exit code {proc.returncode}]")
        elif not parts:
            parts.append("[ok — no output]")

        return _truncate("\n".join(parts).rstrip() + "\n", _MAX_OUTPUT_BYTES)
