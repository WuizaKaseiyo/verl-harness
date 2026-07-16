"""Tool abstraction + execution context.

Every tool is a small class exposing:
  - name / description / input_schema  → Anthropic tool schema fields
  - execute(input, ctx) -> str          → sync, returns tool_result content string

Tools raise `ToolError` for user-facing failures (bad input, file missing).
The registry turns those into `is_error=True` tool_result blocks. Unexpected
exceptions get wrapped the same way so the loop never crashes on a tool bug.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ToolError(Exception):
    """User-facing tool error — becomes `tool_result` with is_error=true."""


@dataclass
class ToolContext:
    """Everything a tool might need from the run.

    Mutable (todo_state) so tools can persist per-run state without a DB.
    """

    cwd: Path
    workdir: Path
    run_id: str
    workspace: Path
    env: dict[str, str] = field(default_factory=dict)
    todo_state: dict[str, Any] = field(default_factory=dict)


class Tool(abc.ABC):
    """A callable tool the agent can invoke."""

    name: str
    description: str
    input_schema: dict[str, Any]

    @abc.abstractmethod
    def execute(self, input: dict[str, Any], ctx: ToolContext) -> str:
        """Run the tool. Return the string that becomes tool_result content."""
        raise NotImplementedError

    def schema(self) -> dict[str, Any]:
        """Anthropic tool schema shape."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
