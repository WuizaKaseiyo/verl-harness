"""Todo tool — run-scoped in-memory task list (mirrors Claude Code's TaskCreate/List)."""

from __future__ import annotations

from typing import Any

from harness.tools.base import Tool, ToolContext, ToolError

_VALID_STATUSES = {"pending", "in_progress", "completed"}


class TodoTool(Tool):
    name = "todo"
    description = (
        "Replace the run's todo list. Provide the full new list each call — "
        "the tool overwrites, not appends. Each item: {id, content, status}. "
        "status is one of pending / in_progress / completed."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "Full replacement list.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "content": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": sorted(_VALID_STATUSES),
                        },
                    },
                    "required": ["id", "content", "status"],
                },
            }
        },
        "required": ["todos"],
    }

    def execute(self, input: dict[str, Any], ctx: ToolContext) -> str:
        todos = input.get("todos")
        if not isinstance(todos, list):
            raise ToolError("`todos` must be an array")

        clean: list[dict[str, str]] = []
        for i, item in enumerate(todos):
            if not isinstance(item, dict):
                raise ToolError(f"todos[{i}] must be an object")
            iid = item.get("id")
            content = item.get("content")
            status = item.get("status")
            if not (isinstance(iid, str) and iid):
                raise ToolError(f"todos[{i}].id must be a non-empty string")
            if not (isinstance(content, str) and content):
                raise ToolError(f"todos[{i}].content must be a non-empty string")
            if status not in _VALID_STATUSES:
                raise ToolError(
                    f"todos[{i}].status must be one of {sorted(_VALID_STATUSES)}, "
                    f"got {status!r}"
                )
            clean.append({"id": iid, "content": content, "status": status})

        ctx.todo_state["todos"] = clean

        counts = {s: sum(1 for t in clean if t["status"] == s) for s in _VALID_STATUSES}
        return (
            f"todos updated ({len(clean)} items: "
            f"{counts['pending']} pending, "
            f"{counts['in_progress']} in_progress, "
            f"{counts['completed']} completed)\n"
        )
