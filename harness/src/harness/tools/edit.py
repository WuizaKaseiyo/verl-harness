"""Edit tool — exact-string replacement, uniqueness-enforced by default."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.tools.base import Tool, ToolContext, ToolError


class EditTool(Tool):
    name = "edit"
    description = (
        "Replace an exact substring in a file. "
        "If `replace_all` is false (default), `old_string` must appear exactly once "
        "— otherwise the edit fails with an ambiguity error. Set `replace_all` to "
        "true to replace every occurrence."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "The exact text to replace.",
            },
            "new_string": {
                "type": "string",
                "description": "The replacement text (may be empty).",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence (default false).",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def execute(self, input: dict[str, Any], ctx: ToolContext) -> str:
        raw_path = input.get("file_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ToolError("`file_path` must be a non-empty string")

        path = Path(raw_path)
        if not path.is_absolute():
            path = (ctx.cwd / path).resolve()
        if not path.exists():
            raise ToolError(f"file not found: {path}")

        old = input.get("old_string")
        new = input.get("new_string")
        if not isinstance(old, str):
            raise ToolError("`old_string` must be a string")
        if not isinstance(new, str):
            raise ToolError("`new_string` must be a string")
        if old == new:
            raise ToolError("`old_string` and `new_string` are identical — nothing to do")
        if old == "":
            raise ToolError("`old_string` must be non-empty; use `write` to create a file")

        replace_all = bool(input.get("replace_all", False))

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ToolError(f"file is not utf-8 text: {e}") from e

        count = text.count(old)
        if count == 0:
            raise ToolError("`old_string` not found in file")
        if count > 1 and not replace_all:
            raise ToolError(
                f"`old_string` matches {count} times; pass more surrounding context "
                "to make it unique or set `replace_all=true`"
            )

        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8")

        return (
            f"edited {path}: replaced {count if replace_all else 1} occurrence"
            f"{'s' if (count if replace_all else 1) != 1 else ''}\n"
        )
