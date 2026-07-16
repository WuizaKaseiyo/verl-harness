"""Write tool — overwrite a file (creates parents)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.tools.base import Tool, ToolContext, ToolError


class WriteTool(Tool):
    name = "write"
    description = (
        "Overwrite `file_path` with `content`, creating parent directories "
        "as needed. Use `edit` to modify an existing file — this tool truncates."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "Full new contents of the file.",
            },
        },
        "required": ["file_path", "content"],
    }

    def execute(self, input: dict[str, Any], ctx: ToolContext) -> str:
        raw_path = input.get("file_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ToolError("`file_path` must be a non-empty string")
        content = input.get("content")
        if not isinstance(content, str):
            raise ToolError("`content` must be a string")

        path = Path(raw_path)
        if not path.is_absolute():
            path = (ctx.cwd / path).resolve()

        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.exists()
        path.write_text(content, encoding="utf-8")

        verb = "overwrote" if existed else "created"
        return f"{verb} {path} ({len(content)} bytes)\n"
