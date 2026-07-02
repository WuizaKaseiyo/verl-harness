"""Shared logging helpers for verl-harness built-in tools."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_tool_call(
    *,
    workspace: str | Path | None,
    tool: str,
    args: dict[str, Any],
    status: str,
    output_path: str | None = None,
    error: str | None = None,
) -> None:
    """Append one JSONL record to `<workspace>/logs/tool_calls.jsonl`.

    Logging is best-effort. Tool execution should not fail merely because the log
    cannot be written.
    """
    if workspace is None:
        return
    try:
        root = Path(workspace).resolve()
        logs_dir = root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "args": args,
            "status": status,
            "output_path": output_path,
            "error": error,
        }
        with (logs_dir / "tool_calls.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        return
