"""stream-json event emission — matches Claude Code's format so the existing
`web/` dashboard consumes runtime output with no parser changes.

Each event is a single JSON object on its own line (newline-delimited JSON, NDJSON).
Shape reference: web/src/verl_harness_web/submit.py reads:

  {"type": "system", "subtype": "init", ...}
  {"type": "assistant", "message": {"content": [{"type": "text", ...},
                                                 {"type": "tool_use", ...}]}, ...}
  {"type": "user",      "message": {"content": [{"type": "tool_result", ...}]}, ...}
  {"type": "result", "is_error": bool, ...}

M1 emits one `assistant` event per completed model turn (with the full accumulated
content), not per streaming delta. That's what parsers actually need to extract
tool_use blocks. Live per-delta streaming can be added later without breaking
consumers.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import IO, Any


@dataclass
class EventEmitter:
    """Writes newline-JSON events to a sink. Default sink is stdout.

    Also captures every event into `.history` so callers (state driver, tests)
    can inspect what was emitted without re-parsing stdout.
    """

    session_id: str
    model: str
    cwd: str
    sink: IO[str] = field(default_factory=lambda: sys.stdout)
    history: list[dict[str, Any]] = field(default_factory=list)
    _t0: float = field(default_factory=time.monotonic)
    _turns: int = 0

    # ── low-level ──────────────────────────────────────────────────────────

    def _emit(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        self.sink.write(line + "\n")
        try:
            self.sink.flush()
        except (OSError, ValueError):
            pass  # closed sink — history still gets the event
        self.history.append(event)

    # ── high-level (call these from the loop) ──────────────────────────────

    def system_init(self, tools: list[str]) -> None:
        """Emit exactly once at the start of a run."""
        self._emit(
            {
                "type": "system",
                "subtype": "init",
                "session_id": self.session_id,
                "cwd": self.cwd,
                "model": self.model,
                "tools": list(tools),
                "mcp_servers": [],
            }
        )

    def assistant(
        self,
        content: list[dict[str, Any]],
        *,
        stop_reason: str | None,
        usage: dict[str, int] | None = None,
    ) -> None:
        """Emit one assistant message with the full content list for this turn."""
        self._turns += 1
        self._emit(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "model": self.model,
                    "content": content,
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                    "usage": usage or {},
                },
                "session_id": self.session_id,
            }
        )

    def user_tool_result(
        self,
        results: list[dict[str, Any]],
    ) -> None:
        """Emit a user-role message carrying one or more tool_result blocks."""
        self._emit(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": results,
                },
                "session_id": self.session_id,
            }
        )

    def approval_request(self, request: dict[str, Any]) -> None:
        """Emit an `approval_request` event carrying the hand-off details.

        Consumed by `events.jsonl` tailers (audit trail). The dashboard reads
        the request FILE directly, not this event — the event is redundant
        with the file but useful for after-the-fact log replay.
        """
        self._emit(
            {
                "type": "approval_request",
                "session_id": self.session_id,
                "request": request,
            }
        )

    def approval_decision(self, request_id: str, decision: str) -> None:
        """Emit the paired `approval_decision` event once the user responds."""
        self._emit(
            {
                "type": "approval_decision",
                "session_id": self.session_id,
                "request_id": request_id,
                "decision": decision,
            }
        )

    def result(
        self,
        *,
        is_error: bool,
        subtype: str = "success",
        text: str = "",
        usage: dict[str, int] | None = None,
    ) -> None:
        """Emit the terminal event. Exactly one per run."""
        self._emit(
            {
                "type": "result",
                "subtype": subtype,
                "is_error": is_error,
                "duration_ms": int((time.monotonic() - self._t0) * 1000),
                "num_turns": self._turns,
                "result": text,
                "session_id": self.session_id,
                "usage": usage or {},
            }
        )


# ── convenience block builders — Anthropic content-block shape ──────────────


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def tool_use_block(tool_id: str, name: str, input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": input}


def tool_result_block(
    tool_use_id: str,
    content: str,
    *,
    is_error: bool = False,
) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": is_error,
    }
