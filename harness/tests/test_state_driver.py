"""state_driver.py: scripted backend runs → StateResult + emitted event trace.

These tests do NOT hit a live API. The `ScriptedBackend` from test_loop.py is
reused inline (kept simple; no shared fixture module).

Live-API verification happens in T7 (E2E smoke).
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from harness.backends.base import Backend, RawEvent
from harness.cli import RunConfig
from harness.state_driver import drive_state


REPO_ROOT = Path(__file__).resolve().parents[2]


class ScriptedBackend(Backend):
    model_id = "scripted"

    def __init__(self, script: list[list[RawEvent]]) -> None:
        self._script = script
        self.turn = 0

    async def stream(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> AsyncIterator[RawEvent]:
        # Sanity: intake state should advertise transition_to
        if self.turn == 0:
            names = {t["name"] for t in tools}
            assert "transition_to" in names, "control tool not advertised"
        events = self._script[self.turn]
        self.turn += 1
        for e in events:
            yield e


def _config(workdir: Path, state: str = "intake") -> RunConfig:
    return RunConfig(
        workdir=workdir,
        model="scripted/scripted",
        goal="test goal",
        state=state,
        run_id="R-test",
        hitl=False,
        verl_root=None,
    )


def _run(coro):
    return asyncio.run(coro)


# ── happy path ─────────────────────────────────────────────────────────────

def test_drive_state_writes_deliverable_and_transitions(tmp_path: Path) -> None:
    """Model writes training_intent.md + verl_root.txt via write, then transitions.

    intake → locate_recipe cites `workspace/intake/verl_root.txt` in its
    Deliverables prose, so the workspace-contract check requires that file.
    """
    workspace_root = tmp_path / "runs"
    ws = workspace_root / "R-test" / "workspace"
    target_path = ws / "intake" / "training_intent.md"
    verl_root_path = ws / "intake" / "verl_root.txt"

    script = [
        # turn 1: write training_intent.md
        [
            RawEvent(kind="text_delta", text="Writing training_intent.md."),
            RawEvent(
                kind="tool_use",
                tool_use={
                    "id": "t1",
                    "name": "write",
                    "input": {
                        "file_path": str(target_path),
                        "content": "goal: train\nalgorithm: grpo\nmodel: Qwen/Qwen2.5-3B-Instruct\n",
                    },
                },
            ),
            RawEvent(kind="message_stop", stop_reason="tool_use"),
        ],
        # turn 2: write verl_root.txt (the cited deliverable)
        [
            RawEvent(
                kind="tool_use",
                tool_use={
                    "id": "t2",
                    "name": "write",
                    "input": {
                        "file_path": str(verl_root_path),
                        "content": "/opt/verl",
                    },
                },
            ),
            RawEvent(kind="message_stop", stop_reason="tool_use"),
        ],
        # turn 3: transition_to
        [
            RawEvent(kind="text_delta", text="Intake done."),
            RawEvent(
                kind="tool_use",
                tool_use={
                    "id": "c1",
                    "name": "transition_to",
                    "input": {"next_state": "locate_recipe"},
                },
            ),
            RawEvent(kind="message_stop", stop_reason="tool_use"),
        ],
    ]
    backend = ScriptedBackend(script)
    sink = io.StringIO()
    result = _run(
        drive_state(
            config=_config(REPO_ROOT),
            backend=backend,
            workspace_root=workspace_root,
            session_id="sess-test",
            sink=sink,
        )
    )

    assert result.is_error is False
    assert result.next_state == "locate_recipe"
    assert result.reason == "transition"
    assert target_path.exists()
    assert verl_root_path.exists()
    assert "algorithm: grpo" in target_path.read_text()


def test_drive_state_invalid_transition_target(tmp_path: Path) -> None:
    """Model calls transition_to with a state not in intake's Next States → error.

    The control-tool schema advertises next_state as an enum, but the model may
    still send a bogus string (or the backend may not enforce enums). The driver
    validates.
    """
    workspace_root = tmp_path / "runs"

    script = [
        [
            RawEvent(
                kind="tool_use",
                tool_use={
                    "id": "c1",
                    "name": "transition_to",
                    "input": {"next_state": "bogus_state"},
                },
            ),
            RawEvent(kind="message_stop", stop_reason="tool_use"),
        ]
    ]
    backend = ScriptedBackend(script)
    sink = io.StringIO()
    result = _run(
        drive_state(
            config=_config(REPO_ROOT),
            backend=backend,
            workspace_root=workspace_root,
            session_id="sess-test",
            sink=sink,
        )
    )

    assert result.is_error is True
    assert result.reason == "invalid_transition"
    assert "bogus_state" in result.message


def test_drive_state_end_turn_without_transition_is_error(tmp_path: Path) -> None:
    workspace_root = tmp_path / "runs"
    script = [
        [
            RawEvent(kind="text_delta", text="I'm done."),
            RawEvent(kind="message_stop", stop_reason="end_turn"),
        ]
    ]
    backend = ScriptedBackend(script)
    sink = io.StringIO()
    result = _run(
        drive_state(
            config=_config(REPO_ROOT),
            backend=backend,
            workspace_root=workspace_root,
            session_id="sess-test",
            sink=sink,
        )
    )
    assert result.is_error is True
    assert result.reason == "no_transition"


# ── terminal state (finalize) ─────────────────────────────────────────────

def test_drive_terminal_state_end_turn_is_success(tmp_path: Path) -> None:
    """finalize has no ## Next States — end_turn is success."""
    workspace_root = tmp_path / "runs"

    class TerminalBackend(Backend):
        model_id = "t"

        def __init__(self):
            self.turn = 0

        async def stream(
            self, *, system, messages, tools, max_tokens=4096,
        ) -> AsyncIterator[RawEvent]:
            names = {t["name"] for t in tools}
            assert "transition_to" not in names, "control tool leaked into terminal state"
            self.turn += 1
            yield RawEvent(kind="text_delta", text="finalize done.")
            yield RawEvent(kind="message_stop", stop_reason="end_turn")

    result = _run(
        drive_state(
            config=_config(REPO_ROOT, state="finalize"),
            backend=TerminalBackend(),
            workspace_root=workspace_root,
            session_id="sess-t",
            sink=io.StringIO(),
        )
    )
    assert result.is_error is False
    assert result.reason == "terminal_end"
    assert result.next_state is None


# ── context error ──────────────────────────────────────────────────────────

def test_drive_state_missing_state_file(tmp_path: Path) -> None:
    """workdir with no states/ dir → context error, but events emit cleanly."""
    (tmp_path / "states").mkdir()  # empty
    config = _config(tmp_path, state="nonexistent")

    class NeverCalledBackend(Backend):
        model_id = "n"

        async def stream(
            self, *, system, messages, tools, max_tokens=4096,
        ) -> AsyncIterator[RawEvent]:
            raise AssertionError("backend must not be invoked when context fails")
            yield  # unreachable — pragma: no cover

    sink = io.StringIO()
    result = _run(
        drive_state(
            config=config,
            backend=NeverCalledBackend(),
            workspace_root=tmp_path / "runs",
            session_id="sess-t",
            sink=sink,
        )
    )
    assert result.is_error is True
    assert result.reason == "context_error"
    # system.init still emitted so the dashboard sees the run
    assert '"type":"system"' in sink.getvalue()
    assert '"type":"result"' in sink.getvalue()


# ── event trace shape ──────────────────────────────────────────────────────

def test_drive_state_contract_retry_recovers(tmp_path: Path) -> None:
    """Model tries transition_to too early → gets rejection → writes files → tries again.

    Proves the contract-retry loop actually recovers when the model corrects.
    """
    workspace_root = tmp_path / "runs"
    ws = workspace_root / "R-test" / "workspace"
    verl_root_path = ws / "intake" / "verl_root.txt"

    script = [
        # Attempt 1: bare transition (no deliverables yet) → will be rejected
        [
            RawEvent(
                kind="tool_use",
                tool_use={
                    "id": "c1",
                    "name": "transition_to",
                    "input": {"next_state": "locate_recipe"},
                },
            ),
            RawEvent(kind="message_stop", stop_reason="tool_use"),
        ],
        # Attempt 2 (after rejection): write the file
        [
            RawEvent(
                kind="tool_use",
                tool_use={
                    "id": "t1",
                    "name": "write",
                    "input": {
                        "file_path": str(verl_root_path),
                        "content": "/opt/verl",
                    },
                },
            ),
            RawEvent(kind="message_stop", stop_reason="tool_use"),
        ],
        # Attempt 2 continued: transition_to again
        [
            RawEvent(
                kind="tool_use",
                tool_use={
                    "id": "c2",
                    "name": "transition_to",
                    "input": {"next_state": "locate_recipe"},
                },
            ),
            RawEvent(kind="message_stop", stop_reason="tool_use"),
        ],
    ]
    backend = ScriptedBackend(script)
    sink = io.StringIO()
    result = _run(
        drive_state(
            config=_config(REPO_ROOT),
            backend=backend,
            workspace_root=workspace_root,
            session_id="sess-retry",
            sink=sink,
        )
    )
    assert result.is_error is False
    assert result.next_state == "locate_recipe"
    assert verl_root_path.exists()

    # Assert the rejection tool_result actually got emitted
    lines = [ln for ln in sink.getvalue().splitlines() if ln]
    import json as _json
    events = [_json.loads(ln) for ln in lines]
    user_events = [e for e in events if e.get("type") == "user"]
    # At least one user event carries a tool_result with is_error=True from the contract check
    contract_errors = [
        e for e in user_events
        for b in e["message"]["content"]
        if b.get("is_error") is True and "verl_root.txt" in b.get("content", "")
    ]
    assert contract_errors, "expected a rejection tool_result mentioning verl_root.txt"


def test_drive_state_contract_exhausts_retries(tmp_path: Path) -> None:
    """Model keeps calling transition_to without writing files → runtime gives up."""
    workspace_root = tmp_path / "runs"
    # Never write anything — every attempt fails
    script = [
        [
            RawEvent(
                kind="tool_use",
                tool_use={
                    "id": f"c{i}",
                    "name": "transition_to",
                    "input": {"next_state": "locate_recipe"},
                },
            ),
            RawEvent(kind="message_stop", stop_reason="tool_use"),
        ]
        for i in range(10)  # more than max_contract_retries
    ]
    backend = ScriptedBackend(script)
    sink = io.StringIO()
    result = _run(
        drive_state(
            config=_config(REPO_ROOT),
            backend=backend,
            workspace_root=workspace_root,
            session_id="sess-exhaust",
            sink=sink,
            max_contract_retries=2,
        )
    )
    assert result.is_error is True
    assert result.reason == "contract_violation"
    assert "verl_root.txt" in result.message


def test_drive_state_emits_dashboard_shape(tmp_path: Path) -> None:
    workspace_root = tmp_path / "runs"
    # Pre-seed the workspace with the cited deliverable so the contract check
    # accepts on the first attempt.
    ws = workspace_root / "R-test" / "workspace" / "intake"
    ws.mkdir(parents=True)
    (ws / "verl_root.txt").write_text("/opt/verl")

    script = [
        [
            RawEvent(
                kind="tool_use",
                tool_use={
                    "id": "c1",
                    "name": "transition_to",
                    "input": {"next_state": "locate_recipe"},
                },
            ),
            RawEvent(kind="message_stop", stop_reason="tool_use"),
        ]
    ]
    backend = ScriptedBackend(script)
    sink = io.StringIO()
    _run(
        drive_state(
            config=_config(REPO_ROOT),
            backend=backend,
            workspace_root=workspace_root,
            session_id="sess-shape",
            sink=sink,
        )
    )
    lines = [line for line in sink.getvalue().splitlines() if line]
    types = []
    import json as _json
    for line in lines:
        types.append(_json.loads(line)["type"])
    # system.init → assistant → result
    assert types == ["system", "assistant", "result"]
