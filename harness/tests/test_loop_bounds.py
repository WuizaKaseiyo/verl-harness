"""Loop cycle bound tests — reflect → configure_algorithm max_iterations semantics.

Uses the real reflect.md (max_iterations: 3) but scripts the backend so we
don't need a live model.
"""

from __future__ import annotations

import asyncio
import io
import json
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
        events = self._script[self.turn]
        self.turn += 1
        for e in events:
            yield e


def _reflect_config(workdir: Path) -> RunConfig:
    return RunConfig(
        workdir=REPO_ROOT,
        model="scripted/x",
        goal="reflect loop test",
        state="reflect",
        run_id="loop-test",
        hitl=False,
        verl_root=None,
    )


def _seed_reflect_deliverables(workspace_root: Path, run_id: str) -> None:
    ws = workspace_root / run_id / "workspace"
    (ws / "reflect").mkdir(parents=True, exist_ok=True)
    (ws / "reflect" / "refinement_plan.md").write_text(
        "delta: {lr: 0.5x}\n"
    )
    (ws / "reflect" / "reflect_report.md").write_text(
        "history:\nstop_reason: budget_hit\n"
    )


def _prime_loop_counter(workspace_root: Path, run_id: str, edge: str, count: int) -> None:
    """Seed workspace/reflect/loop_state.json with a pre-existing count."""
    ws = workspace_root / run_id / "workspace"
    (ws / "reflect").mkdir(parents=True, exist_ok=True)
    (ws / "reflect" / "loop_state.json").write_text(
        json.dumps({"edges": {edge: count}}, indent=2)
    )


def _run(coro):
    return asyncio.run(coro)


def _txn(target: str, tid: str) -> list[RawEvent]:
    return [
        RawEvent(
            kind="tool_use",
            tool_use={"id": tid, "name": "transition_to", "input": {"next_state": target}},
        ),
        RawEvent(kind="message_stop", stop_reason="tool_use"),
    ]


# ── happy path: fresh state, first traversal accepted, counter increments ──

def test_first_loop_traversal_accepted(tmp_path: Path) -> None:
    workspace_root = tmp_path / "runs"
    _seed_reflect_deliverables(workspace_root, "loop-test")

    script = [_txn("configure_algorithm", "c1")]
    result = _run(
        drive_state(
            config=_reflect_config(REPO_ROOT),
            backend=ScriptedBackend(script),
            workspace_root=workspace_root,
            session_id="s",
            sink=io.StringIO(),
        )
    )
    assert result.is_error is False
    assert result.next_state == "configure_algorithm"

    # loop_state.json has count 1 for this edge
    ls = json.loads(
        (workspace_root / "loop-test" / "workspace" / "reflect" / "loop_state.json").read_text()
    )
    assert ls["edges"]["reflect->configure_algorithm"] == 1


# ── exhausted budget: model insists on loop → runtime rejects with retries ─

def test_exhausted_budget_rejects_and_redirects(tmp_path: Path) -> None:
    """Counter pre-primed at 3; model tries once more, gets rejected, then picks finalize."""
    workspace_root = tmp_path / "runs"
    _seed_reflect_deliverables(workspace_root, "loop-test")
    _prime_loop_counter(
        workspace_root, "loop-test", "reflect->configure_algorithm", 3
    )

    script = [
        _txn("configure_algorithm", "c1"),  # rejected
        _txn("finalize", "c2"),             # accepted (exit branch)
    ]
    result = _run(
        drive_state(
            config=_reflect_config(REPO_ROOT),
            backend=ScriptedBackend(script),
            workspace_root=workspace_root,
            session_id="s",
            sink=io.StringIO(),
        )
    )
    assert result.is_error is False
    assert result.next_state == "finalize"


def test_budget_error_when_model_wont_redirect(tmp_path: Path) -> None:
    """Counter at 3, model keeps trying configure_algorithm → runtime gives up."""
    workspace_root = tmp_path / "runs"
    _seed_reflect_deliverables(workspace_root, "loop-test")
    _prime_loop_counter(
        workspace_root, "loop-test", "reflect->configure_algorithm", 3
    )

    script = [_txn("configure_algorithm", f"c{i}") for i in range(6)]
    sink = io.StringIO()
    result = _run(
        drive_state(
            config=_reflect_config(REPO_ROOT),
            backend=ScriptedBackend(script),
            workspace_root=workspace_root,
            session_id="s",
            sink=sink,
            max_contract_retries=2,
        )
    )
    assert result.is_error is True
    assert result.reason == "loop_budget_exhausted"
    assert "reflect->configure_algorithm" in result.message or "configure_algorithm" in result.message

    # Emitted a user tool_result explaining the budget error
    events = [json.loads(ln) for ln in sink.getvalue().splitlines() if ln]
    user_events = [e for e in events if e.get("type") == "user"]
    assert any(
        "max_iterations" in b.get("content", "")
        for e in user_events
        for b in e["message"]["content"]
    )


# ── counter persistence across state entries ─────────────────────────────

def test_counter_accumulates_across_entries(tmp_path: Path) -> None:
    """Enter reflect twice, each time transitioning to configure_algorithm.
    After two runs, the counter should be at 2."""
    workspace_root = tmp_path / "runs"
    _seed_reflect_deliverables(workspace_root, "loop-test")

    for _ in range(2):
        script = [_txn("configure_algorithm", "c")]
        result = _run(
            drive_state(
                config=_reflect_config(REPO_ROOT),
                backend=ScriptedBackend(script),
                workspace_root=workspace_root,
                session_id="s",
                sink=io.StringIO(),
            )
        )
        assert result.is_error is False

    ls = json.loads(
        (workspace_root / "loop-test" / "workspace" / "reflect" / "loop_state.json").read_text()
    )
    assert ls["edges"]["reflect->configure_algorithm"] == 2


# ── non-loop edges don't touch the counter ───────────────────────────────

def test_non_loop_edge_does_not_touch_counter(tmp_path: Path) -> None:
    """reflect → finalize has no Loop marker → counter file stays absent."""
    workspace_root = tmp_path / "runs"
    _seed_reflect_deliverables(workspace_root, "loop-test")

    script = [_txn("finalize", "c1")]
    result = _run(
        drive_state(
            config=_reflect_config(REPO_ROOT),
            backend=ScriptedBackend(script),
            workspace_root=workspace_root,
            session_id="s",
            sink=io.StringIO(),
        )
    )
    assert result.is_error is False
    assert result.next_state == "finalize"

    loop_state_path = (
        workspace_root / "loop-test" / "workspace" / "reflect" / "loop_state.json"
    )
    assert not loop_state_path.exists()
