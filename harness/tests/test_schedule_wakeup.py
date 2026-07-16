"""schedule_wakeup / long-running state tests.

Uses scripted backends. Real API in E2E smoke (T10).
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
from harness.hitl import AutoApprovePrompter
from harness.orchestrator import MAX_TOTAL_WAIT_PER_STATE_SEC, orchestrate
from harness.state_driver import WAIT_CAPABLE_STATES


REPO_ROOT = Path(__file__).resolve().parents[2]


class RouterBackend(Backend):
    """Same scripted-router shape as other tests; also records per-state entries."""

    model_id = "router"

    def __init__(self, scripts: dict[str, list[list[RawEvent]]]) -> None:
        self.scripts = scripts
        self.counters: dict[str, int] = {name: 0 for name in scripts}
        self.tool_names_seen: dict[str, set[str]] = {}

    async def stream(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> AsyncIterator[RawEvent]:
        state = _detect_state(system)
        self.tool_names_seen.setdefault(state, set()).update(t["name"] for t in tools)
        script = self.scripts.get(state)
        assert script is not None, f"no script for {state}"
        idx = self.counters[state]
        self.counters[state] = idx + 1
        events = script[idx]
        for e in events:
            yield e


def _detect_state(system) -> str:
    if isinstance(system, list):
        system = "\n".join(b.get("text", "") for b in system if isinstance(b, dict))
    marker = "## states/"
    i = system.find(marker)
    tail = system[i + len(marker) :]
    return tail[: tail.find(".md")]


def _wakeup(seconds: int, reason: str, tid: str = "w1") -> list[RawEvent]:
    return [
        RawEvent(
            kind="tool_use",
            tool_use={
                "id": tid,
                "name": "schedule_wakeup",
                "input": {"delaySeconds": seconds, "reason": reason},
            },
        ),
        RawEvent(kind="message_stop", stop_reason="tool_use"),
    ]


def _txn(target: str, tid: str = "c1") -> list[RawEvent]:
    return [
        RawEvent(
            kind="tool_use",
            tool_use={"id": tid, "name": "transition_to", "input": {"next_state": target}},
        ),
        RawEvent(kind="message_stop", stop_reason="tool_use"),
    ]


def _end_turn() -> list[RawEvent]:
    return [
        RawEvent(kind="text_delta", text="done"),
        RawEvent(kind="message_stop", stop_reason="end_turn"),
    ]


def _seed(workspace_root: Path, run_id: str) -> None:
    """Deliverables seed that lets a monitor_training→summarize walk pass contracts."""
    ws = workspace_root / run_id / "workspace"
    files = {
        "intake/verl_root.txt": "/opt/verl",
        "intake/training_intent.md": "goal: train\n",
        "recipe/recipe.md": "launch: /opt/verl/x.sh\n",
        "algorithm/algorithm_config.md": "estimator: gspo\n",
        "dataset/dataset.md": "train_files: []\n",
        "reward/reward_config.md": "reward_kind: rule\n",
        "compute/compute_choice.md": "target: local-direct\n",
        "env/env_state.md": "torch: 2.4.0\n",
        "env/launch_env.sh": "#!/bin/bash\n",
        "sanity/sanity_report.md": "verdict: green\n",
        "job/job_info.md": "pid: 12345\n",
        "job/job_status.md": "success\n",
        "logs/job_log.md": "training complete\n",
        "summary/summary.md": "topline: success\n",
    }
    for rel, body in files.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)


def _cfg(state: str, run_id: str) -> RunConfig:
    return RunConfig(
        workdir=REPO_ROOT,
        model="scripted/r",
        goal="wakeup test",
        state=state,
        run_id=run_id,
        hitl=False,
        verl_root=None,
    )


def _run(coro):
    return asyncio.run(coro)


# ── WAIT_CAPABLE_STATES is exposed and stable ─────────────────────────────

def test_wait_capable_states_advertised() -> None:
    assert "monitor_training" in WAIT_CAPABLE_STATES
    assert "launch_training" in WAIT_CAPABLE_STATES
    assert "sanity_rollout" in WAIT_CAPABLE_STATES
    assert "intake" not in WAIT_CAPABLE_STATES


# ── tool exposure gated by state name ─────────────────────────────────────

def test_intake_does_not_expose_schedule_wakeup(tmp_path: Path) -> None:
    workspace_root = tmp_path / "runs"
    # eval-track path: intake → run_eval → finalize (only 3 states, minimal)
    ws = workspace_root / "no-wake" / "workspace"
    (ws / "intake").mkdir(parents=True, exist_ok=True)
    (ws / "intake" / "verl_root.txt").write_text("/opt/verl")
    (ws / "intake" / "training_intent.md").write_text("goal: eval\n")
    (ws / "eval").mkdir(parents=True, exist_ok=True)
    (ws / "eval" / "eval_report.md").write_text("acc: 0.5\n")

    scripts = {
        "intake": [_txn("run_eval")],
        "run_eval": [_txn("finalize")],
        "finalize": [_end_turn()],
    }
    backend = RouterBackend(scripts)
    _run(
        orchestrate(
            config=_cfg("intake", "no-wake"),
            backend=backend,
            workspace_root=workspace_root,
            session_id="s",
            sink=io.StringIO(),
            hitl_prompter=AutoApprovePrompter(),
        )
    )
    assert "schedule_wakeup" not in backend.tool_names_seen.get("intake", set())
    assert "schedule_wakeup" not in backend.tool_names_seen.get("run_eval", set())


def test_monitor_training_exposes_schedule_wakeup(tmp_path: Path) -> None:
    workspace_root = tmp_path / "runs"
    _seed(workspace_root, "mt-1")
    scripts = {"monitor_training": [_txn("summarize")], "summarize": [_txn("finalize")], "finalize": [_end_turn()]}
    backend = RouterBackend(scripts)
    _run(
        orchestrate(
            config=_cfg("monitor_training", "mt-1"),
            backend=backend,
            workspace_root=workspace_root,
            session_id="s",
            sink=io.StringIO(),
            hitl_prompter=AutoApprovePrompter(),
        )
    )
    assert "schedule_wakeup" in backend.tool_names_seen.get("monitor_training", set())


# ── schedule_wakeup fires, orchestrator sleeps, state re-enters ──────────

def test_wakeup_reenters_same_state(tmp_path: Path) -> None:
    """monitor_training: 2 wakeups then transition. Should enter monitor_training 3× total."""
    workspace_root = tmp_path / "runs"
    _seed(workspace_root, "mt-loop")

    scripts = {
        "monitor_training": [
            _wakeup(30, "job still running", "w1"),
            _wakeup(30, "still running", "w2"),
            _txn("summarize"),
        ],
        "summarize": [_txn("finalize")],
        "finalize": [_end_turn()],
    }
    backend = RouterBackend(scripts)

    # Patch asyncio.sleep to be instant for tests
    import asyncio as _a
    original_sleep = _a.sleep

    async def fast_sleep(seconds):
        pass  # noop

    _a.sleep = fast_sleep
    try:
        result = _run(
            orchestrate(
                config=_cfg("monitor_training", "mt-loop"),
                backend=backend,
                workspace_root=workspace_root,
                session_id="s",
                sink=io.StringIO(),
                hitl_prompter=AutoApprovePrompter(),
            )
        )
    finally:
        _a.sleep = original_sleep

    assert result.is_error is False
    assert result.reason == "completed"
    # monitor_training entered 3 times (2 waits, one final transition)
    assert backend.counters["monitor_training"] == 3

    # wakeup_state.json shows accumulated counts
    ws_state = json.loads(
        (workspace_root / "mt-loop" / "workspace" / "logs" / "wakeup_state.json").read_text()
    )
    assert ws_state["waits"]["monitor_training"]["count"] == 2
    assert ws_state["waits"]["monitor_training"]["total_seconds"] == 60


def test_wakeup_recorded_in_state_log(tmp_path: Path) -> None:
    workspace_root = tmp_path / "runs"
    _seed(workspace_root, "mt-note")
    scripts = {
        "monitor_training": [
            _wakeup(30, "polling squeue"),
            _txn("summarize"),
        ],
        "summarize": [_txn("finalize")],
        "finalize": [_end_turn()],
    }
    backend = RouterBackend(scripts)

    import asyncio as _a
    original_sleep = _a.sleep

    async def fast_sleep(seconds):
        pass

    _a.sleep = fast_sleep
    try:
        _run(
            orchestrate(
                config=_cfg("monitor_training", "mt-note"),
                backend=backend,
                workspace_root=workspace_root,
                session_id="s",
                sink=io.StringIO(),
                hitl_prompter=AutoApprovePrompter(),
            )
        )
    finally:
        _a.sleep = original_sleep

    body = (workspace_root / "mt-note" / "workspace" / "logs" / "state_log.md").read_text()
    assert "wakeup at monitor_training" in body
    assert "polling squeue" in body
    assert "resumed at monitor_training" in body


# ── cap enforcement ──────────────────────────────────────────────────────

def test_wait_cap_exceeded_crashes(tmp_path: Path) -> None:
    """When cumulative wait passes MAX_TOTAL_WAIT_PER_STATE_SEC, orchestrator halts."""
    workspace_root = tmp_path / "runs"
    _seed(workspace_root, "mt-cap")

    # Pre-seed wakeup_state near the cap
    ws = workspace_root / "mt-cap" / "workspace" / "logs"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "wakeup_state.json").write_text(
        json.dumps({
            "waits": {
                "monitor_training": {
                    "count": 10,
                    "total_seconds": MAX_TOTAL_WAIT_PER_STATE_SEC,
                }
            }
        })
    )

    scripts = {
        "monitor_training": [_wakeup(60, "one more poll")],
    }
    backend = RouterBackend(scripts)

    import asyncio as _a
    original_sleep = _a.sleep

    async def fast_sleep(seconds):
        pass

    _a.sleep = fast_sleep
    try:
        result = _run(
            orchestrate(
                config=_cfg("monitor_training", "mt-cap"),
                backend=backend,
                workspace_root=workspace_root,
                session_id="s",
                sink=io.StringIO(),
                hitl_prompter=AutoApprovePrompter(),
            )
        )
    finally:
        _a.sleep = original_sleep

    assert result.is_error is True
    assert result.reason == "max_wait_exceeded"
    meta = json.loads((workspace_root / "mt-cap" / "meta.json").read_text())
    assert meta["status"] == "crashed"


# ── bounds ────────────────────────────────────────────────────────────────

def test_wakeup_clamps_below_min(tmp_path: Path) -> None:
    """delaySeconds < 30 → clamped up to 30."""
    workspace_root = tmp_path / "runs"
    _seed(workspace_root, "mt-clamp")
    scripts = {
        "monitor_training": [
            _wakeup(5, "want short"),  # clamped to 30
            _txn("summarize"),
        ],
        "summarize": [_txn("finalize")],
        "finalize": [_end_turn()],
    }
    backend = RouterBackend(scripts)

    import asyncio as _a
    original_sleep = _a.sleep
    slept: list[float] = []

    async def track_sleep(seconds):
        slept.append(seconds)

    _a.sleep = track_sleep
    try:
        _run(
            orchestrate(
                config=_cfg("monitor_training", "mt-clamp"),
                backend=backend,
                workspace_root=workspace_root,
                session_id="s",
                sink=io.StringIO(),
                hitl_prompter=AutoApprovePrompter(),
            )
        )
    finally:
        _a.sleep = original_sleep

    assert slept == [30]


def test_wakeup_clamps_above_max(tmp_path: Path) -> None:
    """delaySeconds > 3600 → clamped down to 3600."""
    workspace_root = tmp_path / "runs"
    _seed(workspace_root, "mt-clamp-hi")
    scripts = {
        "monitor_training": [
            _wakeup(999999, "want huge"),
            _txn("summarize"),
        ],
        "summarize": [_txn("finalize")],
        "finalize": [_end_turn()],
    }
    backend = RouterBackend(scripts)

    import asyncio as _a
    original_sleep = _a.sleep
    slept: list[float] = []

    async def track_sleep(seconds):
        slept.append(seconds)

    _a.sleep = track_sleep
    try:
        _run(
            orchestrate(
                config=_cfg("monitor_training", "mt-clamp-hi"),
                backend=backend,
                workspace_root=workspace_root,
                session_id="s",
                sink=io.StringIO(),
                hitl_prompter=AutoApprovePrompter(),
            )
        )
    finally:
        _a.sleep = original_sleep

    assert slept == [3600]
