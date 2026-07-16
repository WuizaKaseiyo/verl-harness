"""Resume + interrupt tests."""

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
from harness.orchestrator import orchestrate
from harness.resume import ResumeError, flip_status_to_running, load_resume_plan
from harness.runlog import RunLog, new_meta


REPO_ROOT = Path(__file__).resolve().parents[2]


class ScriptedRouter(Backend):
    """Same helper used in test_orchestrator, kept local so this file is standalone."""

    model_id = "scripted-resume"

    def __init__(self, scripts: dict[str, list[list[RawEvent]]]) -> None:
        self.scripts = scripts
        self.counters = {n: 0 for n in scripts}

    async def stream(self, *, system, messages, tools, max_tokens=4096) -> AsyncIterator[RawEvent]:
        state = _detect_state(system)
        script = self.scripts[state]
        idx = self.counters[state]
        self.counters[state] = idx + 1
        for e in script[idx]:
            yield e


def _detect_state(system) -> str:
    if isinstance(system, list):
        system = "\n".join(b.get("text", "") for b in system if isinstance(b, dict))
    marker = "## states/"
    i = system.find(marker)
    tail = system[i + len(marker) :]
    return tail[: tail.find(".md")]


def _txn(target: str, tid: str = "c1") -> list[RawEvent]:
    return [
        RawEvent(
            kind="tool_use",
            tool_use={"id": tid, "name": "transition_to", "input": {"next_state": target}},
        ),
        RawEvent(kind="message_stop", stop_reason="tool_use"),
    ]


def _end_turn(text: str = "done") -> list[RawEvent]:
    return [
        RawEvent(kind="text_delta", text=text),
        RawEvent(kind="message_stop", stop_reason="end_turn"),
    ]


def _seed(workspace_root: Path, run_id: str) -> None:
    ws = workspace_root / run_id / "workspace"
    files = {
        "intake/verl_root.txt": "/opt/verl",
        "intake/training_intent.md": "goal: train\n",
        "recipe/recipe.md": "launch: /opt/verl/examples/grpo_trainer/run.sh\n",
        "algorithm/algorithm_config.md": "estimator: gspo\n",
        "algorithm/algorithm_unsupported.md": "reason: n/a\n",
    }
    for rel, body in files.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)


def _cfg(workspace_root: Path, state: str = "intake", run_id: str = "resume-test") -> RunConfig:
    return RunConfig(
        workdir=REPO_ROOT,
        model="scripted/x",
        goal="resume test",
        state=state,
        run_id=run_id,
        hitl=False,
        verl_root=None,
    )


def _run(coro):
    return asyncio.run(coro)


# ── load_resume_plan ──────────────────────────────────────────────────────

def test_load_resume_plan_reads_last_entered(tmp_path: Path) -> None:
    workspace_root = tmp_path / "runs"
    rl = RunLog(runs_root=workspace_root, run_id="R")
    rl.init(
        new_meta(
            run_id="R",
            goal="g",
            model="m",
            session_id="s",
            hitl=False,
            verl_root=None,
        )
    )
    rl.enter_state("intake")
    rl.enter_state("locate_recipe", previous="intake")
    rl.mark_terminal(status="cancelled", note="test")

    plan = load_resume_plan(workspace_root, "R")
    assert plan.last_state == "locate_recipe"
    assert plan.meta.goal == "g"


def test_load_resume_plan_missing_run(tmp_path: Path) -> None:
    with pytest.raises(ResumeError, match="does not exist"):
        load_resume_plan(tmp_path / "runs", "never-existed")


def test_load_resume_plan_no_state_log(tmp_path: Path) -> None:
    """meta.json exists but state_log.md is empty — no `entered` line to resume from."""
    workspace_root = tmp_path / "runs"
    rl = RunLog(runs_root=workspace_root, run_id="R")
    rl.init(
        new_meta(
            run_id="R",
            goal="g",
            model="m",
            session_id="s",
            hitl=False,
            verl_root=None,
        )
    )
    # No enter_state calls
    with pytest.raises(ResumeError, match="no `entered` lines"):
        load_resume_plan(workspace_root, "R")


# ── flip_status_to_running ────────────────────────────────────────────────

def test_flip_status_to_running(tmp_path: Path) -> None:
    workspace_root = tmp_path / "runs"
    rl = RunLog(runs_root=workspace_root, run_id="R")
    rl.init(
        new_meta(
            run_id="R",
            goal="g",
            model="m",
            session_id="s",
            hitl=False,
            verl_root=None,
        )
    )
    rl.enter_state("intake")
    rl.mark_terminal(status="cancelled", note="user Ctrl-C")
    assert rl.read_meta().status == "cancelled"

    flip_status_to_running(workspace_root, "R", note="test resume")
    meta = rl.read_meta()
    assert meta.status == "running"
    assert meta.completed_at is None
    log = rl.state_log_path.read_text()
    assert "resumed" in log


# ── Interrupt: KeyboardInterrupt marks status=cancelled ───────────────────

class RaisingBackend(Backend):
    model_id = "raising"

    async def stream(self, *, system, messages, tools, max_tokens=4096):
        raise KeyboardInterrupt("user Ctrl-C")
        yield  # unreachable — pragma: no cover


def test_keyboard_interrupt_marks_cancelled(tmp_path: Path) -> None:
    workspace_root = tmp_path / "runs"
    _seed(workspace_root, "resume-test")
    result = _run(
        orchestrate(
            config=_cfg(workspace_root),
            backend=RaisingBackend(),
            workspace_root=workspace_root,
            session_id="s",
            sink=io.StringIO(),
            hitl_prompter=AutoApprovePrompter(),
        )
    )
    assert result.is_error is True
    assert result.reason == "cancelled"

    meta = json.loads((workspace_root / "resume-test" / "meta.json").read_text())
    assert meta["status"] == "cancelled"


# ── Full round-trip: interrupt → resume → finish ──────────────────────────

def test_interrupt_then_resume_finishes(tmp_path: Path) -> None:
    workspace_root = tmp_path / "runs"
    _seed(workspace_root, "resume-test")

    # Run 1: intake succeeds, locate_recipe interrupts
    run1_scripts = {
        "intake": [_txn("locate_recipe")],
        # locate_recipe crashes mid-turn
    }

    class InterruptOnce(Backend):
        model_id = "interrupt-once"
        turns_by_state = {"intake": 0, "locate_recipe": 0}

        async def stream(self, *, system, messages, tools, max_tokens=4096):
            state = _detect_state(system)
            self.turns_by_state[state] = self.turns_by_state.get(state, 0) + 1
            if state == "intake":
                for e in _txn("locate_recipe"):
                    yield e
            elif state == "locate_recipe":
                raise KeyboardInterrupt()
            else:
                raise AssertionError(f"unexpected state {state}")

    _run(
        orchestrate(
            config=_cfg(workspace_root),
            backend=InterruptOnce(),
            workspace_root=workspace_root,
            session_id="s1",
            sink=io.StringIO(),
            hitl_prompter=AutoApprovePrompter(),
        )
    )
    meta = json.loads((workspace_root / "resume-test" / "meta.json").read_text())
    assert meta["status"] == "cancelled"
    assert meta["current_state"] == "locate_recipe"

    # Prepare for resume
    plan = load_resume_plan(workspace_root, "resume-test")
    assert plan.last_state == "locate_recipe"
    flip_status_to_running(workspace_root, "resume-test", note="resuming for test")

    # Run 2: resume from locate_recipe → configure_algorithm → finalize
    scripts = {
        "locate_recipe": [_txn("configure_algorithm")],
        "configure_algorithm": [_txn("finalize")],
        "finalize": [_end_turn()],
    }
    backend = ScriptedRouter(scripts)
    result = _run(
        orchestrate(
            config=_cfg(workspace_root, state=plan.last_state),
            backend=backend,
            workspace_root=workspace_root,
            session_id="s2",
            sink=io.StringIO(),
            hitl_prompter=AutoApprovePrompter(),
            resumed=True,
        )
    )
    assert result.is_error is False
    assert result.final_state == "finalize"

    meta = json.loads((workspace_root / "resume-test" / "meta.json").read_text())
    assert meta["status"] == "completed"

    # state_log has resumed + additional entered lines
    body = (workspace_root / "resume-test" / "workspace" / "logs" / "state_log.md").read_text()
    assert "resumed" in body
    assert body.count("entered locate_recipe") == 2  # once per run
