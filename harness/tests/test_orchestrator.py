"""Multi-state orchestrator tests using scripted backends.

Covers:
  - Chain intake → locate_recipe → configure_algorithm (early stop via scripted error)
  - Chain through a terminal state (finalize) as SUCCESS
  - meta.json goes running → completed
  - state_log.md gains one entry per state entered
  - Each state emits its own `system.init`
  - Global max_states cap fires on runaway
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
from harness.orchestrator import orchestrate


REPO_ROOT = Path(__file__).resolve().parents[2]


class ScriptedRouter(Backend):
    """A scripted backend that chooses different responses based on the current STATE.

    We inspect the system prompt (which embeds `## states/<name>.md`) to figure
    out which state we're being asked to drive, then look up the pre-registered
    script for that state.
    """

    model_id = "scripted-router"

    def __init__(self, scripts: dict[str, list[list[RawEvent]]]) -> None:
        # scripts[state_name] is a list of turns for that state (in order).
        self.scripts = scripts
        self.turn_counters: dict[str, int] = {name: 0 for name in scripts}

    async def stream(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> AsyncIterator[RawEvent]:
        # Detect the state from the system prompt's `## states/<name>.md` header.
        state = _detect_state(system)
        script = self.scripts.get(state)
        if script is None:
            raise AssertionError(f"no scripted turns for state {state!r}")
        idx = self.turn_counters[state]
        if idx >= len(script):
            raise AssertionError(
                f"state {state!r} exhausted its script at turn {idx + 1}"
            )
        self.turn_counters[state] = idx + 1
        for e in script[idx]:
            yield e


def _detect_state(system) -> str:
    """Find `## states/<name>.md` in the assembled system prompt (str or list-of-blocks)."""
    if isinstance(system, list):
        system = "\n".join(b.get("text", "") for b in system if isinstance(b, dict))
    marker = "## states/"
    i = system.find(marker)
    if i == -1:
        raise AssertionError("system prompt does not embed a state file")
    tail = system[i + len(marker) :]
    end = tail.find(".md")
    return tail[:end]


def _transition_event(next_state: str, tool_id: str = "c1") -> list[RawEvent]:
    return [
        RawEvent(
            kind="tool_use",
            tool_use={
                "id": tool_id,
                "name": "transition_to",
                "input": {"next_state": next_state},
            },
        ),
        RawEvent(kind="message_stop", stop_reason="tool_use"),
    ]


def _end_turn(text: str) -> list[RawEvent]:
    return [
        RawEvent(kind="text_delta", text=text),
        RawEvent(kind="message_stop", stop_reason="end_turn"),
    ]


def _seed_deliverables(workspace_root: Path, run_id: str) -> None:
    """Pre-populate the workspace with the canonical/cited files each transition
    in the walk needs so contract enforcement waves them through.

    Test intent here is orchestrator mechanics — contract enforcement itself
    is covered in test_state_driver / test_contracts.
    """
    ws = workspace_root / run_id / "workspace"
    files = {
        # intake → locate_recipe cites verl_root.txt
        "intake/verl_root.txt": "/opt/verl",
        # intake also produces training_intent.md (per state description)
        "intake/training_intent.md": "goal: train\nalgorithm: grpo\n",
        # locate_recipe → configure_algorithm canonical
        "recipe/recipe.md": "launch: /opt/verl/examples/grpo_trainer/run.sh\n",
        # configure_algorithm → finalize cites algorithm_unsupported.md;
        # configure_algorithm → prepare_data canonical algorithm_config.md.
        # Cover both branches.
        "algorithm/algorithm_config.md": "estimator: gspo\n",
        "algorithm/algorithm_unsupported.md": "reason: no trainer\n",
    }
    for rel, body in files.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)


def _config(tmp_path: Path, state: str = "intake") -> RunConfig:
    return RunConfig(
        workdir=REPO_ROOT,
        model="scripted/scripted",
        goal="orchestrator test",
        state=state,
        run_id="orch-test",
        hitl=False,
        verl_root=None,
    )


def _run(coro):
    return asyncio.run(coro)


# ── happy path: intake → locate_recipe → configure_algorithm, then bail  ──

def test_three_state_walk(tmp_path: Path) -> None:
    """Chain 3 states, then have configure_algorithm route to finalize as terminal."""
    workspace_root = tmp_path / "runs"
    _seed_deliverables(workspace_root, "orch-test")
    scripts = {
        "intake": [_transition_event("locate_recipe")],
        "locate_recipe": [_transition_event("configure_algorithm")],
        # configure_algorithm has two branches; pick the terminal one
        "configure_algorithm": [_transition_event("finalize")],
        "finalize": [_end_turn("done")],
    }
    backend = ScriptedRouter(scripts)
    sink = io.StringIO()
    result = _run(
        orchestrate(
            config=_config(tmp_path),
            backend=backend,
            workspace_root=workspace_root,
            session_id="sess-orch",
            sink=sink,
            hitl_prompter=AutoApprovePrompter(),
        )
    )
    assert result.is_error is False
    assert result.reason == "completed"
    assert result.visited == ["intake", "locate_recipe", "configure_algorithm", "finalize"]
    assert result.final_state == "finalize"

    # meta.json goes running → completed
    meta_path = workspace_root / "orch-test" / "meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["status"] == "completed"
    assert meta["step"] == 4
    assert meta["current_state"] == "finalize"

    # state_log.md has 4 machine-readable lines
    state_log = (
        workspace_root / "orch-test" / "workspace" / "logs" / "state_log.md"
    ).read_text()
    entered_lines = [
        ln for ln in state_log.splitlines() if ln.startswith("- [") and "entered" in ln
    ]
    assert len(entered_lines) == 4

    # Each state emitted a system.init event on the stream
    events = [json.loads(ln) for ln in sink.getvalue().splitlines() if ln]
    inits = [e for e in events if e.get("type") == "system" and e.get("subtype") == "init"]
    assert len(inits) == 4


# ── failure: a state's driver errors out mid-run  ─────────────────────────

def test_state_driver_error_halts_and_marks_crashed(tmp_path: Path) -> None:
    workspace_root = tmp_path / "runs"
    _seed_deliverables(workspace_root, "orch-test")
    scripts = {
        "intake": [_transition_event("locate_recipe")],
        # locate_recipe ends without transitioning → driver marks this as error
        "locate_recipe": [_end_turn("gave up")],
    }
    backend = ScriptedRouter(scripts)
    result = _run(
        orchestrate(
            config=_config(tmp_path),
            backend=backend,
            workspace_root=workspace_root,
            session_id="sess-crash",
            sink=io.StringIO(),
            hitl_prompter=AutoApprovePrompter(),
        )
    )
    assert result.is_error is True
    assert result.reason == "crashed"
    assert result.final_state == "locate_recipe"
    assert result.visited == ["intake", "locate_recipe"]

    meta = json.loads((workspace_root / "orch-test" / "meta.json").read_text())
    assert meta["status"] == "crashed"
    assert meta["current_state"] == "locate_recipe"


# ── max_states cap  ───────────────────────────────────────────────────────

def test_max_states_cap_fires(tmp_path: Path) -> None:
    """Fake a cycle by having states transition to each other back-and-forth …

    We can't easily do that with the real spec (validator would reject), so
    instead: keep one state route to the same next state, and set max_states=2.
    The orchestrator hits the cap on step 3.
    """
    workspace_root = tmp_path / "runs"
    _seed_deliverables(workspace_root, "orch-test")
    scripts = {
        "intake": [
            _transition_event("locate_recipe"),
            _transition_event("locate_recipe"),
            _transition_event("locate_recipe"),
        ],
        "locate_recipe": [
            _transition_event("configure_algorithm"),
            _transition_event("configure_algorithm"),
            _transition_event("configure_algorithm"),
        ],
        "configure_algorithm": [_transition_event("finalize")],
    }
    backend = ScriptedRouter(scripts)
    result = _run(
        orchestrate(
            config=_config(tmp_path),
            backend=backend,
            workspace_root=workspace_root,
            session_id="sess-cap",
            sink=io.StringIO(),
            max_states=2,
            hitl_prompter=AutoApprovePrompter(),
        )
    )
    assert result.is_error is True
    assert result.reason == "max_states"


# ── FSM load failure surfaces cleanly  ─────────────────────────────────────

def test_fsm_load_error(tmp_path: Path) -> None:
    """Point at a workdir that isn't a real verl-harness → orchestrator marks crashed."""
    (tmp_path / "states").mkdir()  # empty — no intake.md, no drain-to-terminal
    workspace_root = tmp_path / "runs"

    class NeverBackend(Backend):
        model_id = "n"

        async def stream(self, **kw):
            raise AssertionError("should not be called")
            yield  # unreachable — pragma: no cover

    result = _run(
        orchestrate(
            config=RunConfig(
                workdir=tmp_path,
                model="scripted/x",
                goal="g",
                state="intake",
                run_id="fsm-fail",
                hitl=False,
                verl_root=None,
            ),
            backend=NeverBackend(),
            workspace_root=workspace_root,
            session_id="s",
            sink=io.StringIO(),
        )
    )
    assert result.is_error
    assert result.reason == "fsm_error"

    meta = json.loads((workspace_root / "fsm-fail" / "meta.json").read_text())
    assert meta["status"] == "crashed"
