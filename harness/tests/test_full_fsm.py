"""Scripted full-FSM end-to-end walk — proves the runtime can carry every
state in the real states/ tree.

Uses a workspace pre-seeded with every canonical deliverable, then scripts
each state's backend response as "transition to the pre-planned target".
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
from harness.fsm import FSM
from harness.hitl import AutoApprovePrompter
from harness.orchestrator import orchestrate


REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Canonical deliverables for EVERY state, seeded upfront ────────────────

_DELIVERABLES: dict[str, str] = {
    # intake
    "intake/training_intent.md":       "goal: train\nalgorithm: grpo\nmodel: Qwen/Qwen2.5-3B\n",
    "intake/verl_root.txt":            "/opt/verl",
    # locate_recipe
    "recipe/recipe.md":                "launch: /opt/verl/examples/grpo_trainer/run.sh\n",
    # configure_algorithm
    "algorithm/algorithm_config.md":   "estimator: gspo\n",
    "algorithm/algorithm_unsupported.md": "reason: n/a\n",
    # prepare_data / generate_preprocess
    "dataset/dataset.md":              "train_files: [.../train.parquet]\n",
    "dataset/intent.md":               "hf_dataset_id: myorg/foo\nschema: [prompt, answer]\n",
    # configure_reward
    "reward/reward_config.md":         "reward_kind: rule\n",
    # select_compute
    "compute/compute_choice.md":       "target: local-direct\n",
    # provision_env
    "env/env_state.md":                "torch: 2.4.0\n",
    "env/launch_env.sh":               "#!/bin/bash\n",
    "env/env_failed.md":               "n/a\n",
    # sanity_rollout
    "sanity/sanity_report.md":         "verdict: green\n",
    # launch_training
    "job/job_info.md":                 "pid: 12345\n",
    # monitor_training
    "job/job_status.md":               "success\n",
    "logs/job_log.md":                 "training complete\n",
    # summarize
    "summary/summary.md":              "topline: success\n",
    # reflect
    "reflect/refinement_plan.md":      "delta: {}\n",
    "reflect/reflect_report.md":       "history: []\n",
    # run_generate / run_eval
    "generate/generate_report.md":     "rows: 100\n",
    "generate/generate_failed.md":     "n/a\n",
    "eval/eval_report.md":             "acc: 0.5\n",
    "eval/eval_failed.md":             "n/a\n",
    # finalize
    "final_report.md":                 "run complete\n",
}


def _seed_all(workspace_root: Path, run_id: str) -> None:
    ws = workspace_root / run_id / "workspace"
    for rel, body in _DELIVERABLES.items():
        p = ws / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)


# ── Scripted plan: straight-through train path (no reflect loop) ──────────

_STRAIGHT_TRAIN_PLAN: list[tuple[str, str]] = [
    ("intake", "locate_recipe"),
    ("locate_recipe", "configure_algorithm"),
    ("configure_algorithm", "prepare_data"),
    ("prepare_data", "configure_reward"),
    ("configure_reward", "select_compute"),
    ("select_compute", "provision_env"),
    ("provision_env", "sanity_rollout"),
    ("sanity_rollout", "launch_training"),
    ("launch_training", "monitor_training"),
    ("monitor_training", "summarize"),
    ("summarize", "finalize"),
]


class ScriptedPlanBackend(Backend):
    """For each (source, target) in the plan, emit a transition_to(target) call.

    Each state may be entered multiple times (in reflect-loop paths), so we
    keep a per-state cursor.
    """

    model_id = "scripted-plan"

    def __init__(self, plan: list[tuple[str, str]]) -> None:
        # source_state → deque of targets in the order they should fire
        from collections import defaultdict, deque
        self.queues: dict[str, "deque[str]"] = defaultdict(deque)
        for src, dst in plan:
            self.queues[src].append(dst)

    async def stream(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int = 4096,
    ) -> AsyncIterator[RawEvent]:
        state = _detect_state(system)
        q = self.queues.get(state)
        if not q:
            # Terminal state (finalize) — end_turn with no transition
            yield RawEvent(kind="text_delta", text="terminal reached.")
            yield RawEvent(kind="message_stop", stop_reason="end_turn")
            return
        target = q.popleft()
        yield RawEvent(
            kind="tool_use",
            tool_use={
                "id": f"c-{state}-{target}",
                "name": "transition_to",
                "input": {"next_state": target},
            },
        )
        yield RawEvent(kind="message_stop", stop_reason="tool_use")


def _detect_state(system) -> str:
    if isinstance(system, list):
        system = "\n".join(b.get("text", "") for b in system if isinstance(b, dict))
    marker = "## states/"
    i = system.find(marker)
    tail = system[i + len(marker) :]
    return tail[: tail.find(".md")]


def _cfg(run_id: str) -> RunConfig:
    return RunConfig(
        workdir=REPO_ROOT,
        model="scripted/plan",
        goal="full FSM E2E",
        state="intake",
        run_id=run_id,
        hitl=False,
        verl_root=None,
    )


def _run(coro):
    return asyncio.run(coro)


# ── happy path: straight train, 12 states, ending at finalize ─────────────

def test_straight_train_walk_covers_12_states(tmp_path: Path) -> None:
    workspace_root = tmp_path / "runs"
    _seed_all(workspace_root, "full-e2e")

    backend = ScriptedPlanBackend(_STRAIGHT_TRAIN_PLAN)
    sink = io.StringIO()
    result = _run(
        orchestrate(
            config=_cfg("full-e2e"),
            backend=backend,
            workspace_root=workspace_root,
            session_id="s-e2e",
            sink=sink,
            hitl_prompter=AutoApprovePrompter(),
            max_states=25,
        )
    )
    assert result.is_error is False, result.message
    assert result.reason == "completed"
    assert result.final_state == "finalize"

    expected = [src for src, _ in _STRAIGHT_TRAIN_PLAN] + ["finalize"]
    assert result.visited == expected

    # meta.json says completed with step count matching visited length
    meta = json.loads((workspace_root / "full-e2e" / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["step"] == len(expected)

    # state_log entries match
    body = (workspace_root / "full-e2e" / "workspace" / "logs" / "state_log.md").read_text()
    entered = [ln for ln in body.splitlines() if "entered" in ln and ln.startswith("- [")]
    assert len(entered) == len(expected)

    # Each state emitted its own system.init
    events = [json.loads(ln) for ln in sink.getvalue().splitlines() if ln]
    inits = [e for e in events if e.get("type") == "system"]
    assert len(inits) == len(expected)


# ── reflect-loop path: full walk WITH loop iterations ─────────────────────

_REFLECT_LOOP_PLAN: list[tuple[str, str]] = [
    # First iteration
    ("intake", "locate_recipe"),
    ("locate_recipe", "configure_algorithm"),
    ("configure_algorithm", "prepare_data"),
    ("prepare_data", "configure_reward"),
    ("configure_reward", "select_compute"),
    ("select_compute", "provision_env"),
    ("provision_env", "sanity_rollout"),
    ("sanity_rollout", "launch_training"),
    ("launch_training", "monitor_training"),
    ("monitor_training", "summarize"),
    ("summarize", "reflect"),
    ("reflect", "configure_algorithm"),  # loop back (iteration 1)
    # Second iteration
    ("configure_algorithm", "prepare_data"),
    ("prepare_data", "configure_reward"),
    ("configure_reward", "select_compute"),
    ("select_compute", "provision_env"),
    ("provision_env", "sanity_rollout"),
    ("sanity_rollout", "launch_training"),
    ("launch_training", "monitor_training"),
    ("monitor_training", "summarize"),
    ("summarize", "reflect"),
    ("reflect", "finalize"),  # exit the loop
]


def test_reflect_loop_walk_terminates_at_finalize(tmp_path: Path) -> None:
    workspace_root = tmp_path / "runs"
    _seed_all(workspace_root, "full-loop")

    backend = ScriptedPlanBackend(_REFLECT_LOOP_PLAN)
    result = _run(
        orchestrate(
            config=_cfg("full-loop"),
            backend=backend,
            workspace_root=workspace_root,
            session_id="s-loop",
            sink=io.StringIO(),
            hitl_prompter=AutoApprovePrompter(),
            max_states=50,
        )
    )
    assert result.is_error is False, result.message
    assert result.final_state == "finalize"

    # configure_algorithm should have been entered twice (loop iteration)
    assert result.visited.count("configure_algorithm") == 2
    assert result.visited.count("reflect") == 2

    # Loop counter is 1 (only reflect→configure_algorithm fires once; the exit branch is to finalize)
    ls = json.loads(
        (workspace_root / "full-loop" / "workspace" / "reflect" / "loop_state.json").read_text()
    )
    assert ls["edges"]["reflect->configure_algorithm"] == 1


# ── FSM sanity: prove every state has a plan-covered response path ────────

def test_fsm_all_states_reachable_in_scripted_plan() -> None:
    """Sanity: the plans above cover every non-terminal state at least once."""
    fsm = FSM.load(REPO_ROOT)
    covered = {src for src, _ in _STRAIGHT_TRAIN_PLAN}
    covered |= {src for src, _ in _REFLECT_LOOP_PLAN}
    # Add finalize (terminal) — covered by end_turn in ScriptedPlanBackend
    covered.add("finalize")
    # states we don't touch in the training path: run_generate, run_eval,
    # generate_preprocess. Those are goal=generate/eval + dataset-bounce paths
    # and are exercised by their own path tests below.
    for name in fsm.states:
        if name in {"run_generate", "run_eval", "generate_preprocess"}:
            continue
        assert name in covered, f"state {name} not covered by any plan"


def test_generate_track(tmp_path: Path) -> None:
    """goal=generate → intake → run_generate → finalize."""
    workspace_root = tmp_path / "runs"
    _seed_all(workspace_root, "gen")

    plan = [
        ("intake", "run_generate"),
        ("run_generate", "finalize"),
    ]
    backend = ScriptedPlanBackend(plan)
    result = _run(
        orchestrate(
            config=_cfg("gen"),
            backend=backend,
            workspace_root=workspace_root,
            session_id="s",
            sink=io.StringIO(),
            hitl_prompter=AutoApprovePrompter(),
        )
    )
    assert result.is_error is False
    assert result.final_state == "finalize"
    assert result.visited == ["intake", "run_generate", "finalize"]


def test_eval_track(tmp_path: Path) -> None:
    """goal=eval → intake → run_eval → finalize."""
    workspace_root = tmp_path / "runs"
    _seed_all(workspace_root, "ev")

    plan = [
        ("intake", "run_eval"),
        ("run_eval", "finalize"),
    ]
    result = _run(
        orchestrate(
            config=_cfg("ev"),
            backend=ScriptedPlanBackend(plan),
            workspace_root=workspace_root,
            session_id="s",
            sink=io.StringIO(),
            hitl_prompter=AutoApprovePrompter(),
        )
    )
    assert result.is_error is False
    assert result.visited == ["intake", "run_eval", "finalize"]


def test_dataset_bounce(tmp_path: Path) -> None:
    """prepare_data → generate_preprocess → prepare_data (declared loop=1) → configure_reward."""
    workspace_root = tmp_path / "runs"
    _seed_all(workspace_root, "bounce")

    plan = [
        ("intake", "locate_recipe"),
        ("locate_recipe", "configure_algorithm"),
        ("configure_algorithm", "prepare_data"),
        ("prepare_data", "generate_preprocess"),
        ("generate_preprocess", "prepare_data"),  # bounce back
        ("prepare_data", "configure_reward"),
        ("configure_reward", "select_compute"),
        ("select_compute", "provision_env"),
        ("provision_env", "sanity_rollout"),
        ("sanity_rollout", "launch_training"),
        ("launch_training", "monitor_training"),
        ("monitor_training", "summarize"),
        ("summarize", "finalize"),
    ]
    result = _run(
        orchestrate(
            config=_cfg("bounce"),
            backend=ScriptedPlanBackend(plan),
            workspace_root=workspace_root,
            session_id="s",
            sink=io.StringIO(),
            hitl_prompter=AutoApprovePrompter(),
            max_states=30,
        )
    )
    assert result.is_error is False
    assert result.final_state == "finalize"
    assert result.visited.count("prepare_data") == 2
    assert result.visited.count("generate_preprocess") == 1
