"""HITL layer — parsing, classification, prompter behaviour."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from harness.hitl import (
    AutoApprovePrompter,
    AutoDenyPrompter,
    EventPrompter,
    HandOffPoint,
    PromptDecision,
    StdioPrompter,
    condition_met,
    evaluate_hand_offs,
    parse_handoff_points,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


# ── parsing ────────────────────────────────────────────────────────────────

def test_parse_intake_ordinary() -> None:
    md = (REPO_ROOT / "states" / "intake.md").read_text()
    hops = parse_handoff_points(md, "intake")
    assert len(hops) == 1
    hop = hops[0]
    assert hop.title.startswith("Confirm normalised intent")
    assert hop.is_always_on is False


def test_parse_generate_preprocess_always_on() -> None:
    md = (REPO_ROOT / "states" / "generate_preprocess.md").read_text()
    hops = parse_handoff_points(md, "generate_preprocess")
    assert len(hops) == 1
    assert hops[0].is_always_on is True
    assert "Always-on" in hops[0].description


def test_parse_configure_reward_conditional_always_on() -> None:
    md = (REPO_ROOT / "states" / "configure_reward.md").read_text()
    hops = parse_handoff_points(md, "configure_reward")
    always_on_hops = [h for h in hops if h.is_always_on]
    assert always_on_hops, "configure_reward has an Always-on for custom/shaped bullet"
    # The condition is exposed for runtime evaluation
    assert any(
        h.condition and "reward_kind" in h.condition and "custom" in h.condition
        for h in always_on_hops
    )


def test_parse_sanity_rollout_always_on() -> None:
    md = (REPO_ROOT / "states" / "sanity_rollout.md").read_text()
    hops = parse_handoff_points(md, "sanity_rollout")
    assert any(h.is_always_on for h in hops)


def test_parse_launch_training_threshold_gate() -> None:
    md = (REPO_ROOT / "states" / "launch_training.md").read_text()
    hops = parse_handoff_points(md, "launch_training")
    assert any(h.is_always_on for h in hops), (
        "launch_training cost gate is Threshold-based → treated as always-on"
    )


def test_parse_empty_section() -> None:
    assert parse_handoff_points("## Description\n\nno hand-offs here.\n", "x") == []


def test_parse_human_checkpoints_backcompat() -> None:
    md = (
        "## Human Checkpoints\n\n"
        "- **Confirm plan.** normal prompt. Skipped with `--no-hitl`.\n"
    )
    hops = parse_handoff_points(md, "x")
    assert len(hops) == 1
    assert hops[0].title == "Confirm plan"
    assert hops[0].is_always_on is False


# ── stdio prompter ────────────────────────────────────────────────────────

def _hop(title: str, always_on: bool = False) -> HandOffPoint:
    return HandOffPoint(state="s", title=title, description="d", is_always_on=always_on)


def test_stdio_prompter_yes() -> None:
    err = io.StringIO()
    inp = io.StringIO("y\n")
    p = StdioPrompter(err=err, inp=inp)
    assert p.ask(_hop("go")) == PromptDecision.APPROVE
    assert "hand-off" in err.getvalue()


def test_stdio_prompter_no() -> None:
    p = StdioPrompter(err=io.StringIO(), inp=io.StringIO("n\n"))
    assert p.ask(_hop("go")) == PromptDecision.DENY


def test_stdio_prompter_labels_always_on_gate() -> None:
    err = io.StringIO()
    p = StdioPrompter(err=err, inp=io.StringIO("y\n"))
    p.ask(_hop("go", always_on=True))
    assert "ALWAYS-ON GATE" in err.getvalue()


def test_stdio_prompter_skip() -> None:
    p = StdioPrompter(err=io.StringIO(), inp=io.StringIO("s\n"))
    assert p.ask(_hop("go")) == PromptDecision.SKIP


# ── evaluate_hand_offs — hitl=True, all prompt ────────────────────────────

def test_hitl_on_all_prompted() -> None:
    md = (REPO_ROOT / "states" / "intake.md").read_text()
    result = evaluate_hand_offs(
        state_name="intake", state_md=md, hitl=True, prompter=AutoApprovePrompter()
    )
    assert result.approved is True
    assert len(result.approved_titles) == 1


def test_hitl_on_deny_marks_denied() -> None:
    md = (REPO_ROOT / "states" / "intake.md").read_text()
    result = evaluate_hand_offs(
        state_name="intake", state_md=md, hitl=True, prompter=AutoDenyPrompter()
    )
    assert result.approved is False
    assert len(result.denied_titles) == 1


# ── evaluate_hand_offs — hitl=False, only always-on prompt ────────────────

def test_no_hitl_skips_ordinary(monkeypatch: pytest.MonkeyPatch) -> None:
    md = (REPO_ROOT / "states" / "intake.md").read_text()

    calls: list[str] = []

    class TrackingPrompter:
        def ask(self, hop):
            calls.append(hop.title)
            return PromptDecision.APPROVE

    result = evaluate_hand_offs(
        state_name="intake", state_md=md, hitl=False, prompter=TrackingPrompter()
    )
    # intake's hand-off is ordinary → should be skipped, not prompted
    assert calls == []
    assert result.approved is True
    assert result.skipped_titles  # at least one


def test_no_hitl_still_prompts_always_on_gate() -> None:
    md = (REPO_ROOT / "states" / "generate_preprocess.md").read_text()

    calls: list[str] = []

    class TrackingPrompter:
        def ask(self, hop):
            calls.append(hop.title)
            return PromptDecision.APPROVE

    result = evaluate_hand_offs(
        state_name="generate_preprocess",
        state_md=md,
        hitl=False,
        prompter=TrackingPrompter(),
    )
    # generate_preprocess has an always-on gate → should be prompted even with --no-hitl
    assert calls, "always-on gate must fire under --no-hitl"
    assert result.approved is True


def test_no_hitl_launch_training_cost_gate_fires() -> None:
    md = (REPO_ROOT / "states" / "launch_training.md").read_text()

    calls: list[str] = []

    class TrackingPrompter:
        def ask(self, hop):
            calls.append(hop.title)
            return PromptDecision.APPROVE

    evaluate_hand_offs(
        state_name="launch_training",
        state_md=md,
        hitl=False,
        prompter=TrackingPrompter(),
    )
    assert any("cost" in t.lower() or "gate" in t.lower() for t in calls)


# ── stderr routing ────────────────────────────────────────────────────────

# ── condition_met ─────────────────────────────────────────────────────────

def _write_intent(workspace: Path, body: str) -> None:
    p = workspace / "intake" / "training_intent.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)


def test_condition_met_set_membership_true(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_intent(ws, "reward_kind: custom\n")
    assert condition_met("reward_kind ∈ {custom, shaped}", ws) is True


def test_condition_met_set_membership_false(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_intent(ws, "reward_kind: rule\n")
    assert condition_met("reward_kind ∈ {custom, shaped}", ws) is False


def test_condition_met_strips_inline_comments(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_intent(ws, "reward_kind: rule  # picked at intake\n")
    assert condition_met("reward_kind ∈ {custom, shaped}", ws) is False


def test_condition_met_missing_key(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_intent(ws, "goal: train\n")
    # Key not present → be conservative and prompt anyway
    assert condition_met("reward_kind ∈ {custom, shaped}", ws) is None


def test_condition_met_no_intent_file(tmp_path: Path) -> None:
    assert condition_met("reward_kind ∈ {custom, shaped}", tmp_path / "empty") is None


def test_condition_met_equality(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_intent(ws, "goal: train\n")
    assert condition_met("goal == train", ws) is True
    assert condition_met("goal = train", ws) is True
    assert condition_met("goal == eval", ws) is False


# ── slash-list form (M3-T7) ────────────────────────────────────────────────

def test_condition_met_slash_list_algo_match(tmp_path: Path) -> None:
    """`dpo/rm` in a when-clause matches when algorithm=dpo."""
    ws = tmp_path / "ws"
    _write_intent(ws, "algorithm: dpo\n")
    assert condition_met("dpo/rm without first-class trainer", ws) is True


def test_condition_met_slash_list_algo_no_match(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_intent(ws, "algorithm: grpo\n")
    assert condition_met("dpo/rm without first-class trainer", ws) is False


def test_condition_met_slash_list_algo_missing(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_intent(ws, "goal: train\n")  # no algorithm key
    assert condition_met("dpo/rm without first-class trainer", ws) is None


def test_parse_configure_algorithm_when_clause_now_recognized() -> None:
    """configure_algorithm's `**Always-on** when ... (dpo/rm ...)` is classified as always-on."""
    md = (REPO_ROOT / "states" / "configure_algorithm.md").read_text()
    hops = parse_handoff_points(md, "configure_algorithm")
    ac_hops = [h for h in hops if h.is_always_on]
    assert ac_hops, "configure_algorithm's dpo/rm halt gate should be always-on"
    assert any(h.condition and "dpo" in h.condition for h in ac_hops)


def test_configure_algorithm_gate_downgraded_for_grpo(tmp_path: Path) -> None:
    """--no-hitl + algorithm=grpo → configure_algorithm's when-clause gate skips."""
    ws = tmp_path / "ws"
    _write_intent(ws, "algorithm: grpo\n")

    md = (REPO_ROOT / "states" / "configure_algorithm.md").read_text()
    calls: list[str] = []

    class Track:
        def ask(self, hop):
            calls.append(hop.title)
            return PromptDecision.APPROVE

    evaluate_hand_offs(
        state_name="configure_algorithm",
        state_md=md,
        hitl=False,
        prompter=Track(),
        workspace=ws,
    )
    # gate should NOT fire for grpo
    assert not calls, f"expected no prompts for grpo, got: {calls}"


def test_configure_algorithm_gate_fires_for_dpo(tmp_path: Path) -> None:
    """--no-hitl + algorithm=dpo → configure_algorithm's when-clause gate FIRES."""
    ws = tmp_path / "ws"
    _write_intent(ws, "algorithm: dpo\n")

    md = (REPO_ROOT / "states" / "configure_algorithm.md").read_text()
    calls: list[str] = []

    class Track:
        def ask(self, hop):
            calls.append(hop.title)
            return PromptDecision.APPROVE

    evaluate_hand_offs(
        state_name="configure_algorithm",
        state_md=md,
        hitl=False,
        prompter=Track(),
        workspace=ws,
    )
    assert calls, "gate should fire when algorithm=dpo"


# ── evaluate_hand_offs with workspace-conditional gates ───────────────────

def test_no_hitl_conditional_gate_skipped_when_condition_false(tmp_path: Path) -> None:
    """The bug that broke the smoke: reward_kind=rule under --no-hitl should NOT prompt."""
    ws = tmp_path / "ws"
    _write_intent(ws, "reward_kind: rule\n")

    md = (REPO_ROOT / "states" / "configure_reward.md").read_text()
    calls: list[str] = []

    class Track:
        def ask(self, hop):
            calls.append(hop.title)
            return PromptDecision.APPROVE

    result = evaluate_hand_offs(
        state_name="configure_reward",
        state_md=md,
        hitl=False,
        prompter=Track(),
        workspace=ws,
    )
    # reward_kind not in {custom, shaped} → gate downgraded → skipped under --no-hitl
    assert not calls, f"gate should not fire when condition false, but fired: {calls}"
    assert result.approved is True


def test_no_hitl_conditional_gate_fires_when_condition_true(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    _write_intent(ws, "reward_kind: custom\n")

    md = (REPO_ROOT / "states" / "configure_reward.md").read_text()
    calls: list[str] = []

    class Track:
        def ask(self, hop):
            calls.append(hop.title)
            return PromptDecision.APPROVE

    evaluate_hand_offs(
        state_name="configure_reward",
        state_md=md,
        hitl=False,
        prompter=Track(),
        workspace=ws,
    )
    assert calls, "gate should fire when reward_kind=custom"


def test_no_hitl_conditional_gate_fires_when_workspace_unavailable(tmp_path: Path) -> None:
    """Missing training_intent.md → runtime can't evaluate → fires safely."""
    ws = tmp_path / "ws"
    md = (REPO_ROOT / "states" / "configure_reward.md").read_text()
    calls: list[str] = []

    class Track:
        def ask(self, hop):
            calls.append(hop.title)
            return PromptDecision.APPROVE

    evaluate_hand_offs(
        state_name="configure_reward",
        state_md=md,
        hitl=False,
        prompter=Track(),
        workspace=ws,
    )
    assert calls, "gate should fire when condition unevaluable"


def test_prompt_goes_to_stderr_not_stdout(capsys) -> None:
    p = StdioPrompter(err=sys.stderr, inp=io.StringIO("y\n"))
    p.ask(_hop("go"))
    captured = capsys.readouterr()
    assert "hand-off" in captured.err
    assert captured.out == ""


# ── EventPrompter (dashboard channel) ──────────────────────────────────────


import json as _json
import threading
import time


def _run_prompt_with_decision(
    prompter: EventPrompter,
    hop: HandOffPoint,
    decision_body: dict,
    *,
    delay: float = 0.05,
) -> tuple[str, Path, Path]:
    """Fire prompter.ask on the current thread; a helper thread waits for the
    request file to appear then drops `decision_body` in the decisions dir.
    Returns (decision, request_path, decision_path) — request/decision paths
    let callers assert cleanup after the call returns."""
    reqs = prompter.workspace / "hitl" / "requests"
    decs = prompter.workspace / "hitl" / "decisions"
    seen: dict[str, Path] = {}

    def writer():
        deadline = time.time() + 5
        while time.time() < deadline:
            if reqs.is_dir():
                found = list(reqs.glob("*.json"))
                if found:
                    seen["req"] = found[0]
                    rid = found[0].stem
                    decs.mkdir(parents=True, exist_ok=True)
                    dec_path = decs / f"{rid}.json"
                    dec_path.write_text(_json.dumps(decision_body))
                    seen["dec"] = dec_path
                    return
            time.sleep(delay)

    t = threading.Thread(target=writer)
    t.start()
    result = prompter.ask(hop)
    t.join(timeout=5)
    return result, seen.get("req"), seen.get("dec")


def test_event_prompter_approve_roundtrip(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    p = EventPrompter(workspace=ws, poll_interval=0.02)
    dec, req_path, dec_path = _run_prompt_with_decision(
        p, _hop("t"), {"decision": "approve"}
    )
    assert dec == PromptDecision.APPROVE
    # Both files must be cleaned up after ask() returns
    assert not req_path.exists(), "request file should be deleted"
    assert not dec_path.exists(), "decision file should be deleted"


def test_event_prompter_deny_and_skip(tmp_path: Path) -> None:
    p = EventPrompter(workspace=tmp_path, poll_interval=0.02)
    d1, *_ = _run_prompt_with_decision(p, _hop("a"), {"decision": "deny"})
    assert d1 == PromptDecision.DENY
    d2, *_ = _run_prompt_with_decision(p, _hop("b"), {"decision": "skip"})
    assert d2 == PromptDecision.SKIP


def test_event_prompter_unknown_decision_falls_back_to_deny(tmp_path: Path) -> None:
    p = EventPrompter(workspace=tmp_path, poll_interval=0.02)
    dec, *_ = _run_prompt_with_decision(p, _hop("q"), {"decision": "yes-please"})
    assert dec == PromptDecision.DENY


def test_event_prompter_malformed_json_falls_back_to_deny(tmp_path: Path) -> None:
    p = EventPrompter(workspace=tmp_path, poll_interval=0.02)
    reqs = tmp_path / "hitl" / "requests"
    decs = tmp_path / "hitl" / "decisions"

    def writer():
        deadline = time.time() + 5
        while time.time() < deadline:
            if reqs.is_dir():
                found = list(reqs.glob("*.json"))
                if found:
                    decs.mkdir(parents=True, exist_ok=True)
                    (decs / (found[0].stem + ".json")).write_text("{not json")
                    return
            time.sleep(0.02)

    t = threading.Thread(target=writer)
    t.start()
    dec = p.ask(_hop("q"))
    t.join(timeout=5)
    assert dec == PromptDecision.DENY


def test_event_prompter_request_contains_hop_fields(tmp_path: Path) -> None:
    p = EventPrompter(workspace=tmp_path, poll_interval=0.02)
    hop = HandOffPoint(
        state="configure_reward",
        title="Approve custom reward",
        description="always-on gate — inspect reward.py",
        is_always_on=True,
        condition="reward_kind ∈ {custom, shaped}",
    )
    captured: dict = {}

    def writer():
        reqs = tmp_path / "hitl" / "requests"
        decs = tmp_path / "hitl" / "decisions"
        deadline = time.time() + 5
        while time.time() < deadline:
            if reqs.is_dir():
                found = list(reqs.glob("*.json"))
                if found:
                    captured["body"] = _json.loads(found[0].read_text())
                    decs.mkdir(parents=True, exist_ok=True)
                    (decs / (found[0].stem + ".json")).write_text('{"decision":"approve"}')
                    return
            time.sleep(0.02)

    t = threading.Thread(target=writer)
    t.start()
    p.ask(hop)
    t.join(timeout=5)
    body = captured["body"]
    assert body["state"] == "configure_reward"
    assert body["title"] == "Approve custom reward"
    assert body["is_always_on"] is True
    assert body["condition"] == "reward_kind ∈ {custom, shaped}"
    assert "id" in body and isinstance(body["id"], str)
    assert isinstance(body["created_ts"], float)


def test_event_prompter_emits_events_when_emitter_set(tmp_path: Path) -> None:
    """Optional emitter integration — approval_request/approval_decision events."""
    from harness.events import EventEmitter

    sink = io.StringIO()
    emitter = EventEmitter(session_id="S", model="M", cwd=".", sink=sink)
    p = EventPrompter(workspace=tmp_path, poll_interval=0.02, emitter=emitter)

    dec, *_ = _run_prompt_with_decision(p, _hop("gate"), {"decision": "approve"})
    assert dec == PromptDecision.APPROVE

    events = [_json.loads(l) for l in sink.getvalue().splitlines() if l.strip()]
    types = [e["type"] for e in events]
    assert types == ["approval_request", "approval_decision"]
    assert events[0]["request"]["title"] == "gate"
    assert events[1]["decision"] == PromptDecision.APPROVE
    assert events[0]["request"]["id"] == events[1]["request_id"]


def test_event_prompter_evaluate_hand_offs_integration(tmp_path: Path) -> None:
    """End-to-end: evaluate_hand_offs uses EventPrompter, threads decisions in."""
    ws = tmp_path / "workspace"
    p = EventPrompter(workspace=ws, poll_interval=0.02)

    md = (REPO_ROOT / "states" / "intake.md").read_text()
    # writer thread: approve whatever request appears
    stop = threading.Event()

    def writer():
        reqs = ws / "hitl" / "requests"
        decs = ws / "hitl" / "decisions"
        while not stop.is_set():
            if reqs.is_dir():
                for f in reqs.glob("*.json"):
                    decs.mkdir(parents=True, exist_ok=True)
                    (decs / (f.stem + ".json")).write_text('{"decision":"approve"}')
            time.sleep(0.02)

    t = threading.Thread(target=writer)
    t.start()
    try:
        result = evaluate_hand_offs(
            state_name="intake",
            state_md=md,
            hitl=True,
            prompter=p,
            workspace=ws,
        )
    finally:
        stop.set()
        t.join(timeout=5)
    assert result.approved is True
    assert not result.denied_titles


import sys  # bottom-of-file so the fixture is not confused
