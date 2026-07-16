"""RunLog — meta.json + state_log.md primitives."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from harness.runlog import RunLog, new_meta


REPO_ROOT = Path(__file__).resolve().parents[2]


# The exact regex the dashboard uses to parse state_log entries.
_DASHBOARD_STATE_LOG_RE = re.compile(
    r"^-\s*\[(?P<ts>[^\]]+)\]\s*#(?P<step>\d+)\s+entered\s+"
    r"(?P<state>[^,]+?)\s*,\s*from\s+(?P<from>.+?)\s*$"
)


@pytest.fixture
def rl(tmp_path: Path) -> RunLog:
    rl = RunLog(runs_root=tmp_path / "runs", run_id="T-test")
    rl.init(
        new_meta(
            run_id="T-test",
            goal="test goal",
            model="anthropic/claude-haiku-4-5",
            session_id="sess-1",
            hitl=False,
            verl_root="/opt/verl",
        )
    )
    return rl


def test_init_creates_layout(rl: RunLog) -> None:
    assert rl.meta_path.exists()
    assert rl.state_log_path.exists()
    assert rl.logs_dir.is_dir()


def test_init_meta_schema(rl: RunLog) -> None:
    data = json.loads(rl.meta_path.read_text())
    assert data["run_id"] == "T-test"
    assert data["goal"] == "test goal"
    assert data["model"] == "anthropic/claude-haiku-4-5"
    assert data["session_id"] == "sess-1"
    assert data["status"] == "running"
    assert data["step"] == 0
    assert data["completed_at"] is None
    assert data["hitl"] is False
    assert data["verl_root"] == "/opt/verl"
    assert data["started_ts"] > 0


def test_enter_state_appends_dashboard_compatible_line(rl: RunLog) -> None:
    step1 = rl.enter_state("intake")
    step2 = rl.enter_state("locate_recipe", previous="intake")
    assert step1 == 1
    assert step2 == 2

    lines = [
        ln
        for ln in rl.state_log_path.read_text().splitlines()
        if ln.startswith("- [")
    ]
    assert len(lines) == 2

    m1 = _DASHBOARD_STATE_LOG_RE.match(lines[0])
    assert m1 is not None
    assert m1.group("step") == "1"
    assert m1.group("state") == "intake"
    assert m1.group("from") == "<start>"

    m2 = _DASHBOARD_STATE_LOG_RE.match(lines[1])
    assert m2 is not None
    assert m2.group("step") == "2"
    assert m2.group("state") == "locate_recipe"
    assert m2.group("from") == "intake"


def test_meta_tracks_current_state_and_step(rl: RunLog) -> None:
    rl.enter_state("intake")
    rl.enter_state("locate_recipe", previous="intake")
    meta = rl.read_meta()
    assert meta is not None
    assert meta.step == 2
    assert meta.current_state == "locate_recipe"


def test_mark_terminal_completed(rl: RunLog) -> None:
    rl.enter_state("intake")
    rl.mark_terminal(status="completed")
    meta = rl.read_meta()
    assert meta.status == "completed"
    assert meta.completed_at is not None
    assert meta.completed_ts is not None


def test_mark_terminal_cancelled_with_note(rl: RunLog) -> None:
    rl.enter_state("intake")
    rl.mark_terminal(status="cancelled", note="user Ctrl-C at intake")
    text = rl.state_log_path.read_text()
    assert "cancelled" in text
    assert "user Ctrl-C" in text


def test_mark_terminal_rejects_bad_status(rl: RunLog) -> None:
    with pytest.raises(ValueError, match="invalid terminal status"):
        rl.mark_terminal(status="running")
    with pytest.raises(ValueError, match="invalid terminal status"):
        rl.mark_terminal(status="not-a-real-status")


def test_note_adds_line_without_touching_meta(rl: RunLog) -> None:
    rl.enter_state("intake")
    step_before = rl.read_meta().step
    rl.note("side note")
    assert rl.read_meta().step == step_before
    assert "side note" in rl.state_log_path.read_text()


def test_read_meta_missing_returns_none(tmp_path: Path) -> None:
    rl = RunLog(runs_root=tmp_path / "runs", run_id="never-inited")
    assert rl.read_meta() is None


def test_enter_state_before_init_errors(tmp_path: Path) -> None:
    rl = RunLog(runs_root=tmp_path / "runs", run_id="no-init")
    with pytest.raises(RuntimeError, match="init"):
        rl.enter_state("intake")


def test_meta_atomic_write_survives_interrupted_read(rl: RunLog, tmp_path: Path) -> None:
    """A concurrent read while enter_state is writing should never see partial JSON."""
    # Simulate the invariant by checking .tmp doesn't leak
    rl.enter_state("intake")
    tmp = rl.meta_path.with_suffix(".json.tmp")
    assert not tmp.exists()
    # And the real meta.json is always valid JSON
    json.loads(rl.meta_path.read_text())


def test_init_truncates_stale_state_log(tmp_path: Path) -> None:
    """Re-running a run_id starts state_log fresh — no stale lines survive."""
    from harness.runlog import new_meta

    # First session
    rl1 = RunLog(runs_root=tmp_path / "runs", run_id="R")
    rl1.init(
        new_meta(run_id="R", goal="g1", model="m", session_id="s1", hitl=False, verl_root=None)
    )
    rl1.enter_state("intake")
    rl1.enter_state("locate_recipe", previous="intake")
    stale = [
        ln for ln in rl1.state_log_path.read_text().splitlines()
        if _DASHBOARD_STATE_LOG_RE.match(ln)
    ]
    assert len(stale) == 2

    # Second session — same run_id, NOT resumed → state_log truncated
    rl2 = RunLog(runs_root=tmp_path / "runs", run_id="R")
    rl2.init(
        new_meta(run_id="R", goal="g2", model="m", session_id="s2", hitl=False, verl_root=None),
        resumed=False,
    )
    body = rl2.state_log_path.read_text()
    fresh = [ln for ln in body.splitlines() if _DASHBOARD_STATE_LOG_RE.match(ln)]
    assert "state_log — run R" in body  # header restored
    assert fresh == []  # no stale entries survive


def test_init_resumed_preserves_state_log(tmp_path: Path) -> None:
    """--resume-run continuations keep prior state_log content."""
    from harness.runlog import new_meta

    rl1 = RunLog(runs_root=tmp_path / "runs", run_id="R")
    rl1.init(
        new_meta(run_id="R", goal="g1", model="m", session_id="s1", hitl=False, verl_root=None)
    )
    rl1.enter_state("intake")
    rl1.enter_state("locate_recipe", previous="intake")

    rl2 = RunLog(runs_root=tmp_path / "runs", run_id="R")
    rl2.init(
        new_meta(run_id="R", goal="g1", model="m", session_id="s2", hitl=False, verl_root=None),
        resumed=True,
    )
    lines = [
        ln for ln in rl2.state_log_path.read_text().splitlines()
        if _DASHBOARD_STATE_LOG_RE.match(ln)
    ]
    assert len(lines) == 2  # both prior entries preserved


def test_state_log_line_count_matches_step(rl: RunLog) -> None:
    for i, s in enumerate(["intake", "locate_recipe", "configure_algorithm", "finalize"]):
        prev = ["<start>", "intake", "locate_recipe", "configure_algorithm"][i]
        rl.enter_state(s, previous=prev)
    body = rl.state_log_path.read_text()
    entered = [ln for ln in body.splitlines() if _DASHBOARD_STATE_LOG_RE.match(ln)]
    assert len(entered) == 4
