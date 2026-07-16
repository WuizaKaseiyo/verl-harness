"""Resume support — pick up a run at its last recorded state.

M2 baseline is intentionally simple: re-enter the last state FRESH. We do NOT
resurrect the mid-state conversation. Rationale:
  - The workspace already contains everything upstream states wrote.
  - The T5 snapshot renderer will seed the fresh conversation with those files.
  - Resurrecting a mid-state chat requires replaying tool_results and matching
    session ids, and the payoff is small (each state usually 5-15 turns).

Interrupt semantics:
  - Ctrl-C → OrchestrationResult(reason="cancelled") + meta.status="cancelled".
  - `--resume-run <id>` reads state_log's last "entered" line, flips meta.status
    back to "running", and starts the orchestrator at that state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from harness.runlog import RunLog, RunMeta


_STATE_LOG_ENTER = re.compile(
    r"^-\s*\[[^\]]+\]\s*#\d+\s+entered\s+(?P<state>[^,]+?)\s*,",
    re.MULTILINE,
)


class ResumeError(Exception):
    """No meta.json / no state_log / corrupt state — cannot resume."""


@dataclass(frozen=True)
class ResumePlan:
    """Everything the orchestrator needs to pick up a paused run."""

    meta: RunMeta
    last_state: str


def load_resume_plan(runs_root: Path, run_id: str) -> ResumePlan:
    """Read the run's meta.json + state_log.md to figure out where to restart."""
    rl = RunLog(runs_root=runs_root, run_id=run_id)
    if not rl.meta_path.exists():
        raise ResumeError(f"cannot resume: {rl.meta_path} does not exist")
    meta = rl.read_meta()
    if meta is None:
        raise ResumeError(f"cannot resume: {rl.meta_path} is unreadable")
    if not rl.state_log_path.exists():
        raise ResumeError(f"cannot resume: {rl.state_log_path} does not exist")

    body = rl.state_log_path.read_text(encoding="utf-8")
    entered = _STATE_LOG_ENTER.findall(body)
    if not entered:
        raise ResumeError(
            f"cannot resume: {rl.state_log_path} has no `entered` lines"
        )
    last_state = entered[-1]
    return ResumePlan(meta=meta, last_state=last_state)


def flip_status_to_running(runs_root: Path, run_id: str, *, note: str) -> None:
    """Reset meta.status back to running before an orchestrator re-entry."""
    rl = RunLog(runs_root=runs_root, run_id=run_id)
    meta = rl.read_meta()
    if meta is None:
        raise ResumeError(f"cannot flip status: meta.json missing for {run_id}")
    meta.status = "running"
    meta.completed_at = None
    meta.completed_ts = None
    rl.write_meta(meta)
    rl.note(f"resumed: {note}")
