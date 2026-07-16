"""Run metadata + state-transition log.

Two artefacts per run:

  runs/<run_id>/meta.json                     — status + provenance
  runs/<run_id>/workspace/logs/state_log.md   — human-readable timeline

meta.json schema (all keys, some nullable):

  {
    "run_id":       str,
    "goal":         str,
    "model":        str,
    "verl_root":    str | None,
    "session_id":   str,
    "started_at":   ISO-8601 UTC,
    "started_ts":   float (unix seconds),
    "current_state": str,
    "status":       "running" | "completed" | "crashed" | "cancelled",
    "hitl":         bool,
    "completed_at": ISO-8601 UTC | null,
    "completed_ts": float | null,
  }

state_log.md line format matches web/src/verl_harness_web/parser.py::_STATE_LOG_RE:

  - [2026-07-14T10:35:00Z] #1 entered intake, from <start>
  - [2026-07-14T10:35:22Z] #2 entered locate_recipe, from intake
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_VALID_STATUS = {"running", "completed", "crashed", "cancelled"}


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_ts() -> float:
    return time.time()


@dataclass
class RunMeta:
    """The mutable half of the run's metadata."""

    run_id: str
    goal: str
    model: str
    session_id: str
    hitl: bool
    started_at: str
    started_ts: float
    verl_root: str | None = None
    current_state: str = ""
    status: str = "running"
    completed_at: str | None = None
    completed_ts: float | None = None
    step: int = 0
    extra: dict = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        d = asdict(self)
        return d


class RunLog:
    """Owns `runs/<run_id>/meta.json` + `workspace/logs/state_log.md`.

    Instance methods are safe to call from a single process; concurrent writes
    from *multiple* processes should not happen in normal use (a run has one
    owning runtime), but we still take an flock on state_log for defence.
    """

    def __init__(self, runs_root: Path, run_id: str) -> None:
        self.run_dir = runs_root / run_id
        self.workspace = self.run_dir / "workspace"
        self.logs_dir = self.workspace / "logs"
        self.meta_path = self.run_dir / "meta.json"
        self.state_log_path = self.logs_dir / "state_log.md"
        self.run_id = run_id

    # ── setup / read ──────────────────────────────────────────────────────

    def init(self, meta: RunMeta, *, resumed: bool = False) -> None:
        """First-time run bootstrap. Creates dirs, writes meta.json + header.

        When `resumed=False` (default), any stale `state_log.md` is truncated
        so the new run starts with a clean history. Pass `resumed=True` to
        keep the existing state_log — used by `--resume-run` continuations.
        """
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.write_meta(meta)
        if not resumed or not self.state_log_path.exists():
            self.state_log_path.write_text(
                f"# state_log — run {self.run_id}\n\n"
                f"Machine-readable format (dashboard-compatible):\n"
                f"`- [<ISO8601>] #<step> entered <state>, from <prev>`\n\n"
            )

    def read_meta(self) -> RunMeta | None:
        if not self.meta_path.exists():
            return None
        try:
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return RunMeta(
            run_id=data.get("run_id", self.run_id),
            goal=data.get("goal", ""),
            model=data.get("model", ""),
            session_id=data.get("session_id", ""),
            hitl=bool(data.get("hitl", False)),
            started_at=data.get("started_at", ""),
            started_ts=float(data.get("started_ts", 0.0)),
            verl_root=data.get("verl_root"),
            current_state=data.get("current_state", ""),
            status=data.get("status", "running"),
            completed_at=data.get("completed_at"),
            completed_ts=data.get("completed_ts"),
            step=int(data.get("step", 0)),
            extra=data.get("extra", {}),
        )

    def write_meta(self, meta: RunMeta) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(meta.to_json_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, self.meta_path)

    # ── mutations ─────────────────────────────────────────────────────────

    def enter_state(self, target: str, *, previous: str = "<start>") -> int:
        """Append a state_log line, bump step, and update meta.current_state.

        Returns the new step number.
        """
        meta = self.read_meta()
        if meta is None:
            raise RuntimeError(
                f"RunLog.enter_state called before init(): {self.meta_path} missing"
            )
        meta.step += 1
        meta.current_state = target
        line = f"- [{_now_iso()}] #{meta.step} entered {target}, from {previous}\n"
        self._append_state_log(line)
        self.write_meta(meta)
        return meta.step

    def mark_terminal(self, *, status: str, note: str | None = None) -> None:
        """Move meta.status to a terminal value and record when."""
        if status not in _VALID_STATUS or status == "running":
            raise ValueError(f"invalid terminal status {status!r}")
        meta = self.read_meta()
        if meta is None:
            raise RuntimeError("no meta.json to mark terminal")
        meta.status = status
        meta.completed_at = _now_iso()
        meta.completed_ts = _now_ts()
        self.write_meta(meta)
        if note:
            self._append_state_log(f"- [{_now_iso()}] -- {status}: {note} --\n")
        else:
            self._append_state_log(f"- [{_now_iso()}] -- {status} --\n")

    def note(self, message: str) -> None:
        """Add a free-form line to state_log without touching meta."""
        self._append_state_log(f"- [{_now_iso()}] -- {message} --\n")

    def _append_state_log(self, text: str) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_log_path, "a", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except OSError:
                pass  # some filesystems don't support flock; append is still ~atomic
            try:
                f.write(text)
            finally:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass


def new_meta(
    *,
    run_id: str,
    goal: str,
    model: str,
    session_id: str,
    hitl: bool,
    verl_root: str | None,
) -> RunMeta:
    """Build a fresh RunMeta with started_at/ts populated."""
    return RunMeta(
        run_id=run_id,
        goal=goal,
        model=model,
        session_id=session_id,
        hitl=hitl,
        started_at=_now_iso(),
        started_ts=_now_ts(),
        verl_root=verl_root,
    )
