"""SIGTERM / SIGINT handling — sends signal mid-run, checks graceful shutdown.

Uses a subprocess because Python's signal handling is process-global; can't
safely test in the same process as pytest.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_SRC = REPO_ROOT / "harness" / "src"


def _script(runs_root: Path, run_id: str) -> str:
    """A tiny driver that runs orchestrate() with a backend that never returns."""
    return f"""
import asyncio, sys, io
from pathlib import Path
sys.path.insert(0, {str(HARNESS_SRC)!r})

from harness.backends.base import Backend, RawEvent
from harness.cli import RunConfig
from harness.hitl import AutoApprovePrompter
from harness.orchestrator import orchestrate

class HangingBackend(Backend):
    model_id = "hanging"
    async def stream(self, *, system, messages, tools, max_tokens=4096):
        sys.stderr.write("HANG STARTED\\n"); sys.stderr.flush()
        await asyncio.sleep(60)  # hang until cancelled
        yield RawEvent(kind="message_stop", stop_reason="end_turn")

async def main():
    cfg = RunConfig(
        workdir=Path({str(REPO_ROOT)!r}),
        model="scripted/hanging",
        goal="signal test",
        state="intake",
        run_id={run_id!r},
        hitl=False,
        verl_root=None,
    )
    r = await orchestrate(
        config=cfg,
        backend=HangingBackend(),
        workspace_root=Path({str(runs_root)!r}),
        session_id="sess-sig",
        sink=io.StringIO(),
        hitl_prompter=AutoApprovePrompter(),
    )
    sys.stderr.write(f"ORCH DONE reason={{r.reason}}\\n"); sys.stderr.flush()
    print(r.reason)

asyncio.run(main())
"""


def _spawn(runs_root: Path, run_id: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _script(runs_root, run_id)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_meta(runs_root: Path, run_id: str, *, timeout: float = 10.0) -> Path:
    """Poll for meta.json to appear."""
    meta_path = runs_root / run_id / "meta.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if meta_path.exists():
            return meta_path
        time.sleep(0.1)
    raise TimeoutError(f"meta.json never appeared: {meta_path}")


def test_sigterm_marks_cancelled(tmp_path: Path) -> None:
    """External SIGTERM mid-run → meta.status=cancelled + state_log has cancel line."""
    runs_root = tmp_path / "runs"
    run_id = "sigterm-test"

    proc = _spawn(runs_root, run_id)
    try:
        # Wait for the child to enter the intake state (meta.json + state_log written)
        _wait_for_meta(runs_root, run_id)
        time.sleep(0.5)  # give state_driver a moment to enter drive_state
        proc.send_signal(signal.SIGTERM)
        _stdout, stderr = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()

    meta = json.loads((runs_root / run_id / "meta.json").read_text())
    assert meta["status"] == "cancelled", (meta, stderr)
    assert meta["completed_at"] is not None

    log = (runs_root / run_id / "workspace" / "logs" / "state_log.md").read_text()
    assert "cancelled" in log
    assert "SIGTERM" in log, log


def test_sigint_marks_cancelled(tmp_path: Path) -> None:
    """SIGINT (Ctrl-C) → meta.status=cancelled + state_log records SIGINT."""
    runs_root = tmp_path / "runs"
    run_id = "sigint-test"

    proc = _spawn(runs_root, run_id)
    try:
        _wait_for_meta(runs_root, run_id)
        time.sleep(0.5)
        proc.send_signal(signal.SIGINT)
        _stdout, stderr = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()

    meta = json.loads((runs_root / run_id / "meta.json").read_text())
    assert meta["status"] == "cancelled", (meta, stderr)
    log = (runs_root / run_id / "workspace" / "logs" / "state_log.md").read_text()
    assert "SIGINT" in log, log
