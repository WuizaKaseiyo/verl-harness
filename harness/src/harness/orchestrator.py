"""Multi-state orchestrator — walks the FSM from entry to a terminal.

Each state's execution is delegated to `drive_state` (state_driver.py). The
orchestrator adds:
  - RunLog init + per-state `enter_state` line + terminal `mark_terminal`
  - FSM.load(workdir) validation
  - Chaining logic (current → next_state until terminal or error)
  - Global step cap (`max_states`) to prevent runaway walks
  - Interrupt handling — SIGINT + SIGTERM both translate to CancelledError
    so external `kill -TERM <pid>` marks meta.status=cancelled cleanly
"""

from __future__ import annotations

import asyncio
import dataclasses
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import json

from harness.backends.base import Backend
from harness.cli import RunConfig
from harness.fsm import FSM, FSMError
from harness.hitl import Prompter
from harness.runlog import RunLog, new_meta
from harness.state_driver import StateResult, drive_state


# ── wakeup budget ─────────────────────────────────────────────────────────

MAX_TOTAL_WAIT_PER_STATE_SEC = 24 * 3600  # 24 hours per state


def _wakeup_state_path(workspace: Path) -> Path:
    return workspace / "logs" / "wakeup_state.json"


def _load_wakeup_state(workspace: Path) -> dict[str, dict[str, int]]:
    p = _wakeup_state_path(workspace)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    waits = data.get("waits") if isinstance(data, dict) else None
    if not isinstance(waits, dict):
        return {}
    # Normalize: each value should be {count, total_seconds}
    out: dict[str, dict[str, int]] = {}
    for k, v in waits.items():
        if isinstance(v, dict):
            out[k] = {
                "count": int(v.get("count", 0)),
                "total_seconds": int(v.get("total_seconds", 0)),
            }
    return out


def _save_wakeup_state(workspace: Path, state: dict[str, dict[str, int]]) -> None:
    p = _wakeup_state_path(workspace)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"waits": state}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    import os as _os
    _os.replace(tmp, p)


def _bump_wakeup(workspace: Path, state_name: str, seconds: int) -> int:
    ws = _load_wakeup_state(workspace)
    entry = ws.setdefault(state_name, {"count": 0, "total_seconds": 0})
    entry["count"] += 1
    entry["total_seconds"] += seconds
    _save_wakeup_state(workspace, ws)
    return entry["total_seconds"]


# ── signal handling ───────────────────────────────────────────────────────


class _SignalBox:
    """Shared mutable state between the signal handler and orchestrate()'s
    except block. asyncio.Task has no `__dict__` so we can't stash attributes
    on it directly."""

    def __init__(self) -> None:
        self.signal_name: str | None = None


def _install_signal_handlers(box: _SignalBox) -> list[signal.Signals]:
    """Register asyncio-safe SIGTERM+SIGINT handlers that cancel the current task.

    Returns the signals actually installed so orchestrate() can restore them
    on exit — matters when the runtime is embedded (tests, web server) rather
    than running as a standalone CLI.
    """
    loop = asyncio.get_running_loop()
    current = asyncio.current_task()
    installed: list[signal.Signals] = []

    def _handler_for(sig: signal.Signals):
        def _cancel() -> None:
            box.signal_name = sig.name
            if current is not None and not current.done():
                current.cancel()
        return _cancel

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handler_for(sig))
            installed.append(sig)
        except (NotImplementedError, RuntimeError, ValueError):
            # NotImplementedError: some platforms (older Windows).
            # RuntimeError: not in main thread.
            # ValueError: SIGTERM on Windows.
            pass
    return installed


def _restore_signal_handlers(installed: list[signal.Signals]) -> None:
    loop = asyncio.get_running_loop()
    for sig in installed:
        try:
            loop.remove_signal_handler(sig)
        except (NotImplementedError, ValueError, RuntimeError):
            pass


@dataclass
class OrchestrationResult:
    """End-of-run summary."""

    is_error: bool
    reason: str  # "completed" | "crashed" | "max_states" | "fsm_error" | "cancelled" | "max_wait_exceeded"
    message: str
    final_state: str
    visited: list[str] = field(default_factory=list)
    total_turns: int = 0
    state_results: list[StateResult] = field(default_factory=list)
    total_usage: dict[str, int] = field(default_factory=dict)


async def orchestrate(
    *,
    config: RunConfig,
    backend: Backend,
    workspace_root: Path,
    session_id: str,
    sink: IO[str] = sys.stdout,
    max_iterations_per_state: int = 100,
    max_tokens: int = 4096,
    max_states: int = 50,
    hitl_prompter: Prompter | None = None,
    resumed: bool = False,
) -> OrchestrationResult:
    """Drive from `config.state` until a terminal state or an error.

    `max_states` is a safety cap — the FSM's own loop bounds land in T7.

    `resumed=True` indicates the caller is restarting an in-progress run:
    skip fresh RunLog.init() (would clobber the existing meta.json + header)
    and instead flip status back to "running" from wherever it was left.
    """
    # ── FSM load ──────────────────────────────────────────────────────────
    try:
        fsm = FSM.load(config.workdir)
    except FSMError as e:
        # Bootstrap enough runlog to make the failure visible to the dashboard.
        rl = RunLog(runs_root=workspace_root, run_id=config.run_id)
        if not rl.meta_path.exists():
            rl.init(
                new_meta(
                    run_id=config.run_id,
                    goal=config.goal,
                    model=config.model,
                    session_id=session_id,
                    hitl=config.hitl,
                    verl_root=str(config.verl_root) if config.verl_root else None,
                )
            )
        rl.mark_terminal(status="crashed", note=f"FSM load failed: {e}")
        return OrchestrationResult(
            is_error=True,
            reason="fsm_error",
            message=str(e),
            final_state=config.state,
        )

    runlog = RunLog(runs_root=workspace_root, run_id=config.run_id)
    if not resumed:
        runlog.init(
            new_meta(
                run_id=config.run_id,
                goal=config.goal,
                model=config.model,
                session_id=session_id,
                hitl=config.hitl,
                verl_root=str(config.verl_root) if config.verl_root else None,
            )
        )
    # else: caller (cli.py) has already flip_status_to_running()'d the meta.

    current = config.state
    previous = "<start>" if not resumed else "<resume>"
    # `total_usage` and `total_turns` are threaded through _walk via this dict
    # so the exception handler can surface them if the run is cancelled.
    shared: dict[str, Any] = {"total_usage": {}, "total_turns": 0}

    # Install signal handlers so SIGTERM (external timeout / kill) is treated
    # the same way as SIGINT (Ctrl-C) — task cancellation → clean shutdown.
    signal_box = _SignalBox()
    installed = _install_signal_handlers(signal_box)

    try:
        return await _walk(
            fsm=fsm,
            runlog=runlog,
            config=config,
            backend=backend,
            workspace_root=workspace_root,
            session_id=session_id,
            sink=sink,
            max_iterations_per_state=max_iterations_per_state,
            max_tokens=max_tokens,
            max_states=max_states,
            hitl_prompter=hitl_prompter,
            current=current,
            previous=previous,
            shared=shared,
        )
    except (KeyboardInterrupt, asyncio.CancelledError) as e:
        meta = runlog.read_meta()
        landed = meta.current_state if meta else current
        signal_name = signal_box.signal_name or type(e).__name__
        runlog.mark_terminal(
            status="cancelled", note=f"interrupted at {landed} ({signal_name})"
        )
        return OrchestrationResult(
            is_error=True,
            reason="cancelled",
            message=f"interrupted at {landed} ({signal_name})",
            final_state=landed,
            total_turns=shared["total_turns"],
            total_usage=dict(shared["total_usage"]),
        )
    finally:
        _restore_signal_handlers(installed)


async def _walk(
    *,
    fsm: FSM,
    runlog: RunLog,
    config: RunConfig,
    backend: Backend,
    workspace_root: Path,
    session_id: str,
    sink: IO[str],
    max_iterations_per_state: int,
    max_tokens: int,
    max_states: int,
    hitl_prompter: Prompter | None,
    current: str,
    previous: str,
    shared: dict[str, Any],
) -> OrchestrationResult:
    visited: list[str] = []
    state_results: list[StateResult] = []
    # Share these with the outer orchestrate() so cancellation still reports
    # what was accumulated up to the interrupt.
    total_usage: dict[str, int] = shared["total_usage"]
    total_turns: int = shared["total_turns"]

    def _accumulate_usage(u: dict[str, int]) -> None:
        for k, v in u.items():
            total_usage[k] = total_usage.get(k, 0) + int(v or 0)

    def _persist_usage_to_meta() -> None:
        meta = runlog.read_meta()
        if meta is None:
            return
        meta.extra = dict(meta.extra or {})
        meta.extra["total_usage"] = dict(total_usage)
        meta.extra["total_turns"] = shared["total_turns"]
        runlog.write_meta(meta)

    for _ in range(max_states):
        # Sanity: don't try to enter a non-existent state
        try:
            _ = fsm.state(current)
        except FSMError as e:
            runlog.mark_terminal(status="crashed", note=str(e))
            return OrchestrationResult(
                is_error=True,
                reason="fsm_error",
                message=str(e),
                final_state=current,
                visited=visited,
                total_turns=total_turns,
                state_results=state_results,
                total_usage=dict(total_usage),
            )

        runlog.enter_state(current, previous=previous)
        visited.append(current)

        state_config = dataclasses.replace(config, state=current)
        result = await drive_state(
            config=state_config,
            backend=backend,
            workspace_root=workspace_root,
            session_id=session_id,
            sink=sink,
            max_iterations=max_iterations_per_state,
            max_tokens=max_tokens,
            hitl_prompter=hitl_prompter,
        )
        state_results.append(result)
        total_turns += result.turns
        shared["total_turns"] = total_turns
        _accumulate_usage(result.usage)
        _persist_usage_to_meta()

        # ── schedule_wakeup: pause, then re-enter same state ──────────────
        if result.reason == "wait" and result.wait_seconds is not None:
            workspace = workspace_root / config.run_id / "workspace"
            total_waited = _bump_wakeup(workspace, current, result.wait_seconds)
            if total_waited > MAX_TOTAL_WAIT_PER_STATE_SEC:
                note = (
                    f"{current}: cumulative wait {total_waited}s "
                    f"exceeded cap {MAX_TOTAL_WAIT_PER_STATE_SEC}s"
                )
                runlog.mark_terminal(status="crashed", note=note)
                return OrchestrationResult(
                    is_error=True,
                    reason="max_wait_exceeded",
                    message=note,
                    final_state=current,
                    visited=visited,
                    total_turns=total_turns,
                    state_results=state_results,
                    total_usage=dict(total_usage),
                )
            runlog.note(
                f"wakeup at {current}: sleeping {result.wait_seconds}s "
                f"({result.wait_reason}); total wait {total_waited}s"
            )
            await asyncio.sleep(result.wait_seconds)
            runlog.note(f"resumed at {current} after wakeup")
            # Re-enter same state fresh — no transition, no visited bump.
            continue

        if result.is_error:
            runlog.mark_terminal(
                status="crashed",
                note=f"{current}: {result.reason} — {result.message}",
            )
            return OrchestrationResult(
                is_error=True,
                reason="crashed",
                message=f"{current}: {result.message}",
                final_state=current,
                visited=visited,
                total_turns=total_turns,
                state_results=state_results,
                total_usage=dict(total_usage),
            )

        if fsm.is_terminal(current):
            runlog.mark_terminal(status="completed")
            return OrchestrationResult(
                is_error=False,
                reason="completed",
                message=f"FSM drained through {current}",
                final_state=current,
                visited=visited,
                total_turns=total_turns,
                state_results=state_results,
                total_usage=dict(total_usage),
            )

        # Chain to the next state
        if result.next_state is None:
            runlog.mark_terminal(
                status="crashed",
                note=f"{current}: state driver returned no next_state on a non-terminal",
            )
            return OrchestrationResult(
                is_error=True,
                reason="crashed",
                message=f"{current} yielded no next_state on non-terminal",
                final_state=current,
                visited=visited,
                total_turns=total_turns,
                state_results=state_results,
                total_usage=dict(total_usage),
            )
        previous = current
        current = result.next_state

    runlog.mark_terminal(
        status="crashed", note=f"exceeded max_states={max_states}"
    )
    return OrchestrationResult(
        is_error=True,
        reason="max_states",
        message=f"exceeded orchestration cap max_states={max_states}",
        final_state=current,
        visited=visited,
        total_turns=total_turns,
        state_results=state_results,
        total_usage=dict(total_usage),
    )
