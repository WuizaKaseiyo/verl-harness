"""Task submission — spawn `claude` in the background and expose log tail.

Detached subprocess; server request handler returns immediately with a
task_id. Task metadata + log are stored under /tmp/verl-harness-web-tasks/
so they survive dashboard restart but do not pollute the repo.
"""
from __future__ import annotations

import errno
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

MAX_RESUMES = 30                # backstop against runaway resume loops
AUTO_RESUME_COOLDOWN = 30       # seconds between "no ScheduleWakeup" resumes

TASKS_ROOT = Path("/tmp/verl-harness-web-tasks")
CLAUDE_BIN = os.environ.get(
    "CLAUDE_BIN", "/home/y50047367/.npm-global/bin/claude")
DEFAULT_VERL_HOME = os.environ.get(
    "VERL_HOME", "/home/y50047367/verl")
# Explicit model pin. Without this, --print defaults to whatever the CLI's
# system default is (was fable-5 in earlier runs, which is heavy on thinking).
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-8")

PROMPT_TEMPLATE = """You are driving the verl-harness FSM at {harness_root}.
Read CLAUDE.md, task-overview.md, and states/intake.md, then apply intake.
Walk transitions per each state's `## Next States` block.
Honor every `## Hand-off Points` block.

Intent: Train {algorithm} on {dataset} using {model}.{extra_line}
verl checkout: {verl_home}
HITL: {hitl_flag}
"""


def _new_task_id() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"t-{ts}-{secrets.token_hex(2)}"


def _compose_prompt(harness_root: Path, form: dict, verl_home: str) -> str:
    extra = (form.get("extra") or "").strip()
    return PROMPT_TEMPLATE.format(
        harness_root=str(harness_root),
        algorithm=(form.get("algorithm") or "").strip(),
        model=(form.get("model") or "").strip(),
        dataset=(form.get("dataset") or "").strip(),
        extra_line=(f" {extra}" if extra else ""),
        verl_home=verl_home,
        hitl_flag=("on" if form.get("hitl") else "--no-hitl"),
    )


def _spawn_claude(harness_root: Path, verl_home: str, session_id: str,
                  log_f, resume: bool = False) -> subprocess.Popen:
    """Spawn `claude --print` with a stable session id so continuations can
    resume it. Callers write the prompt via stdin after spawn."""
    args = [CLAUDE_BIN,
            "--print",
            "--output-format=stream-json",
            "--verbose",
            "--model", CLAUDE_MODEL]
    args += (["--resume", session_id]
             if resume else ["--session-id", session_id])
    args += ["--dangerously-skip-permissions",
             "--add-dir", str(harness_root)]
    return subprocess.Popen(
        args,
        cwd=str(harness_root),
        stdin=subprocess.PIPE,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "VERL_HOME": verl_home},
    )


def submit_task(harness_root: Path, form: dict) -> dict:
    algorithm = (form.get("algorithm") or "").strip()
    model = (form.get("model") or "").strip()
    dataset = (form.get("dataset") or "").strip()
    if not (algorithm and model and dataset):
        return {"error": "algorithm, model, dataset are required"}

    verl_home = (form.get("verl_home") or DEFAULT_VERL_HOME).strip()
    prompt = _compose_prompt(harness_root, form, verl_home)

    task_id = _new_task_id()
    session_id = str(uuid.uuid4())
    task_dir = TASKS_ROOT / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (task_dir / "session_id").write_text(session_id)
    (task_dir / "meta.json").write_text(json.dumps({
        "task_id": task_id,
        "session_id": session_id,
        "started_ts": time.time(),
        "form": form,
        "verl_home": verl_home,
    }), encoding="utf-8")

    log_path = task_dir / "stdout.log"
    log_f = log_path.open("w")
    try:
        proc = _spawn_claude(harness_root, verl_home, session_id, log_f)
    except FileNotFoundError as e:
        return {"error": f"claude binary not found: {e}"}
    try:
        proc.stdin.write(prompt.encode("utf-8"))
        proc.stdin.close()
    except Exception:
        pass
    (task_dir / "pid").write_text(str(proc.pid))

    # Supervisor watches for ScheduleWakeup after each turn ends, sleeps the
    # requested delay, then respawns with --resume so the FSM can keep walking.
    threading.Thread(
        target=_supervise_task,
        args=(task_dir, session_id, harness_root, verl_home),
        daemon=True,
        name=f"supervise-{task_id}",
    ).start()

    return {"task_id": task_id, "pid": proc.pid,
            "log_path": str(log_path), "session_id": session_id}


def _find_last_wakeup(log_path: Path) -> dict | None:
    """Return the last ScheduleWakeup tool_use input, only considering
    events emitted AFTER the most recent `system/init` (i.e. the current
    invocation, not stale ones from prior continuations)."""
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    lines = text.splitlines()
    last_init = 0
    for i in range(len(lines) - 1, -1, -1):
        try:
            e = json.loads(lines[i])
        except Exception:
            continue
        if e.get("type") == "system" and e.get("subtype") == "init":
            last_init = i
            break
    last_wakeup = None
    for i in range(last_init, len(lines)):
        try:
            e = json.loads(lines[i])
        except Exception:
            continue
        if e.get("type") != "assistant":
            continue
        for c in e.get("message", {}).get("content", []):
            if c.get("type") == "tool_use" and c.get("name") == "ScheduleWakeup":
                last_wakeup = c.get("input", {})
    return last_wakeup


def _last_result_is_error(log_path: Path) -> bool | None:
    """Walk the log backward to find the last {type:result} event's is_error.
    Returns None if no result event found."""
    if not log_path.exists() or log_path.stat().st_size == 0:
        return None
    try:
        with log_path.open("rb") as f:
            f.seek(max(0, log_path.stat().st_size - 65536))
            tail = f.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "result":
            return bool(e.get("is_error"))
    return None


def _find_active_run(harness_root: Path, started_ts: float) -> Path | None:
    """Newest `runs/*/workspace/logs/state_log.md` modified after task started.
    That's the run this task's subprocess most recently touched."""
    runs_dir = harness_root / "runs"
    if not runs_dir.is_dir():
        return None
    best: tuple[float, Path] | None = None
    for run in runs_dir.iterdir():
        if not run.is_dir():
            continue
        sl = run / "workspace" / "logs" / "state_log.md"
        if not sl.exists():
            continue
        mt = sl.stat().st_mtime
        if mt < started_ts:
            continue
        if best is None or mt > best[0]:
            best = (mt, run)
    return best[1] if best else None


_STATE_LINE = re.compile(r"^-\s*\[[^\]]*\]\s*#(\d+)\s+entered\s+(\w+)")


def _read_last_state(run: Path) -> tuple[int, str] | None:
    """Parse the last `#N entered NAME, from ...` line in a run's state_log.md."""
    sl = run / "workspace" / "logs" / "state_log.md"
    try:
        text = sl.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    for line in reversed(text.splitlines()):
        m = _STATE_LINE.match(line.strip())
        if m:
            return (int(m.group(1)), m.group(2))
    return None


def _bump_resume_count(task_dir: Path) -> int:
    p = task_dir / "resume_count"
    try:
        n = int(p.read_text().strip()) if p.exists() else 0
    except Exception:
        n = 0
    n += 1
    p.write_text(str(n))
    return n


def _read_resume_count(task_dir: Path) -> int:
    p = task_dir / "resume_count"
    try:
        return int(p.read_text().strip()) if p.exists() else 0
    except Exception:
        return 0


def _decide_next_step(task_dir: Path, harness_root: Path) -> dict | None:
    """After a subprocess exits, decide whether/how to spawn a continuation.

    Returns None if the task is truly finished (or unrecoverable). Otherwise
    returns {'kind': 'wakeup'|'auto', 'delay': int, 'prompt': str} — supervisor
    sleeps `delay`, then spawns claude --resume with `prompt` on stdin.
    """
    if _read_resume_count(task_dir) >= MAX_RESUMES:
        return None
    log_path = task_dir / "stdout.log"

    wakeup = _find_last_wakeup(log_path)
    if wakeup:
        try:
            delay = int(wakeup.get("delaySeconds", 60))
        except Exception:
            delay = 60
        delay = max(10, min(delay, 3600))
        wp = (wakeup.get("prompt") or "").strip()
        if not wp or wp.startswith("<<autonomous-loop"):
            wp = (
                "Wakeup fired. Continue driving the harness FSM: check any "
                "background jobs, refresh the freshest workspace files, and "
                "transition to the next state. "
                f"Prior reason: {wakeup.get('reason', '')}"
            )
        return {"kind": "wakeup", "delay": delay, "prompt": wp}

    # No ScheduleWakeup. Auto-resume if the subprocess exited *cleanly* but the
    # FSM's most recent run isn't at `finalize` yet — this catches the pattern
    # where claude detaches an sbatch and returns without scheduling a wakeup.
    is_err = _last_result_is_error(log_path)
    if is_err is None or is_err:
        return None  # no result / errored → don't auto-resume

    try:
        meta = json.loads((task_dir / "meta.json").read_text())
        started_ts = float(meta.get("started_ts", 0))
    except Exception:
        return None
    active_run = _find_active_run(harness_root, started_ts)
    if active_run is None:
        return None
    last_state = _read_last_state(active_run)
    if last_state is None or last_state[1] == "finalize":
        return None

    prompt = (
        "Wake up. Your prior turn exited cleanly but the FSM is still at "
        f"state #{last_state[0]} `{last_state[1]}` in run "
        f"`{active_run.name}` — you have NOT reached `finalize` yet. "
        "\n\nCheck the freshest artefacts in that run's workspace/ dir:\n"
        "- any detached sbatch job you submitted (e.g., sanity probe or "
        "training job) may have completed; look for signal files like "
        "`workspace/sanity/probe_done` or `workspace/job/job_status.md`.\n"
        "- `squeue -u $USER` to see if a job is still pending/running.\n"
        "- read any log or probe_stdout to inspect results.\n\n"
        "Then transition to the next state per the current state's "
        "`## Next States` block. If a background job is still running, "
        "call ScheduleWakeup with a reasonable delay so I can wake you "
        "again when it's likely done. If work is genuinely blocked "
        "waiting on human decision, halt honestly."
    )
    return {"kind": "auto", "delay": AUTO_RESUME_COOLDOWN, "prompt": prompt}


def _wait_for_pid_exit(pid: int, task_dir: Path, poll_secs: float = 2.0) -> None:
    """Block until `pid` is gone. Uses os.waitpid where the process is our
    child (initial spawn); falls back to polling os.kill(pid, 0) when adopting
    a task whose subprocess was started by a previous server incarnation."""
    try:
        os.waitpid(pid, 0)
        return
    except ChildProcessError:
        pass
    while task_dir.is_dir():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(poll_secs)


def _supervise_task(task_dir: Path, session_id: str,
                    harness_root: Path, verl_home: str) -> None:
    """Per-task daemon thread. Each iteration: wait for the current subprocess
    to exit, decide whether to spawn a continuation (ScheduleWakeup or
    auto-resume when the FSM hasn't reached finalize), sleep the delay, spawn.
    Exits when the task is truly done or `task_dir` is deleted."""
    log_path = task_dir / "stdout.log"
    pid_file = task_dir / "pid"

    while task_dir.is_dir():
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
            except Exception:
                return
            _wait_for_pid_exit(pid, task_dir)
        if not task_dir.is_dir():
            return

        step = _decide_next_step(task_dir, harness_root)
        if step is None:
            return

        (task_dir / "sleeping_until").write_text(str(time.time() + step["delay"]))
        remaining = step["delay"]
        while remaining > 0 and task_dir.is_dir():
            time.sleep(min(remaining, 5))
            remaining -= 5
        try:
            (task_dir / "sleeping_until").unlink()
        except FileNotFoundError:
            pass
        if not task_dir.is_dir():
            return

        _bump_resume_count(task_dir)
        try:
            log_f = log_path.open("a", encoding="utf-8")
        except Exception:
            return
        try:
            proc = _spawn_claude(harness_root, verl_home, session_id,
                                 log_f, resume=True)
        except FileNotFoundError:
            return
        try:
            proc.stdin.write(step["prompt"].encode("utf-8"))
            proc.stdin.close()
        except Exception:
            pass
        if not task_dir.is_dir():
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
            return
        pid_file.write_text(str(proc.pid))


def _reap_zombies() -> None:
    """Best-effort reap of any children the server owns. Called on read/list.
    Avoids the OS keeping <defunct> claude entries around forever."""
    try:
        while True:
            wpid, _ = os.waitpid(-1, os.WNOHANG)
            if wpid == 0:
                return
    except ChildProcessError:
        return


def _terminal_verdict(task_dir: Path) -> str:
    """`done` if the last stream-json event is a success result; else `failed`.

    Covers both the well-behaved case (claude emits {"type":"result",
    "is_error":false}) and the ugly cases (plain-text error on argv/env
    problems, hard crash mid-stream, no output at all)."""
    log_path = task_dir / "stdout.log"
    if not log_path.exists() or log_path.stat().st_size == 0:
        return "failed"
    try:
        with log_path.open("rb") as f:
            f.seek(max(0, log_path.stat().st_size - 32768))
            tail = f.read().decode("utf-8", errors="replace")
    except Exception:
        return "failed"
    # Walk backward for the *last* `type=result` event. system/task_notification
    # events come AFTER the result event and would otherwise short-circuit this.
    result_event = None
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") == "result":
            result_event = e
            break
    if not isinstance(result_event, dict):
        return "failed"
    return "failed" if result_event.get("is_error") else "done"


def _task_status(task_dir: Path) -> str:
    if (task_dir / "killed").exists():
        return "killed"
    sleep_file = task_dir / "sleeping_until"
    if sleep_file.exists():
        try:
            until = float(sleep_file.read_text().strip())
            if time.time() < until:
                return "sleeping"
        except Exception:
            pass
    pid_file = task_dir / "pid"
    if not pid_file.exists():
        return "unknown"
    try:
        pid = int(pid_file.read_text().strip())
    except Exception:
        return "unknown"
    try:
        os.kill(pid, 0)
        return "running"
    except ProcessLookupError:
        return _terminal_verdict(task_dir)
    except PermissionError:
        return "running"


def _kill_pid(pid: int) -> None:
    """SIGTERM → wait 2s → SIGKILL. Prefers the process group so children die too."""
    def _signal(sig):
        try:
            os.killpg(pid, sig); return True
        except (ProcessLookupError, PermissionError):
            pass
        except OSError as e:
            if e.errno == errno.ESRCH:
                return False
        try:
            os.kill(pid, sig); return True
        except ProcessLookupError:
            return False

    _signal(signal.SIGTERM)
    for _ in range(20):
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
    _signal(signal.SIGKILL)
    time.sleep(0.3)


def delete_task(task_id: str) -> dict:
    """Kill the process if still running, then rmtree the task dir.

    Unified 'x' semantics: removes the task from the recent-tasks list
    regardless of prior state (running / done / failed / killed)."""
    task_dir = TASKS_ROOT / task_id
    if not task_dir.is_dir():
        return {"error": "no such task"}
    was_running = False
    pid_file = task_dir / "pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            was_running = True
            _kill_pid(pid)
        except (ProcessLookupError, ValueError, PermissionError):
            pass
    _reap_zombies()
    try:
        shutil.rmtree(task_dir)
    except Exception as e:
        return {"error": f"could not remove task dir: {e}"}
    return {"task_id": task_id, "removed": True, "was_running": was_running}


def read_task(task_id: str, tail: int = 200) -> dict:
    _reap_zombies()
    task_dir = TASKS_ROOT / task_id
    if not task_dir.is_dir():
        return {"error": "no such task"}
    log_path = task_dir / "stdout.log"
    log = ""
    if log_path.exists():
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            log = "".join(lines[-tail:])
        except Exception:
            log = ""
    meta = {}
    meta_path = task_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass
    return {
        "task_id": task_id,
        "status": _task_status(task_dir),
        "log": log,
        "meta": meta,
    }


def adopt_orphan_supervisors(harness_root: Path) -> list[str]:
    """On server start, spin up a supervisor thread for every task_dir whose
    prior server left it in a "half-successful, awaits resume" state (subprocess
    exited but FSM not finalize). Called once from server startup.

    Returns the list of task_ids adopted (for logging)."""
    adopted: list[str] = []
    if not TASKS_ROOT.is_dir():
        return adopted
    for task_dir in TASKS_ROOT.iterdir():
        if not task_dir.is_dir():
            continue
        if (task_dir / "killed").exists():
            continue
        try:
            meta = json.loads((task_dir / "meta.json").read_text())
        except Exception:
            continue
        session_id = meta.get("session_id")
        if not session_id:
            continue                  # pre-session-id task; can't --resume
        verl_home = meta.get("verl_home") or DEFAULT_VERL_HOME
        # If subprocess still alive, an existing (or restarting) supervisor
        # will pick it up on its next iteration — but from a NEW server, no
        # supervisor exists yet, so start one anyway. The supervisor is idempotent
        # to being spawned twice (both would waitpid the same pid; second one
        # exits harmlessly).
        # If subprocess is dead, we still start one — it'll decide whether to
        # resume via _decide_next_step.
        if _decide_next_step(task_dir, harness_root) is None:
            # Nothing left to do for this task.
            continue
        threading.Thread(
            target=_supervise_task,
            args=(task_dir, session_id, harness_root, verl_home),
            daemon=True,
            name=f"adopt-{task_dir.name}",
        ).start()
        adopted.append(task_dir.name)
    return adopted


def list_tasks(limit: int = 20) -> list[dict]:
    _reap_zombies()
    if not TASKS_ROOT.is_dir():
        return []
    dirs = [d for d in TASKS_ROOT.iterdir() if d.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for d in dirs[:limit]:
        meta = {}
        mp = d / "meta.json"
        if mp.exists():
            try:
                meta = json.loads(mp.read_text())
            except Exception:
                pass
        out.append({
            "task_id": d.name,
            "status": _task_status(d),
            "started_ts": meta.get("started_ts"),
            "form": meta.get("form", {}),
        })
    return out
