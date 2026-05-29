#!/usr/bin/env python3
"""Detached, non-LLM training watcher for `monitor_training` (arm-and-detach).

This is the cheap process the agent launches and steps back from. It owns the
tight poll loop so the LLM agent does not have to busy-loop across a multi-hour /
multi-day run. It:

  * polls job state at the target cadence (local-direct / local-slurm / ssh-slurm),
  * tails stdout/stderr by byte-offset into workspace/logs/job_log.md,
  * parses verl's namespaced metric dict + tqdm line into workspace/logs/progress.csv,
  * scans for anomalies AND divergence (entropy explosion / validation regression),
  * on a fatal/divergence finding, appends workspace/logs/escalation.md (wakes the agent),
  * on terminal status, writes workspace/job/job_status.md and touches workspace/job/terminal.

Reference implementation — stdlib only. Adapt the metric keys / cadence to the run.
Launch detached, e.g.:  nohup python watch_poller.py <run_workspace> >/dev/null 2>&1 &

The companion rules (regexes, thresholds) live in skills/training_monitor/default.md;
this script is the executable mirror of them, kept deliberately conservative
(it SURFACES — it never auto-scancels; auto-intervention is a separate, opt-in policy).
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

# ---- cadence by target (seconds) — minimums for a *script*, not the agent ----
CADENCE = {"local-direct": 30, "local-slurm": 60, "ssh-slurm": 90}

# ---- metric parsing ----
# verl's console logger emits ONE LINE PER STEP in the shape:
#   step:6 - global_seqlen/mean:4185.0 - actor/entropy:0.42 - actor/pg_clipfrac:np.float64(0.0) - ...
# (sometimes prefixed by a ray `(TaskRunner pid=...)` tag). This mirrors the proven
# _KV_RE in web/.../parser.py — values are bare numerics OR np.<dtype>(<numeric>).
_KV_RE = re.compile(
    r"(?P<key>[A-Za-z_][\w/.@+\-]*):(?:np\.\w+\((?P<wrapped>[-+0-9.eE]+)\)|(?P<bare>[-+0-9.eE]+))"
)
_METRIC_LINE_PRE = re.compile(r"\bstep:\d+\s*-\s")
_GS_KEY_HINT = "training/global_step"
TERMINAL_SUCCESS = re.compile(r"Final validation metrics:")

ANOMALY_PATTERNS = {
    "OOM": re.compile(r"torch\.cuda\.OutOfMemoryError|CUDA out of memory|Out of memory: Killed process"),
    "NaN/Inf": re.compile(r"loss is nan|loss=inf|gradient is nan|returned nan values"),
    "vLLM": re.compile(r"vllm\.engine.*?(Error|Aborted)"),
    # Real NCCL failure: a collective HANG or unhandled error — these stall training.
    "NCCL-hang": re.compile(r"Watchdog caught collective operation timeout|Timeout waiting for ack|NCCL.*unhandled (system|cuda) error|NCCL.*remote process exited"),
    # Benign NCCL chatter (e.g. "WARN Call to bind failed: Address already in use" from the
    # RAS daemon during Ray/vLLM bring-up) — record for visibility, do NOT escalate. This
    # bind warning appears on healthy runs and must not trip a divergence alarm.
    "NCCL-warn": re.compile(r"NCCL WARN"),
    "preempt": re.compile(r"slurmstepd: error: \*\*\* (STEP|JOB) \d+.*CANCELLED AT"),
}
FATAL_KINDS = {"OOM", "NaN/Inf", "NCCL-hang"}  # promote ONLY these to escalation.md


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_job_info(ws: str) -> dict:
    info, path = {}, os.path.join(ws, "job", "job_info.md")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            m = re.match(r"\s*[-*]?\s*(\w+):\s*(.+?)\s*$", line)
            if m:
                info[m.group(1)] = m.group(2)
    return info


def poll_status(info: dict) -> tuple[str, str]:
    """Return (phase, final_state). phase in {running, exited}; final_state for slurm."""
    target = info.get("target", "local-direct")
    if target == "local-direct":
        pid = info.get("pid", "")
        if pid and subprocess.run(["kill", "-0", pid]).returncode == 0:
            return "running", ""
        return "exited", ""
    jobid = info.get("slurm_jobid", "")
    pre = ["ssh", info["remote_alias"]] if target == "ssh-slurm" and info.get("remote_alias") else []
    live = subprocess.run(pre + ["squeue", "-j", jobid, "--noheader", "-o", "%T"],
                          capture_output=True, text=True).stdout.strip()
    if live in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING"}:
        return "running", live
    state = subprocess.run(pre + ["sacct", "-j", jobid, "--format=State", "--parsable2", "--noheader"],
                           capture_output=True, text=True).stdout.strip().splitlines()
    return "exited", (state[0].strip() if state else "UNKNOWN")


def tail(ws: str, info: dict) -> str:
    """Append new bytes of stdout/stderr to job_log.md; return the new chunk."""
    state_path = os.path.join(ws, "logs", "_tail_state.json")
    offsets = json.load(open(state_path)) if os.path.exists(state_path) else {}
    chunk = ""
    for key in ("stdout_log", "stderr_log"):
        p = info.get(key)
        if not p or not os.path.exists(p):
            continue
        off = offsets.get(p, 0)
        size = os.path.getsize(p)
        if size < off:  # log rotated / truncated
            off = 0
        if size > off:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                f.seek(off)
                new = f.read()
                offsets[p] = f.tell()
            chunk += new
            with open(os.path.join(ws, "logs", "job_log.md"), "a", encoding="utf-8") as o:
                o.write(new)
    json.dump(offsets, open(state_path, "w"))
    return chunk


def parse_metrics(chunk: str) -> list[dict]:
    rows = []
    for raw in chunk.splitlines():
        if _GS_KEY_HINT not in raw or not _METRIC_LINE_PRE.search(raw):
            continue
        line = raw
        if "(TaskRunner" in line:  # strip ray actor prefix
            idx = line.find("step:")
            if idx >= 0:
                line = line[idx:]
        row = {m.group("key"): (m.group("wrapped") or m.group("bare")) for m in _KV_RE.finditer(line)}
        if row:
            row["wallclock"] = now()
            rows.append(row)
    return rows


def append_progress(ws: str, rows: list[dict]) -> None:
    if not rows:
        return
    path = os.path.join(ws, "logs", "progress.csv")
    existing = []
    if os.path.exists(path):
        existing = list(csv.DictReader(open(path)))
    cols, seen = [], set()
    for r in existing + rows:
        for k in r:
            if k not in seen:
                seen.add(k); cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in existing + rows:
            w.writerow(r)


def record_anomaly(ws: str, body: str) -> None:
    """Append to anomalies.md in the dashboard parser's `- [ts] — body` format."""
    with open(os.path.join(ws, "logs", "anomalies.md"), "a", encoding="utf-8") as f:
        f.write(f"- [{now()}] — {body[:300]}\n")


def escalate(ws: str, kind: str, detail: str) -> None:
    """Wake the agent (escalation.md) AND surface on the dashboard (anomalies.md)."""
    with open(os.path.join(ws, "logs", "escalation.md"), "a", encoding="utf-8") as f:
        f.write(f"- {now()} — **{kind}** — {detail}\n")
    record_anomaly(ws, f"{kind}: {detail}")


def scan_anomalies(ws: str, chunk: str) -> None:
    if not chunk:
        return
    for line in chunk.splitlines():
        for kind, pat in ANOMALY_PATTERNS.items():
            if pat.search(line):
                if kind in FATAL_KINDS:
                    escalate(ws, kind, line.strip()[:300])
                else:
                    record_anomaly(ws, f"{kind}: {line.strip()}")


def _floats(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except (KeyError, ValueError, TypeError):
            pass
    return out


def scan_divergence(ws: str) -> None:
    """Entropy explosion + validation regression (joint = highest confidence)."""
    path = os.path.join(ws, "logs", "progress.csv")
    if not os.path.exists(path):
        return
    rows = list(csv.DictReader(open(path)))
    if len(rows) < 25:
        return
    ent = _floats(rows, "actor/entropy")
    entropy_exploding = False
    if len(ent) >= 25:
        base = sorted(ent[:20])[len(ent[:20]) // 2]  # median of first 20
        recent = ent[-1]
        rising = ent[-1] > ent[-25]
        entropy_exploding = base > 0 and recent > 2 * base and rising
    # validation regression: best-so-far vs latest, higher-is-better reward
    val = _floats(rows, "val/reward/mean")
    val_regressing = len(val) >= 3 and max(val) > 0 and val[-1] < 0.8 * max(val)
    if entropy_exploding:
        escalate(ws, "entropy-explosion", f"actor/entropy {ent[-1]:.3f} > 2x baseline and rising")
    if val_regressing:
        escalate(ws, "val-regression", f"val/reward/mean {val[-1]:.4f} < 80% of best {max(val):.4f}")
    if entropy_exploding and val_regressing:
        escalate(ws, "DIVERGENCE", "entropy climbing AND val reward regressing — policy diverging, will not self-recover")


def write_status(ws: str, status: str, info: dict, extra: dict) -> None:
    lines = [f"# Job status\n\n## Status\n{status}\n",
             f"\n## Job\n- target: {info.get('target')}\n- pid/slurm_jobid: {info.get('pid', info.get('slurm_jobid'))}\n- output_dir: {info.get('output_dir')}\n"]
    if extra:
        lines.append("\n## Terminal facts\n" + "".join(f"- {k}: {v}\n" for k, v in extra.items()))
    with open(os.path.join(ws, "job", "job_status.md"), "w", encoding="utf-8") as f:
        f.write("".join(lines))
    open(os.path.join(ws, "job", "terminal"), "w").close()


def latest_ckpt(output_dir: str) -> str | None:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    steps = [d for d in os.listdir(output_dir) if d.startswith("global_step_")]
    if not steps:
        return None
    best = max(steps, key=lambda d: int(d.rsplit("_", 1)[-1]))
    return os.path.join(output_dir, best)  # <output_dir>/global_step_<N>  (no checkpoints/ subdir)


def main(ws: str) -> int:
    info = read_job_info(ws)
    target = info.get("target", "local-direct")
    interval = CADENCE.get(target, 60)
    os.makedirs(os.path.join(ws, "logs"), exist_ok=True)
    saw_terminal_marker = False

    while True:
        if os.path.exists(os.path.join(ws, "job", "cancel_requested")):
            jobid = info.get("slurm_jobid")
            if jobid:
                subprocess.run((["ssh", info["remote_alias"]] if target == "ssh-slurm" else []) + ["scancel", jobid])
            elif info.get("pid"):
                subprocess.run(["kill", info["pid"]])
            write_status(ws, "cancelled", info, {})
            return 0

        chunk = tail(ws, info)
        if TERMINAL_SUCCESS.search(chunk):
            saw_terminal_marker = True
        scan_anomalies(ws, chunk)
        append_progress(ws, parse_metrics(chunk))
        scan_divergence(ws)

        phase, final_state = poll_status(info)
        if phase == "exited":
            ckpt = latest_ckpt(info.get("output_dir", ""))
            ec = ""
            ecp = os.path.join(ws, "job", "exit_code")
            if os.path.exists(ecp):
                ec = open(ecp).read().strip()
            success = (final_state == "COMPLETED") or (target == "local-direct" and ec == "0")
            success = success and (ckpt is not None or saw_terminal_marker)
            if success:
                write_status(ws, "success", info, {"last_checkpoint": ckpt or "(save_freq=-1, none)"})
            elif final_state in {"TIMEOUT", "PREEMPTED"}:
                write_status(ws, "preempted", info, {"last_checkpoint": ckpt or "none"})
            else:
                write_status(ws, "crashed", info, {"final_state": final_state or f"exit={ec}", "last_checkpoint": ckpt or "none"})
            return 0

        time.sleep(interval)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: watch_poller.py <run_workspace_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
