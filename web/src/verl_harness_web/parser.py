"""Parse a verl-harness folder and its latest run's training artifacts.

Self-contained, no external deps. Two layers:

- Layer 1 (harness FSM): task-overview.md + states/ + skills/ — structurally identical
  to FastHarness in general, since verl-harness is a FastHarness-compatible harness.
- Layer 2 (verl-specific run artefacts): the run's workspace/ directory has
  training-specific files this parser knows how to read — progress.csv,
  anomalies.md, job_status.md, job_info.md, job_log.md, summary.md.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------- Layer 1: harness FSM -----------------------------------------

@dataclass
class Transition:
    target: str
    condition: str
    deliverables: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class State:
    name: str
    path: Path
    description: str
    skills: list[str] = field(default_factory=list)
    human_checkpoints: str = ""
    transitions: list[Transition] = field(default_factory=list)
    raw: str = ""

    @property
    def is_terminal(self) -> bool:
        return not self.transitions


@dataclass
class Harness:
    root: Path
    title: str
    overview: str
    starting_state: str
    hitl: str
    required_capabilities: list[str]
    notes: str
    states: dict[str, State]
    raw_overview: str = ""

    # ---- runs ---------------------------------------------------------------
    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def list_runs(self) -> list[Path]:
        """List run dirs sorted by mtime ascending (latest last).

        Alphabetical sorting confuses runs that interleave prefixes (e.g.,
        `harness-e2e-...` vs `harness-smoke-...`) — mtime always wins.
        """
        if not self.runs_dir.exists():
            return []
        runs = [p for p in self.runs_dir.iterdir() if p.is_dir()]
        try:
            return sorted(runs, key=lambda p: p.stat().st_mtime)
        except OSError:
            return sorted(runs)

    def latest_run(self) -> Path | None:
        runs = self.list_runs()
        return runs[-1] if runs else None

    def run_meta(self, run: Path) -> dict:
        meta_path = run / "meta.json"
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def state_log_path(self, run: Path | None = None) -> Path | None:
        run = run or self.latest_run()
        return (run / "workspace" / "logs" / "state_log.md") if run else None

    # ---- skills -------------------------------------------------------------
    def list_skill_folders(self) -> list[str]:
        seen: set[str] = set()
        for state in self.states.values():
            for s in state.skills:
                seen.add(s.strip("/").rstrip("/"))
        skills_root = self.root / "skills"
        if skills_root.exists():
            for p in sorted(skills_root.iterdir()):
                if p.is_dir():
                    seen.add(f"skills/{p.name}")
        return sorted(seen)

    def states_using_skill(self, skill_path: str) -> list[str]:
        target = skill_path.strip("/")
        return sorted(
            name for name, state in self.states.items()
            if any(s.strip("/") == target for s in state.skills)
        )

    def read_skill_files(self, skill_path: str) -> list[tuple[str, str]]:
        folder = self.root / skill_path.strip("/")
        if not folder.exists() or not folder.is_dir():
            return []
        out: list[tuple[str, str]] = []
        for p in sorted(folder.glob("*.md")):
            try:
                out.append((p.name, p.read_text(encoding="utf-8")))
            except Exception:
                pass
        return out


# ---------- Layer 1 parsing ----------------------------------------------

_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _split_h2(body: str) -> dict[str, str]:
    """Split a markdown body by ## headings; return {heading: section_text}."""
    sections: dict[str, str] = {}
    cur_heading = ""
    cur_chunk: list[str] = []
    for line in body.splitlines():
        m = _H2_RE.match(line)
        if m:
            if cur_heading:
                sections[cur_heading] = "\n".join(cur_chunk).strip()
            cur_heading = m.group(1).strip()
            cur_chunk = []
        else:
            cur_chunk.append(line)
    if cur_heading:
        sections[cur_heading] = "\n".join(cur_chunk).strip()
    return sections


def _parse_bullets(body: str) -> list[str]:
    return [m.group(1).strip()
            for m in re.finditer(r"^-\s+(.+?)\s*$", body, re.MULTILINE)]


def _parse_capability_tokens(body: str) -> list[str]:
    """Capabilities may carry inline `# comments` — strip those when listing."""
    out = []
    for item in _parse_bullets(body):
        token = item.split("#", 1)[0].strip()
        if token:
            out.append(token)
    return out


def _parse_state(path: Path) -> State:
    text = path.read_text(encoding="utf-8")
    secs = _split_h2(text)
    desc = secs.get("Description", "").strip()
    skills_block = secs.get("Skills", "")
    # Skill bullets may carry inline "# comment" — strip them, same as capability tokens.
    skills = []
    for b in _parse_bullets(skills_block):
        token = b.split("#", 1)[0].strip()
        if token.startswith("skills/"):
            skills.append(token)
    # Canonical key is `Hand-off Points` (matches FastHarness validate-harness +
    # fastharness-web/tui parsers). The older `Human Checkpoints` name is read
    # as a fallback for in-flight harnesses that haven't migrated yet.
    hcps = secs.get("Hand-off Points", secs.get("Human Checkpoints", "")).strip()
    transitions: list[Transition] = []

    next_block = secs.get("Next States", "")
    if next_block:
        # Each `### <name>` introduces a transition.
        h3_re = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
        positions = [(m.group(1).strip(), m.end()) for m in h3_re.finditer(next_block)]
        ends = [p[1] for p in positions[1:]] + [len(next_block)]
        for (name, start), end in zip(positions, ends):
            block = next_block[start:end]
            cond_m = re.search(r"\*\*Condition:\*\*\s*(.+?)(?=\n\n|\*\*Deliverables:\*\*|$)",
                               block, re.DOTALL)
            condition = cond_m.group(1).strip() if cond_m else ""
            deliv: list[tuple[str, str]] = []
            deliv_m = re.search(r"\*\*Deliverables:\*\*\s*\n(.+?)$",
                                block, re.DOTALL)
            if deliv_m:
                for line in deliv_m.group(1).splitlines():
                    bm = re.match(r"^-\s*(.+?):\s*(.+?)\s*$", line)
                    if bm:
                        deliv.append((bm.group(1).strip(), bm.group(2).strip()))
            transitions.append(Transition(target=name, condition=condition,
                                          deliverables=deliv))

    return State(name=path.stem, path=path, description=desc, skills=skills,
                 human_checkpoints=hcps, transitions=transitions, raw=text)


def parse_harness(root: Path) -> Harness:
    overview_path = root / "task-overview.md"
    raw = overview_path.read_text(encoding="utf-8") if overview_path.exists() else ""
    title_m = re.match(r"^#\s+(.+?)\s*$", raw, re.MULTILINE)
    title = title_m.group(1).strip() if title_m else root.name
    secs = _split_h2(raw)
    overview_text = secs.get("Overview", "").strip()
    starting = ""
    sstate = secs.get("Starting State", "").strip()
    if sstate:
        # bare path; strip leading bullet if present
        starting = sstate.splitlines()[0].lstrip("- ").strip()
        if starting.startswith("states/"):
            starting = Path(starting).stem
    # Canonical key is `Hand-off Points`; accept the older `Human in the Loop`
    # as a fallback for in-flight harnesses.
    _hitl_raw = secs.get("Hand-off Points", secs.get("Human in the Loop", ""))
    hitl = _hitl_raw.strip().split("\n", 1)[0].strip()
    caps = _parse_capability_tokens(secs.get("Required Capabilities", ""))
    notes = secs.get("Notes", "").strip()

    states: dict[str, State] = {}
    states_dir = root / "states"
    if states_dir.exists():
        for p in sorted(states_dir.glob("*.md")):
            try:
                states[p.stem] = _parse_state(p)
            except Exception:
                pass

    return Harness(root=root, title=title, overview=overview_text,
                   starting_state=starting, hitl=hitl,
                   required_capabilities=caps, notes=notes,
                   states=states, raw_overview=raw)


# ---------- Layer 2: verl run artefacts ----------------------------------

_STATE_LOG_RE = re.compile(
    r"^-\s*\[(?P<ts>[^\]]+)\]\s*#(?P<step>\d+)\s+entered\s+"
    r"(?P<state>[^,]+?)\s*,\s*from\s+(?P<from>.+?)\s*$"
)


def parse_state_log(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _STATE_LOG_RE.match(line.strip())
        if m:
            rows.append({
                "timestamp": m.group("ts"),
                "step": int(m.group("step")),
                "state": m.group("state").strip(),
                "from": m.group("from").strip(),
            })
    return rows


def read_progress_csv(run: Path) -> dict:
    """Read workspace/logs/progress.csv → series for charting."""
    p = run / "workspace" / "logs" / "progress.csv"
    if not p.exists():
        return {"rows": 0, "columns": [], "series": {}}
    rows: list[dict] = []
    try:
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            for r in reader:
                rows.append(r)
    except Exception:
        return {"rows": 0, "columns": [], "series": {}}

    # Convert numeric columns to float where possible.
    series: dict[str, list] = {c: [] for c in cols}
    for r in rows:
        for c in cols:
            v = r.get(c, "")
            try:
                series[c].append(float(v))
            except (TypeError, ValueError):
                series[c].append(v)
    return {"rows": len(rows), "columns": cols, "series": series}


# ---------- stdout metric extractor (verl one-line metric dicts) ----------

# Match each `key:value` segment.
#   key  = chars in [\w / @ . + -] (covers e.g. `val-aux/chess_fen_cycle/board_score/mean@1`)
#   val  = bare numeric (-?\d.\d+|e[-+]?\d+) OR np.<dtype>(<numeric>)
_KV_RE = re.compile(
    r"(?P<key>[A-Za-z_][\w/.@+\-]*)"
    r":"
    r"(?:np\.\w+\((?P<wrapped>[-+0-9.eE]+)\)|(?P<bare>[-+0-9.eE]+))"
)

# The full metric line starts with `step:N - ` and contains `training/global_step:N`.
_METRIC_LINE_PRE = re.compile(r"\bstep:\d+\s*-\s")
_GS_KEY_HINT = "training/global_step"


def _scan_stdout_metric_lines(stdout_path: Path) -> list[dict]:
    """Yield one parsed-row dict per metric line found in the trainer's stdout.

    Each row is a {key: float} dict containing every key:value pair on that line.
    Only lines that contain BOTH `step:N -` AND `training/global_step` are considered
    metric lines (this rules out the millions of `Added request <uuid>` and similar).
    """
    if not stdout_path.exists():
        return []
    rows: list[dict] = []
    # Stream-read so we don't blow memory on huge slurm .out files.
    try:
        with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                if _GS_KEY_HINT not in raw or not _METRIC_LINE_PRE.search(raw):
                    continue
                # Strip ray prefix like `(TaskRunner pid=...)` if present.
                line = raw
                if "(TaskRunner" in line:
                    idx = line.find("step:")
                    if idx >= 0:
                        line = line[idx:]
                row: dict[str, float] = {}
                for m in _KV_RE.finditer(line):
                    val_str = m.group("wrapped") or m.group("bare")
                    try:
                        row[m.group("key")] = float(val_str)
                    except (TypeError, ValueError):
                        continue
                if row.get(_GS_KEY_HINT) is None:
                    continue
                rows.append(row)
    except OSError:
        return []
    return rows


def read_progress_from_stdout(run: Path) -> dict:
    """Synthesise a progress payload from the trainer's stdout if a job is in flight.

    Falls back gracefully to {} when job_info.md / stdout_log is missing.
    """
    info = read_job_info(run)
    stdout_p = info.get("stdout_log")
    if not stdout_p:
        return {"rows": 0, "columns": [], "series": {}}
    rows = _scan_stdout_metric_lines(Path(stdout_p))
    if not rows:
        return {"rows": 0, "columns": [], "series": {}}
    cols: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    # Build series with None for missing keys per row (Chart.js ignores nulls).
    series: dict[str, list] = {c: [] for c in cols}
    for row in rows:
        for c in cols:
            series[c].append(row.get(c))
    return {"rows": len(rows), "columns": cols, "series": series}


def read_progress(run: Path) -> dict:
    """Combined progress source — prefer stdout (real-time, multi-row),
    fall back to progress.csv (only useful when the agent has been writing it)."""
    via_stdout = read_progress_from_stdout(run)
    if via_stdout["rows"] > 0:
        return via_stdout
    return read_progress_csv(run)


_ANOMALY_LINE_RE = re.compile(
    r"^-\s*(?:\[(?P<ts>[^\]]+)\]\s*[—-]\s*)?(?P<body>.+?)\s*$"
)
_SEVERITY_PATTERNS = [
    (re.compile(r"\bOOM\b|out of memory", re.I), "critical"),
    (re.compile(r"\bNaN\b|\bInf\b|is nan", re.I), "critical"),
    (re.compile(r"\bdivergence\b|entropy.?explos|policy diverging", re.I), "critical"),
    (re.compile(r"\bNCCL\b|timeout waiting", re.I), "warning"),
    (re.compile(r"\bvllm\b.*?error", re.I), "warning"),
    (re.compile(r"val.?regress|preempted|cancelled at", re.I), "warning"),
]


def _classify(body: str) -> str:
    for rx, sev in _SEVERITY_PATTERNS:
        if rx.search(body):
            return sev
    return "info"


def read_anomalies(run: Path) -> list[dict]:
    """Read workspace/logs/anomalies.md → [{ts, body, severity}]."""
    p = run / "workspace" / "logs" / "anomalies.md"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        m = _ANOMALY_LINE_RE.match(line.strip())
        if not m or not m.group("body"):
            continue
        body = m.group("body").strip()
        if body.startswith("#") or len(body) < 3:
            continue
        out.append({"timestamp": m.group("ts") or "",
                    "body": body,
                    "severity": _classify(body)})
    return out


def read_job_info(run: Path) -> dict:
    """Read workspace/job/job_info.md (key: value markdown) → dict."""
    p = run / "workspace" / "job" / "job_info.md"
    if not p.exists():
        return {}
    out: dict = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^-?\s*(\w[\w_]*)\s*:\s*(.+?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def read_job_status(run: Path) -> dict:
    """Read workspace/job/job_status.md → {status, ...}."""
    p = run / "workspace" / "job" / "job_status.md"
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8")
    out: dict = {"raw": text}
    m = re.search(r"^##\s*Status\s*\n(.+?)$", text, re.MULTILINE)
    if m:
        raw = m.group(1).strip().split("|")[0].strip()
        # Strip simple markdown emphasis the human report sometimes uses
        # (e.g. **success** in summarize.md's success template).
        raw = re.sub(r"^[*_`]+|[*_`]+$", "", raw).strip()
        out["status"] = raw
    # Pull a handful of standard fields if present.
    for key in ("final_step", "final_epoch", "last_checkpoint",
                "final_loss", "final_reward"):
        m = re.search(rf"-\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text, re.MULTILINE)
        if m:
            out[key] = m.group(1).strip()
    return out


def read_job_log_tail(run: Path, since_byte: int = 0,
                      max_bytes: int = 200_000) -> dict:
    """Incrementally read job_log.md from `since_byte`, capped by max_bytes.

    Fall back to the slurm stdout (from job_info.md `stdout_log`) when
    workspace/logs/job_log.md is missing or empty — useful when an agent isn't
    actively tailing into the workspace log file (e.g., a 4-day background sbatch).
    """
    p = run / "workspace" / "logs" / "job_log.md"
    if p.exists() and p.stat().st_size > 0:
        size = p.stat().st_size
        if since_byte >= size:
            return {"size": size, "since": since_byte, "content": ""}
        start = max(since_byte, size - max_bytes)
        with open(p, "rb") as f:
            f.seek(start)
            chunk = f.read(size - start)
        return {"size": size, "since": start,
                "content": chunk.decode("utf-8", errors="replace")}

    # Fall back to the slurm stdout the job is actively writing.
    info = read_job_info(run)
    stdout_p = info.get("stdout_log")
    if not stdout_p:
        return {"size": 0, "since": since_byte, "content": ""}
    sp = Path(stdout_p)
    if not sp.exists():
        return {"size": 0, "since": since_byte, "content": ""}
    size = sp.stat().st_size
    if since_byte >= size:
        return {"size": size, "since": since_byte, "content": ""}
    start = max(since_byte, size - max_bytes)
    try:
        with open(sp, "rb") as f:
            f.seek(start)
            chunk = f.read(size - start)
    except OSError:
        return {"size": 0, "since": since_byte, "content": ""}
    return {"size": size, "since": start,
            "content": chunk.decode("utf-8", errors="replace")}


def read_summary(run: Path) -> str:
    p = run / "workspace" / "summary" / "summary.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    p2 = run / "workspace" / "final_report.md"
    if p2.exists():
        return p2.read_text(encoding="utf-8")
    return ""
