"""Render helpers — Mermaid FSM graph, compiled markdown views."""
from __future__ import annotations

import re

from .parser import Harness, State

OVERVIEW_NODE = "__overview__"

# Short error tags for edges that exit to a terminal state. First match wins,
# so order specific phrases before generic ones.
_ERROR_TAGS = [
    ("no first-class trainer", "no trainer"),
    ("gpu_cap", "over budget"),
    ("gpu budget", "over budget"),
    ("provisioning failed", "env failed"),
    ("sanity", "sanity fail"),
    ("launch command itself failed", "launch fail"),
    ("sbatch returned nonzero", "launch fail"),
    ("preempt", "preempt"),
    ("crash", "crash"),
    ("timed out", "crash"),
    ("unsupported", "halt"),
    ("fail", "fail"),
]


def _edge_label(source: str, cond: str, target: str, terminals: set[str]) -> tuple[str, bool]:
    """Return (label, dotted) for a transition.

    Keeps the top-down happy spine UNLABELED (the vertical order is self-evident);
    puts a short label only where the path genuinely branches — goal choice at
    intake, the unknown-dataset bounce, and error/halt exits to a terminal state.
    Error exits are dotted so they recede from the solid happy path.
    """
    c = cond.lower()
    m = re.search(r"goal:\s*([a-z_]+)", c)
    if m:
        return m.group(1), False                       # intake fan-out → goal name
    if target == "generate_preprocess" or "unknown" in c or "hf dataset id" in c:
        return "unknown data", False                   # dataset bounce
    if source == "reflect" and target == "configure_algorithm":
        return "refine", False                         # bounded refinement back-edge
    if target in terminals:
        if source in ("summarize", "reflect"):
            return "", False                           # normal end — part of the spine
        if ("succeed" in c or "is written" in c) and "crash" not in c and "fail" not in c:
            return "done", True                        # alternate happy end (generate/eval)
        for kw, tag in _ERROR_TAGS:
            if kw in c:
                return tag, True
        return "halt", True
    return "", False                                   # forward happy edge — clean spine


def harness_to_mermaid(h: Harness) -> str:
    """Render the harness FSM as a Mermaid flowchart string.

    Class assignments encode runtime status (live / visited / terminal /
    selected) which the frontend applies via Mermaid classDef.
    """
    lines = ["flowchart TD"]
    # Overview node — always present, sits above the start state.
    lines.append(f'  {OVERVIEW_NODE}["📋 task-overview.md"]')
    if h.starting_state:
        lines.append(f'  {OVERVIEW_NODE} -.start.-> {h.starting_state}')

    for name, st in h.states.items():
        label = name
        if st.is_terminal:
            label = f"⬛ {name}"
        lines.append(f'  {name}["{label}"]')

    terminals = {n for n, s in h.states.items() if s.is_terminal}
    for name, st in h.states.items():
        for tr in st.transitions:
            cond = (tr.condition or "").split("\n")[0]
            label, dotted = _edge_label(name, cond, tr.target, terminals)
            label = label.replace('"', "'").replace('|', '/').replace('`', '')
            if dotted:
                lines.append(f'  {name} -.->|"{label}"| {tr.target}' if label
                             else f'  {name} -.-> {tr.target}')
            else:
                lines.append(f'  {name} -->|"{label}"| {tr.target}' if label
                             else f'  {name} --> {tr.target}')

    # Class defs are applied client-side based on runtime state.
    # Pre-declared here so Mermaid doesn't choke on the classAssignments
    # the client adds.
    return "\n".join(lines)


def compile_state(h: Harness, st: State) -> str:
    """Compile a state file's content for rendering as nicely-formatted markdown."""
    parts = [f"# {st.name}", "", st.description.strip(), ""]
    if st.skills:
        parts.append("## Skills")
        parts.extend(f"- `{s}`" for s in st.skills)
        parts.append("")
    if st.human_checkpoints:
        # Canonical key is `Hand-off Points` (parser reads either name; this
        # writer emits the new name for consistency with state-file sources).
        parts.append("## Hand-off Points")
        parts.append(st.human_checkpoints)
        parts.append("")
    if st.transitions:
        parts.append("## Next States")
        for tr in st.transitions:
            parts.append(f"### → {tr.target}")
            if tr.condition:
                parts.append(f"**Condition.** {tr.condition}")
            if tr.deliverables:
                parts.append("")
                parts.append("**Deliverables.**")
                for name, desc in tr.deliverables:
                    parts.append(f"- **{name}** — {desc}")
            parts.append("")
    else:
        parts.append("_Terminal state._")
    return "\n".join(parts)


def compile_skill_folder(h: Harness, skill_path: str) -> str:
    files = h.read_skill_files(skill_path)
    if not files:
        return f"_No markdown files under `{skill_path}/`._"
    parts = [f"# {skill_path}", ""]
    for name, content in files:
        parts.append(f"## {name}")
        parts.append(content)
        parts.append("")
    return "\n".join(parts)


def compile_overview(h: Harness) -> str:
    return h.raw_overview
