"""Render helpers — Mermaid FSM graph, compiled markdown views."""
from __future__ import annotations

from .parser import Harness, State

OVERVIEW_NODE = "__overview__"


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

    for name, st in h.states.items():
        for tr in st.transitions:
            cond = (tr.condition or "").split("\n")[0]
            if len(cond) > 60:
                cond = cond[:57] + "…"
            # Mermaid edge-label sanitisation:
            #   - "  → '   (double quotes close the label)
            #   - |  → /   (pipe is the label delimiter itself, e.g. `success | crashed`)
            #   - `  → ''  (backticks confuse Mermaid 11 strict parser)
            cond = (cond.replace('"', "'")
                        .replace('|', '/')
                        .replace('`', ''))
            if cond:
                lines.append(f'  {name} -->|"{cond}"| {tr.target}')
            else:
                lines.append(f'  {name} --> {tr.target}')

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
        parts.append("## Human Checkpoints")
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
