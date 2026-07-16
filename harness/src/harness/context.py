"""Context assembly — parses `states/<name>.md`, resolves skill refs, builds
the system prompt for one state.

CLAUDE.md says every state file follows this schema:
  ## Description → ## Skills → ## Hand-off Points → ## Next States

We parse:
  - `## Skills` → bulleted list of `skills/<dir>` paths
  - `## Next States` → per `### <state_name>` block: Condition, optional
                       `**Loop:** max_iterations: N` back-edge, and
                       `**Deliverables:**` bullet list of `<key>: <description>`

Skills are loaded by concatenating every `.md` file in the referenced dir —
robust across the mixed conventions in the repo (`default.md`, `SKILL.md`,
multi-file skills like `skills/builtin-tools/`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


_SKILL_LINE = re.compile(r"^\s*-\s*(skills/[A-Za-z0-9_\-./]+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Deliverable:
    """One `- <key>: <description>` bullet inside a transition's Deliverables block."""

    key: str
    description: str


@dataclass(frozen=True)
class Transition:
    """One `### <target>` entry under `## Next States`."""

    target: str
    condition: str  # verbatim text (may include markdown emphasis)
    deliverables: list[Deliverable] = field(default_factory=list)
    loop_max_iterations: int | None = None  # None unless this is a declared back-edge


@dataclass(frozen=True)
class StateContext:
    """Everything needed to run one state through the loop."""

    state_name: str
    system_prompt: str
    next_states: list[str]  # names allowed under ## Next States (order-preserved)
    is_terminal: bool  # true when ## Next States is absent or empty
    skill_refs: list[str]  # e.g. ["skills/intake", "skills/global"]
    transitions: list[Transition] = field(default_factory=list)
    # Anthropic-style cacheable blocks: [{"type": "text", "text": "..."}, ...].
    # First 4 get `cache_control` applied by the anthropic backend. Ordered so
    # the most-cacheable content (repo CLAUDE.md, state.md) comes first.
    system_blocks: list[dict[str, str]] = field(default_factory=list)


class ContextError(Exception):
    """Malformed state file or missing skill."""


# ── low-level parsers ───────────────────────────────────────────────────────


def _extract_section(md: str, heading: str) -> str | None:
    """Return the body of a `## heading` section, or None if absent.

    Body ends at the next `## ` heading or EOF.
    """
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$", re.MULTILINE | re.IGNORECASE
    )
    m = pattern.search(md)
    if m is None:
        return None
    start = m.end()
    # Find the next ##  at line start
    next_h = re.search(r"^##\s+", md[start:], re.MULTILINE)
    end = start + next_h.start() if next_h else len(md)
    return md[start:end].strip()


def parse_next_states(md: str) -> list[str]:
    """Extract `### <name>` sub-headings under `## Next States`.

    Returns [] when the section is absent (terminal state) or when it has
    only prose without `###` targets.
    """
    section = _extract_section(md, "Next States")
    if section is None:
        return []
    return re.findall(r"^###\s+([A-Za-z0-9_\-]+)\s*$", section, re.MULTILINE)


# `**<Name>:**` at line start marks a labelled sub-block.
_LABEL_TERMINATOR = r"(?=^\*\*[A-Za-z][A-Za-z ]*:\*\*|\Z)"
_LOOP_LINE = re.compile(
    r"^\*\*Loop:\*\*\s+max_iterations:\s*(\d+)\s*$", re.MULTILINE
)
_BULLET_LINE = re.compile(
    r"^-\s+([A-Za-z0-9_][A-Za-z0-9_\-]*):\s+(.+?)\s*$", re.MULTILINE
)


def _extract_labelled_block(body: str, label: str) -> str:
    """Return the body text after `**<label>:**` up to the next labelled block."""
    m = re.search(
        rf"^\*\*{re.escape(label)}:\*\*\s*(.*?){_LABEL_TERMINATOR}",
        body,
        re.MULTILINE | re.DOTALL,
    )
    return m.group(1).strip() if m else ""


def parse_transitions(md: str) -> list[Transition]:
    """Extract every `### <target>` block under `## Next States` with metadata."""
    section = _extract_section(md, "Next States")
    if section is None:
        return []

    parts = re.split(
        r"^###\s+([A-Za-z0-9_\-]+)\s*$", section, flags=re.MULTILINE
    )
    # parts[0] is prelude, then alternating [name, body, name, body, ...]
    transitions: list[Transition] = []
    for i in range(1, len(parts), 2):
        target = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""

        condition = _extract_labelled_block(body, "Condition")
        loop_m = _LOOP_LINE.search(body)
        loop_max = int(loop_m.group(1)) if loop_m else None

        delivs_text = _extract_labelled_block(body, "Deliverables")
        deliverables = [
            Deliverable(key=m.group(1), description=m.group(2).strip())
            for m in _BULLET_LINE.finditer(delivs_text)
        ]

        transitions.append(
            Transition(
                target=target,
                condition=condition,
                deliverables=deliverables,
                loop_max_iterations=loop_max,
            )
        )
    return transitions


def parse_skill_refs(md: str) -> list[str]:
    """Extract `- skills/<dir>` bullets under `## Skills`."""
    section = _extract_section(md, "Skills")
    if section is None:
        return []
    return [m.group(1).rstrip("/") for m in _SKILL_LINE.finditer(section)]


# ── loaders ─────────────────────────────────────────────────────────────────


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        raise ContextError(f"failed to read {path}: {e}") from e


def _load_skill_dir(workdir: Path, skill_ref: str) -> str:
    """Concat every `.md` file directly inside `<workdir>/<skill_ref>/`.

    `default.md` is loaded first (canonical entry point per CLAUDE.md);
    other `.md` siblings follow in alphabetical order.
    """
    skill_dir = (workdir / skill_ref).resolve()
    if not skill_dir.is_dir():
        raise ContextError(f"skill dir missing: {skill_dir}")

    parts: list[str] = []
    all_mds = sorted(p for p in skill_dir.glob("*.md") if p.is_file())
    if not all_mds:
        raise ContextError(f"skill dir has no *.md files: {skill_dir}")

    priority = [p for p in all_mds if p.name == "default.md"]
    others = [p for p in all_mds if p.name != "default.md"]
    for path in priority + others:
        parts.append(f"### {skill_ref}/{path.name}\n\n{_read(path)}")
    return "\n\n".join(parts)


def load_state_context(workdir: Path, state_name: str) -> StateContext:
    """Parse `states/<name>.md`, resolve skills, assemble a system prompt."""
    state_path = workdir / "states" / f"{state_name}.md"
    if not state_path.is_file():
        raise ContextError(f"state file not found: {state_path}")

    state_md = _read(state_path)
    transitions = parse_transitions(state_md)
    next_states = [t.target for t in transitions] or parse_next_states(state_md)
    skill_refs = parse_skill_refs(state_md)

    parts: list[str] = []
    parts.append(
        "You are the verl-harness runtime agent driving one FSM state.\n"
        "Follow the state's ## Description exactly. When its ## Next States\n"
        "deliverables are complete, call the `transition_to` tool with the\n"
        "chosen next-state name and STOP — do not continue producing output."
    )

    claude_md = workdir / "CLAUDE.md"
    if claude_md.is_file():
        parts.append(f"## CLAUDE.md (repo guidance)\n\n{_read(claude_md)}")

    parts.append(f"## states/{state_name}.md\n\n{state_md}")

    for ref in skill_refs:
        parts.append(f"## {ref} (skill)\n\n{_load_skill_dir(workdir, ref)}")

    if next_states:
        parts.append(
            "## Transition rules\n\n"
            "When the state's deliverables are satisfied, invoke the special "
            "tool `transition_to(next_state=...)`. Legal targets are: "
            + ", ".join(f"`{n}`" for n in next_states)
            + ". Do not invoke it before the deliverables are complete."
        )
    else:
        parts.append(
            "## Terminal state\n\n"
            "This state has no ## Next States block — it is terminal. "
            "When work is done, end your turn (do not call transition_to)."
        )

    # Cacheable blocks — each is a coherent chunk that doesn't change between
    # states in the same run. Order matters: first 4 blocks get cache_control
    # on the anthropic wire.
    blocks: list[dict[str, str]] = [
        {"type": "text", "text": parts[0]},  # runtime preamble
    ]
    if claude_md.is_file():
        blocks.append({"type": "text", "text": f"## CLAUDE.md\n\n{_read(claude_md)}"})
    blocks.append({"type": "text", "text": f"## states/{state_name}.md\n\n{state_md}"})
    if skill_refs:
        skill_bundle = "\n\n---\n\n".join(
            f"## {ref}\n\n{_load_skill_dir(workdir, ref)}"
            for ref in skill_refs
        )
        blocks.append({"type": "text", "text": skill_bundle})
    # Transition rules go last (small, per-state — not worth caching).
    if next_states:
        blocks.append(
            {
                "type": "text",
                "text": (
                    "## Transition rules\n\n"
                    "When the state's deliverables are satisfied, invoke "
                    "`transition_to(next_state=...)`. Legal targets: "
                    + ", ".join(f"`{n}`" for n in next_states)
                ),
            }
        )
    else:
        blocks.append(
            {
                "type": "text",
                "text": (
                    "## Terminal state\n\n"
                    "This state has no ## Next States. When the work is done, "
                    "end your turn."
                ),
            }
        )

    return StateContext(
        state_name=state_name,
        system_prompt="\n\n---\n\n".join(parts),
        next_states=next_states,
        is_terminal=not next_states,
        skill_refs=skill_refs,
        transitions=transitions,
        system_blocks=blocks,
    )


# ── initial user message ────────────────────────────────────────────────────


def _list_workspace(workspace: Path, max_entries: int = 200) -> str:
    if not workspace.exists():
        return "(workspace dir does not exist yet — you will create it)"
    entries = sorted(workspace.rglob("*"))
    if not entries:
        return "(empty)"
    lines: list[str] = []
    for i, p in enumerate(entries):
        if i >= max_entries:
            lines.append(f"... {len(entries) - max_entries} more entries")
            break
        rel = p.relative_to(workspace)
        marker = "/" if p.is_dir() else ""
        lines.append(f"  {rel}{marker}")
    return "\n".join(lines)


def render_workspace_snapshot(
    workspace: Path,
    *,
    max_chars_per_file: int = 800,
    max_files_inlined: int = 20,
) -> str:
    """Directory tree + inlined content of each `.md` file (first ~800 chars).

    Used to seed each state's initial user message so the agent sees what
    upstream states have written without re-parsing every file. Non-md files
    only appear in the tree listing.
    """
    tree = _list_workspace(workspace)
    if not workspace.exists():
        return tree

    md_files = sorted(p for p in workspace.rglob("*.md") if p.is_file())
    if not md_files:
        return f"Directory tree:\n{tree}"

    inlined: list[str] = []
    for i, p in enumerate(md_files):
        if i >= max_files_inlined:
            inlined.append(
                f"[... {len(md_files) - max_files_inlined} more .md files not inlined]"
            )
            break
        rel = p.relative_to(workspace)
        try:
            body = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(body) > max_chars_per_file:
            body = (
                body[:max_chars_per_file]
                + f"\n[... {len(body) - max_chars_per_file} more bytes, use the `read` tool for full contents]"
            )
        # Use ~~~ fences instead of ``` to avoid clashing with markdown fences
        # that appear inside the file contents themselves.
        inlined.append(f"### workspace/{rel}\n~~~\n{body}\n~~~")

    return (
        f"Directory tree:\n{tree}\n\n"
        f"Inlined deliverable contents:\n\n" + "\n\n".join(inlined)
    )


def build_initial_user_message(
    *,
    goal: str,
    run_id: str,
    workdir: Path,
    verl_root: Path | None,
    workspace: Path,
    state_name: str,
    include_snapshot: bool = True,
) -> str:
    """Seed the fresh conversation for one state.

    When `include_snapshot` is True, inline the content of every `.md` file
    under `workspace/` (truncated per-file). That way each state re-enters
    with a compressed view of upstream deliverables, and the runtime does
    NOT need to carry the entire prior conversation forward.
    """
    if include_snapshot:
        snapshot = render_workspace_snapshot(workspace)
    else:
        snapshot = _list_workspace(workspace)

    return (
        f"GOAL: {goal}\n"
        f"RUN_ID: {run_id}\n"
        f"WORKDIR: {workdir}\n"
        f"VERL_ROOT: {verl_root if verl_root else '(unset — resolve per state instructions)'}\n"
        f"WORKSPACE: {workspace}\n"
        f"STATE: {state_name}\n\n"
        f"Existing workspace contents:\n{snapshot}\n\n"
        f"Drive the {state_name} state now. When its deliverables are complete, "
        f"call transition_to and stop."
    )
