#!/usr/bin/env python3
"""Validate the markdown FSM and its cross-state contracts.

The state files are the authoritative FSM definition.  README.md,
task-overview.md, and the dashboard are views over that definition; this
validator deliberately does not infer transitions from those views.

Cycles are allowed only when explicitly declared: the transition that closes
the cycle must carry a `**Loop:** max_iterations: <n>` line between its
`**Condition:**` and `**Deliverables:**` blocks. The graph minus declared
loop edges must be acyclic and still drain every state into `finalize`.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict, deque
from pathlib import Path


H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
H3_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
FINAL_INPUT_RE = re.compile(r"^-\s+`(?P<name>[a-z][a-z0-9_]*)`\s+—\s+`(?P<path>workspace/[^`]+)`", re.MULTILINE)


def split_h2(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    matches = list(H2_RE.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1).strip()] = text[match.end():end].strip()
    return sections


def parse_transitions(block: str) -> list[dict]:
    transitions: list[dict] = []
    matches = list(H3_RE.finditer(block))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(block)
        body = block[match.end():end]
        condition = re.search(
            r"\*\*Condition:\*\*\s*(.+?)(?=\n\n\*\*(?:Loop|Deliverables):\*\*|$)",
            body,
            re.DOTALL,
        )
        deliverables_block = re.search(r"\*\*Deliverables:\*\*\s*(.+)$", body, re.DOTALL)
        loop_heading = re.search(r"\*\*Loop:\*\*", body)
        loop_bound = re.search(r"\*\*Loop:\*\*\s*max_iterations:\s*([1-9][0-9]*)\b", body)
        deliverables: list[tuple[str, str]] = []
        if deliverables_block:
            for line in deliverables_block.group(1).splitlines():
                item = re.match(r"^-\s*([a-z][a-z0-9_]*):\s*(.+)$", line.strip())
                if item:
                    deliverables.append((item.group(1), item.group(2).strip()))
        transitions.append(
            {
                "target": match.group(1).strip(),
                "condition": condition.group(1).strip() if condition else "",
                "deliverables": deliverables,
                "has_deliverables_heading": deliverables_block is not None,
                "has_loop_heading": loop_heading is not None,
                "loop_bound": int(loop_bound.group(1)) if loop_bound else None,
            }
        )
    return transitions


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    states_dir = root / "states"
    state_paths = sorted(states_dir.glob("*.md"))
    if not state_paths:
        return ["states/: no state files found"]

    states: dict[str, dict] = {}
    for path in state_paths:
        text = path.read_text(encoding="utf-8")
        sections = split_h2(text)
        title = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
        if not title or title.group(1).strip() != path.stem:
            errors.append(f"{path.relative_to(root)}: H1 must equal `{path.stem}`")
        for required in ("Description", "Skills", "Hand-off Points"):
            if required not in sections:
                errors.append(f"{path.relative_to(root)}: missing `## {required}`")
        transitions = parse_transitions(sections.get("Next States", ""))
        states[path.stem] = {"path": path, "sections": sections, "transitions": transitions}

        for raw_skill in re.findall(r"^-\s+(skills/[A-Za-z0-9_/-]+)", sections.get("Skills", ""), re.MULTILINE):
            skill = raw_skill.split("#", 1)[0].strip().rstrip("/")
            if not (root / skill).is_dir():
                errors.append(f"{path.relative_to(root)}: skill directory does not exist: `{skill}`")

        if "Next States" in sections and not transitions:
            errors.append(f"{path.relative_to(root)}: `## Next States` has no `### <state>` transitions")
        for transition in transitions:
            label = f"{path.relative_to(root)} -> {transition['target']}"
            if not transition["condition"]:
                errors.append(f"{label}: missing non-empty `**Condition:**`")
            if not transition["has_deliverables_heading"]:
                errors.append(f"{label}: missing `**Deliverables:**`")
            elif not transition["deliverables"]:
                errors.append(f"{label}: deliverables block has no `- name: description` item")
            if transition["has_loop_heading"] and transition["loop_bound"] is None:
                errors.append(f"{label}: `**Loop:**` must declare `max_iterations: <positive integer>`")

    state_names = set(states)
    for name, state in states.items():
        for transition in state["transitions"]:
            if transition["target"] not in state_names:
                errors.append(f"states/{name}.md: transition target does not exist: `{transition['target']}`")

    loop_edges = {
        (name, transition["target"])
        for name, state in states.items()
        for transition in state["transitions"]
        if transition["loop_bound"] is not None
    }

    overview = (root / "task-overview.md").read_text(encoding="utf-8")
    overview_sections = split_h2(overview)
    starting_raw = overview_sections.get("Starting State", "").splitlines()
    starting = Path(starting_raw[0].lstrip("- ").strip()).stem if starting_raw else ""
    if not starting or starting not in state_names:
        errors.append("task-overview.md: `## Starting State` must name an existing state")

    if starting in state_names:
        reachable = {starting}
        queue = deque([starting])
        while queue:
            current = queue.popleft()
            for transition in states[current]["transitions"]:
                target = transition["target"]
                if target in state_names and target not in reachable:
                    reachable.add(target)
                    queue.append(target)
        for name in sorted(state_names - reachable):
            errors.append(f"states/{name}.md: state is unreachable from `{starting}`")

    terminals = {name for name, state in states.items() if not state["transitions"]}
    if terminals != {"finalize"}:
        errors.append(f"FSM must have exactly one terminal state `finalize`; found {sorted(terminals)}")

    # Terminal convergence must hold on the loop-free subgraph: once every
    # declared loop bound is exhausted, the remaining edges must still drain
    # every state into `finalize`.
    reverse: dict[str, set[str]] = defaultdict(set)
    for name, state in states.items():
        for transition in state["transitions"]:
            if (name, transition["target"]) in loop_edges:
                continue
            reverse[transition["target"]].add(name)
    can_finish = {"finalize"} if "finalize" in state_names else set()
    queue = deque(can_finish)
    while queue:
        current = queue.popleft()
        for source in reverse[current]:
            if source not in can_finish:
                can_finish.add(source)
                queue.append(source)
    for name in sorted(state_names - can_finish):
        errors.append(f"states/{name}.md: no loop-free path reaches `finalize`")

    # Every cycle must be broken by an explicitly declared, bounded loop edge
    # (`**Loop:** max_iterations: <n>` on the transition). Kahn's algorithm on
    # the non-loop subgraph: any state left over sits on an undeclared cycle.
    indegree = {name: 0 for name in state_names}
    forward: dict[str, set[str]] = defaultdict(set)
    for name, state in states.items():
        for transition in state["transitions"]:
            target = transition["target"]
            if target in state_names and (name, target) not in loop_edges and target not in forward[name]:
                forward[name].add(target)
                indegree[target] += 1
    queue = deque(name for name in sorted(state_names) if indegree[name] == 0)
    acyclic = set()
    while queue:
        current = queue.popleft()
        acyclic.add(current)
        for target in sorted(forward[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    cyclic = state_names - acyclic
    # Kahn's leftover also contains states merely downstream of a cycle;
    # repeatedly strip states with no outgoing edge back into the leftover so
    # only actual cycle members are reported.
    changed = True
    while changed:
        changed = False
        for name in sorted(cyclic):
            if not any(target in cyclic for target in forward[name]):
                cyclic.discard(name)
                changed = True
    for name in sorted(cyclic):
        errors.append(
            f"states/{name}.md: part of an undeclared cycle; mark the intended "
            f"back-edge with `**Loop:** max_iterations: <n>`"
        )

    # A declared loop edge must actually close a cycle, so `**Loop:**` markers
    # cannot rot into no-ops: the target must be able to reach the source.
    for source, target in sorted(loop_edges):
        if target not in state_names:
            continue
        seen = {target}
        queue = deque([target])
        closes = False
        while queue:
            current = queue.popleft()
            if current == source:
                closes = True
                break
            for transition in states[current]["transitions"]:
                nxt = transition["target"]
                if nxt in state_names and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        if not closes:
            errors.append(
                f"states/{source}.md -> {target}: declared loop edge does not close a cycle"
            )

    final_sections = states.get("finalize", {}).get("sections", {})
    terminal_inputs = {
        match.group("name"): match.group("path")
        for match in FINAL_INPUT_RE.finditer(final_sections.get("Terminal Inputs", ""))
    }
    if not terminal_inputs:
        errors.append("states/finalize.md: `## Terminal Inputs` has no canonical input entries")
    incoming: dict[str, list[str]] = defaultdict(list)
    for name, state in states.items():
        for transition in state["transitions"]:
            if transition["target"] == "finalize":
                for deliverable, description in transition["deliverables"]:
                    incoming[deliverable].append(f"states/{name}.md")
                    expected_path = terminal_inputs.get(deliverable)
                    mentioned_paths = re.findall(r"`(workspace/[^`]+)`", description)
                    if expected_path and expected_path not in mentioned_paths:
                        found = mentioned_paths if mentioned_paths else "no canonical path"
                        errors.append(
                            f"states/{name}.md -> finalize: `{deliverable}` must name `{expected_path}`; "
                            f"found {found}"
                        )
    for deliverable, sources in sorted(incoming.items()):
        if deliverable not in terminal_inputs:
            errors.append(
                f"states/finalize.md: terminal input `{deliverable}` from {', '.join(sources)} is not declared"
            )
    for deliverable in sorted(set(terminal_inputs) - set(incoming)):
        errors.append(f"states/finalize.md: declared terminal input `{deliverable}` has no incoming transition")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="Harness root (default: current directory)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    errors = validate(root)
    if errors:
        print(f"FAIL: {len(errors)} harness contract error(s)", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    count = len(list((root / "states").glob("*.md")))
    print(f"OK: {count} states; schema, graph, loops, skills, and terminal contracts valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
