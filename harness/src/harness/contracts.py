"""Workspace-contract enforcement.

Extracts `workspace/...` paths from each transition's Deliverables block and
verifies those files actually exist before allowing `transition_to` to succeed.

Design choice:
  - Paths are extracted per-transition from the STATE FILE's own Deliverables
    prose (backtick-wrapped `workspace/foo.md` references), NOT from a
    hardcoded table. That keeps the runtime honest to the spec — if you rename
    a deliverable path in `states/foo.md`, the enforcement follows.
  - When a deliverable description doesn't cite a `workspace/...` path (some
    states use free-form key/value blurbs), we fall back to the CLAUDE.md
    canonical file for that source state.
"""

from __future__ import annotations

import re
from pathlib import Path

from harness.context import Transition


# `` `workspace/foo/bar.md` `` — the pattern the state files use.
_PATH_RE = re.compile(r"`(workspace/[A-Za-z0-9_\-./]+)`")


# Fallback table: source-state → files it must produce IF its explicit
# transition prose doesn't cite a backtick path. Sourced from the canonical
# list in CLAUDE.md's `Workspace as inter-state contract` section.
_FALLBACK_DELIVERABLES: dict[str, list[str]] = {
    "intake":              ["workspace/intake/training_intent.md"],
    "locate_recipe":       ["workspace/recipe/recipe.md"],
    "configure_algorithm": ["workspace/algorithm/algorithm_config.md"],
    "prepare_data":        ["workspace/dataset/dataset.md"],
    "generate_preprocess": ["workspace/dataset/dataset.md"],
    "configure_reward":    ["workspace/reward/reward_config.md"],
    "select_compute":      ["workspace/compute/compute_choice.md"],
    "provision_env":       ["workspace/env/env_state.md"],
    "sanity_rollout":      ["workspace/sanity/sanity_report.md"],
    "launch_training":     ["workspace/job/job_info.md"],
    "monitor_training":    ["workspace/job/job_status.md"],
    "summarize":           ["workspace/summary/summary.md"],
    "reflect":             [],  # branch-dependent; parsed from prose
    "run_generate":        [],  # branch-dependent
    "run_eval":            [],  # branch-dependent
    "finalize":            [],  # terminal
}


def declared_paths(transition: Transition) -> list[str]:
    """Return every `workspace/...` path cited (in backticks) by this transition's Deliverables."""
    paths: list[str] = []
    seen: set[str] = set()
    for d in transition.deliverables:
        for m in _PATH_RE.finditer(d.description):
            p = m.group(1)
            if p not in seen:
                seen.add(p)
                paths.append(p)
    return paths


def required_paths(source_state: str, transition: Transition) -> list[str]:
    """The paths the runtime must see on disk before this transition may fire.

    Prefers backtick-cited paths from the transition prose; falls back to the
    canonical CLAUDE.md table for the source state when the prose has none.
    """
    cited = declared_paths(transition)
    if cited:
        return cited
    return list(_FALLBACK_DELIVERABLES.get(source_state, []))


def missing_deliverables(
    workspace: Path,
    source_state: str,
    transition: Transition,
) -> list[str]:
    """Names of expected workspace files that are missing on disk.

    `workspace` is the absolute path to `runs/<run_id>/workspace/`. Path
    prefixes like `workspace/intake/x.md` map to `<workspace>/intake/x.md`.
    """
    missing: list[str] = []
    for p in required_paths(source_state, transition):
        rel = p.removeprefix("workspace/")
        if not (workspace / rel).exists():
            missing.append(p)
    return missing


def contract_violation_message(
    source_state: str,
    target: str,
    missing: list[str],
) -> str:
    """A tool_result message the model can act on."""
    if not missing:
        return ""
    return (
        f"transition_to(next_state={target!r}) rejected — the {source_state} "
        f"state's Deliverables for that branch are not on disk.\n\n"
        f"Missing: {', '.join(missing)}\n\n"
        f"Use the `write` tool to lay these files down, then call transition_to "
        f"again. The paths above are absolute keys under workspace/ — resolve "
        f"them by concatenating the WORKSPACE path from your initial user message."
    )
