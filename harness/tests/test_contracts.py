"""Contract enforcement — path extraction + missing-file detection."""

from __future__ import annotations

from pathlib import Path

from harness.context import Deliverable, Transition, parse_transitions
from harness.contracts import (
    contract_violation_message,
    declared_paths,
    missing_deliverables,
    required_paths,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


# ── path extraction ────────────────────────────────────────────────────────

def test_declared_paths_from_backtick_workspace_refs() -> None:
    t = Transition(
        target="reflect",
        condition="c",
        deliverables=[
            Deliverable(
                key="summary",
                description="`workspace/summary/summary.md` — the run report.",
            )
        ],
    )
    assert declared_paths(t) == ["workspace/summary/summary.md"]


def test_declared_paths_dedupes() -> None:
    t = Transition(
        target="finalize",
        condition="c",
        deliverables=[
            Deliverable(key="a", description="`workspace/foo.md` bar `workspace/foo.md`"),
            Deliverable(key="b", description="also `workspace/bar.md`"),
        ],
    )
    assert declared_paths(t) == ["workspace/foo.md", "workspace/bar.md"]


def test_required_paths_falls_back_to_canonical() -> None:
    t = Transition(
        target="locate_recipe",
        condition="c",
        deliverables=[
            Deliverable(
                key="training_intent",
                description="The structured training spec — no path cited here.",
            )
        ],
    )
    assert required_paths("intake", t) == ["workspace/intake/training_intent.md"]


def test_required_paths_prefers_cited_when_present() -> None:
    t = Transition(
        target="finalize",
        condition="c",
        deliverables=[
            Deliverable(
                key="algorithm_unsupported",
                description="A `workspace/algorithm/algorithm_unsupported.md` recording ...",
            )
        ],
    )
    # cited path wins over the canonical algorithm_config.md
    assert required_paths("configure_algorithm", t) == [
        "workspace/algorithm/algorithm_unsupported.md"
    ]


# ── against the real states/ tree ─────────────────────────────────────────

def test_reflect_configure_algo_edge_cites_refinement_plan() -> None:
    md = (REPO_ROOT / "states" / "reflect.md").read_text()
    txns = {t.target: t for t in parse_transitions(md)}
    assert declared_paths(txns["configure_algorithm"]) == [
        "workspace/reflect/refinement_plan.md"
    ]
    assert declared_paths(txns["finalize"]) == [
        "workspace/reflect/reflect_report.md"
    ]


def test_summarize_transitions_cite_summary_md() -> None:
    md = (REPO_ROOT / "states" / "summarize.md").read_text()
    txns = parse_transitions(md)
    for t in txns:
        assert "workspace/summary/summary.md" in declared_paths(t)


def test_intake_locate_recipe_uses_cited_verl_root_txt() -> None:
    """intake→locate_recipe backticks verl_root.txt in its Deliverables prose,
    so the runtime enforces that path (not the canonical training_intent.md).

    Gap note: the state's ## Description tells the model to write
    training_intent.md, but the transition's Deliverables prose doesn't cite
    it in backticks. Downstream states catch a missing training_intent.md via
    THEIR own preconditions.
    """
    md = (REPO_ROOT / "states" / "intake.md").read_text()
    lr = next(t for t in parse_transitions(md) if t.target == "locate_recipe")
    assert declared_paths(lr) == ["workspace/intake/verl_root.txt"]
    assert required_paths("intake", lr) == ["workspace/intake/verl_root.txt"]


# ── missing_deliverables ──────────────────────────────────────────────────

def test_missing_deliverables_flags_absent_files(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    t = Transition(
        target="locate_recipe",
        condition="c",
        deliverables=[
            Deliverable(
                key="training_intent",
                description="written as `workspace/intake/training_intent.md`",
            )
        ],
    )
    assert missing_deliverables(workspace, "intake", t) == [
        "workspace/intake/training_intent.md"
    ]


def test_missing_deliverables_empty_when_present(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    (workspace / "intake").mkdir(parents=True)
    (workspace / "intake" / "training_intent.md").write_text("done")
    t = Transition(
        target="locate_recipe",
        condition="c",
        deliverables=[
            Deliverable(
                key="training_intent",
                description="`workspace/intake/training_intent.md`",
            )
        ],
    )
    assert missing_deliverables(workspace, "intake", t) == []


def test_missing_deliverables_via_canonical(tmp_path: Path) -> None:
    """A prose-only Deliverables block should still gate on the canonical file."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    t = Transition(
        target="configure_algorithm",
        condition="c",
        deliverables=[Deliverable(key="recipe", description="the recipe.md")],
    )
    assert missing_deliverables(workspace, "locate_recipe", t) == [
        "workspace/recipe/recipe.md"
    ]


def test_violation_message_lists_files() -> None:
    msg = contract_violation_message(
        "intake",
        "locate_recipe",
        ["workspace/intake/training_intent.md"],
    )
    assert "training_intent.md" in msg
    assert "write" in msg  # tells the model what to do
    assert "'locate_recipe'" in msg
