"""context.py: state file parsing + skill assembly.

These tests use the REAL states/ + skills/ + CLAUDE.md from the repo so we
catch drift if the spec files change shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.context import (
    ContextError,
    build_initial_user_message,
    load_state_context,
    parse_next_states,
    parse_skill_refs,
    parse_transitions,
    render_workspace_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


# ── parsers ────────────────────────────────────────────────────────────────

def test_parse_next_states_intake_has_five() -> None:
    md = (REPO_ROOT / "states" / "intake.md").read_text()
    targets = parse_next_states(md)
    assert set(targets) == {
        "locate_recipe",
        "monitor_training",
        "launch_training",
        "run_generate",
        "run_eval",
    }


def test_parse_skill_refs_intake() -> None:
    md = (REPO_ROOT / "states" / "intake.md").read_text()
    refs = parse_skill_refs(md)
    assert set(refs) == {"skills/intake", "skills/builtin-tools", "skills/global"}


def test_parse_next_states_terminal_is_empty() -> None:
    md = (REPO_ROOT / "states" / "finalize.md").read_text()
    assert parse_next_states(md) == []


def test_parse_next_states_no_section() -> None:
    assert parse_next_states("# just a title\n\nsome prose\n") == []


# ── loader ─────────────────────────────────────────────────────────────────

def test_load_state_context_intake() -> None:
    ctx = load_state_context(REPO_ROOT, "intake")
    assert ctx.state_name == "intake"
    assert not ctx.is_terminal
    assert set(ctx.next_states) == {
        "locate_recipe",
        "monitor_training",
        "launch_training",
        "run_generate",
        "run_eval",
    }
    # System prompt embeds CLAUDE.md, the state file, and each skill.
    assert "CLAUDE.md" in ctx.system_prompt
    assert "## states/intake.md" in ctx.system_prompt
    assert "skills/intake" in ctx.system_prompt
    assert "skills/global" in ctx.system_prompt
    # scientific_principles.md is the real content of skills/global — verify
    # skill assembly actually pulled file bodies, not just headings.
    assert "scientific_principles.md" in ctx.system_prompt
    # transition rules with the enum spelled out
    assert "locate_recipe" in ctx.system_prompt


def test_load_state_context_finalize_is_terminal() -> None:
    ctx = load_state_context(REPO_ROOT, "finalize")
    assert ctx.is_terminal
    assert ctx.next_states == []
    assert "Terminal state" in ctx.system_prompt


def test_load_state_context_missing_file(tmp_path: Path) -> None:
    (tmp_path / "states").mkdir()
    with pytest.raises(ContextError, match="not found"):
        load_state_context(tmp_path, "nope")


def test_load_state_context_missing_skill(tmp_path: Path) -> None:
    (tmp_path / "states").mkdir()
    (tmp_path / "states" / "x.md").write_text(
        "## Description\n\nfoo\n\n## Skills\n\n- skills/nonexistent\n\n"
        "## Next States\n\n### y\n\n**Condition:** _\n**Deliverables:** _\n"
    )
    with pytest.raises(ContextError, match="skill dir missing"):
        load_state_context(tmp_path, "x")


# ── initial user message ───────────────────────────────────────────────────

def test_initial_user_message_shape(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    msg = build_initial_user_message(
        goal="train GRPO on gsm8k",
        run_id="R1",
        workdir=tmp_path,
        verl_root=Path("/opt/verl"),
        workspace=ws,
        state_name="intake",
    )
    assert "GOAL: train GRPO on gsm8k" in msg
    assert "RUN_ID: R1" in msg
    assert "/opt/verl" in msg
    assert "STATE: intake" in msg
    assert "workspace dir does not exist" in msg  # ws not yet created


# ── transition parser ─────────────────────────────────────────────────────

def test_parse_transitions_intake_has_five_with_conditions_and_deliverables() -> None:
    md = (REPO_ROOT / "states" / "intake.md").read_text()
    txns = parse_transitions(md)
    by_target = {t.target: t for t in txns}
    assert set(by_target) == {
        "locate_recipe",
        "monitor_training",
        "launch_training",
        "run_generate",
        "run_eval",
    }
    # Each has a non-empty condition and at least one deliverable
    for t in txns:
        assert t.condition, f"{t.target} missing Condition"
        assert t.deliverables, f"{t.target} missing Deliverables"
    # None of intake's edges are declared loops
    assert all(t.loop_max_iterations is None for t in txns)

    lr = by_target["locate_recipe"]
    assert "goal: train" in lr.condition
    deliv_keys = {d.key for d in lr.deliverables}
    assert "training_intent" in deliv_keys
    assert "verl_root" in deliv_keys


def test_parse_transitions_reflect_has_declared_loop() -> None:
    md = (REPO_ROOT / "states" / "reflect.md").read_text()
    txns = parse_transitions(md)
    by_target = {t.target: t for t in txns}
    assert set(by_target) == {"configure_algorithm", "finalize"}

    # The back-edge that closes the reflect loop must be declared.
    back_edge = by_target["configure_algorithm"]
    assert back_edge.loop_max_iterations == 3

    # The exit edge should NOT have a Loop marker.
    exit_edge = by_target["finalize"]
    assert exit_edge.loop_max_iterations is None


def test_parse_transitions_prepare_data_bounce() -> None:
    """prepare_data ⇄ generate_preprocess dataset bounce has max_iterations: 1."""
    md = (REPO_ROOT / "states" / "prepare_data.md").read_text()
    txns = parse_transitions(md)
    by_target = {t.target: t for t in txns}
    assert "generate_preprocess" in by_target
    assert by_target["generate_preprocess"].loop_max_iterations == 1


def test_parse_transitions_summarize_two_branches() -> None:
    md = (REPO_ROOT / "states" / "summarize.md").read_text()
    txns = parse_transitions(md)
    targets = [t.target for t in txns]
    assert targets == ["reflect", "finalize"]  # order preserved
    # Both should have non-empty conditions and deliverables
    for t in txns:
        assert t.condition
        assert t.deliverables


def test_parse_transitions_locate_recipe_single_target() -> None:
    md = (REPO_ROOT / "states" / "locate_recipe.md").read_text()
    txns = parse_transitions(md)
    assert len(txns) == 1
    t = txns[0]
    assert t.target == "configure_algorithm"
    assert "recipe.md" in t.condition
    assert any(d.key == "recipe" for d in t.deliverables)
    assert t.loop_max_iterations is None


def test_parse_transitions_terminal_state_empty() -> None:
    md = (REPO_ROOT / "states" / "finalize.md").read_text()
    assert parse_transitions(md) == []


def test_parse_transitions_synthetic_all_labels() -> None:
    md = (
        "## Next States\n\n"
        "### foo\n\n"
        "**Condition:** just this line.\n\n"
        "**Loop:** max_iterations: 7\n\n"
        "**Deliverables:**\n\n"
        "- k1: one thing.\n"
        "- k2: another thing with a colon: inside it.\n\n"
        "### bar\n\n"
        "**Condition:** two-line\ncondition body.\n\n"
        "**Deliverables:**\n\n- only: single.\n"
    )
    txns = parse_transitions(md)
    assert len(txns) == 2
    a, b = txns
    assert a.target == "foo"
    assert a.condition == "just this line."
    assert a.loop_max_iterations == 7
    assert [d.key for d in a.deliverables] == ["k1", "k2"]
    assert a.deliverables[1].description.endswith("inside it.")

    assert b.target == "bar"
    assert b.condition == "two-line\ncondition body."
    assert b.loop_max_iterations is None
    assert [d.key for d in b.deliverables] == ["only"]


# ── StateContext exposes transitions ──────────────────────────────────────

def test_state_context_exposes_transitions() -> None:
    from harness.context import load_state_context
    ctx = load_state_context(REPO_ROOT, "intake")
    assert len(ctx.transitions) == 5
    # Same order as next_states
    assert [t.target for t in ctx.transitions] == ctx.next_states


def test_initial_user_message_lists_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "intake").mkdir()
    (ws / "intake" / "training_intent.md").write_text("hi")
    msg = build_initial_user_message(
        goal="g",
        run_id="R2",
        workdir=tmp_path,
        verl_root=None,
        workspace=ws,
        state_name="intake",
    )
    assert "intake/training_intent.md" in msg
    assert "unset" in msg  # verl_root=None


# ── workspace snapshot ────────────────────────────────────────────────────

def test_snapshot_inlines_md_content(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "intake").mkdir(parents=True)
    (ws / "intake" / "training_intent.md").write_text(
        "goal: train\nalgorithm: grpo\nmodel: Qwen/Qwen2.5-3B\n"
    )
    (ws / "recipe").mkdir()
    (ws / "recipe" / "recipe.md").write_text("launch: /opt/verl/examples/grpo_trainer/run_gsm8k.sh\n")
    snap = render_workspace_snapshot(ws)
    assert "training_intent.md" in snap
    assert "algorithm: grpo" in snap
    assert "recipe.md" in snap
    assert "/opt/verl/examples" in snap


def test_snapshot_truncates_long_files(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "big.md").write_text("x" * 4000)
    snap = render_workspace_snapshot(ws, max_chars_per_file=200)
    # snapshot has fence overhead, but the raw content is capped
    assert snap.count("x") <= 260
    assert "more bytes" in snap


def test_snapshot_empty_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    snap = render_workspace_snapshot(ws)
    assert "(empty)" in snap


def test_snapshot_non_md_files_only_in_tree(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "intake").mkdir()
    (ws / "intake" / "verl_root.txt").write_text("/opt/verl-checkout-abc123")
    (ws / "intake" / "training_intent.md").write_text("goal: train")
    snap = render_workspace_snapshot(ws)
    # verl_root.txt shows in tree
    assert "verl_root.txt" in snap
    # But its unique content marker must NOT appear anywhere in the snapshot
    assert "verl-checkout-abc123" not in snap
    # Md contents ARE inlined (verified via the ~~~ fences)
    md_bodies = snap.split("~~~")[1::2]
    assert any("goal: train" in b for b in md_bodies)


def test_initial_user_message_includes_snapshot_by_default(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "intake").mkdir(parents=True)
    (ws / "intake" / "training_intent.md").write_text("algorithm: grpo\n")

    msg = build_initial_user_message(
        goal="g",
        run_id="R",
        workdir=tmp_path,
        verl_root=None,
        workspace=ws,
        state_name="locate_recipe",
    )
    assert "algorithm: grpo" in msg
    assert "Inlined deliverable contents" in msg


def test_initial_user_message_snapshot_can_be_disabled(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "intake").mkdir(parents=True)
    (ws / "intake" / "training_intent.md").write_text("algorithm: grpo\n")

    msg = build_initial_user_message(
        goal="g",
        run_id="R",
        workdir=tmp_path,
        verl_root=None,
        workspace=ws,
        state_name="locate_recipe",
        include_snapshot=False,
    )
    assert "algorithm: grpo" not in msg  # NOT inlined
    assert "training_intent.md" in msg   # still in tree
