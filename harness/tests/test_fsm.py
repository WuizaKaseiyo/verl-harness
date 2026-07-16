"""FSM graph loader — validates against the real states/ tree + synthetic cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.fsm import FSM, FSMError

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── against the real states/ tree ────────────────────────────────────────

def test_load_real_fsm() -> None:
    fsm = FSM.load(REPO_ROOT)
    assert set(fsm.states) == {
        "intake",
        "locate_recipe",
        "configure_algorithm",
        "prepare_data",
        "generate_preprocess",
        "configure_reward",
        "select_compute",
        "provision_env",
        "sanity_rollout",
        "launch_training",
        "monitor_training",
        "summarize",
        "reflect",
        "run_generate",
        "run_eval",
        "finalize",
    }


def test_intake_has_five_targets() -> None:
    fsm = FSM.load(REPO_ROOT)
    targets = set(fsm.next_states("intake"))
    assert targets == {
        "locate_recipe",
        "monitor_training",
        "launch_training",
        "run_generate",
        "run_eval",
    }


def test_finalize_is_terminal() -> None:
    fsm = FSM.load(REPO_ROOT)
    assert fsm.is_terminal("finalize")
    assert fsm.terminal_states() == ["finalize"]


def test_declared_loop_edges_identified() -> None:
    fsm = FSM.load(REPO_ROOT)
    loops = fsm.loop_edges()
    # reflect → configure_algorithm and prepare_data → generate_preprocess
    assert ("reflect", "configure_algorithm") in loops
    assert ("prepare_data", "generate_preprocess") in loops
    assert len(loops) == 2  # no others in the current spec


def test_transition_lookup_returns_metadata() -> None:
    fsm = FSM.load(REPO_ROOT)
    t = fsm.transition("reflect", "configure_algorithm")
    assert t is not None
    assert t.loop_max_iterations == 3
    assert t.condition
    assert any(d.key == "refinement_plan" for d in t.deliverables)


def test_real_fsm_validates_clean() -> None:
    fsm = FSM.load(REPO_ROOT, validate=False)
    assert fsm.validate() == []


# ── synthetic error cases ────────────────────────────────────────────────

def _write_state(workdir: Path, name: str, md: str) -> None:
    states = workdir / "states"
    states.mkdir(parents=True, exist_ok=True)
    (states / f"{name}.md").write_text(md)


def _skill_dir(workdir: Path, name: str = "global") -> None:
    (workdir / "skills" / name).mkdir(parents=True, exist_ok=True)
    (workdir / "skills" / name / "default.md").write_text("dummy skill\n")


def _basic_state(name: str, next_name: str | None) -> str:
    body = f"# {name}\n\n## Description\n\nfoo\n\n## Skills\n\n- skills/global\n\n"
    if next_name is None:
        return body
    return body + (
        f"## Next States\n\n### {next_name}\n\n"
        f"**Condition:** always.\n\n"
        f"**Deliverables:**\n\n- k: whatever.\n"
    )


def test_missing_target_state_errors(tmp_path: Path) -> None:
    _skill_dir(tmp_path)
    _write_state(tmp_path, "intake", _basic_state("intake", "ghost"))
    _write_state(tmp_path, "finalize", _basic_state("finalize", None))
    with pytest.raises(FSMError, match="ghost"):
        FSM.load(tmp_path)


def test_orphan_state_errors(tmp_path: Path) -> None:
    _skill_dir(tmp_path)
    _write_state(tmp_path, "intake", _basic_state("intake", "finalize"))
    _write_state(tmp_path, "finalize", _basic_state("finalize", None))
    _write_state(tmp_path, "orphan", _basic_state("orphan", "finalize"))
    with pytest.raises(FSMError, match="unreachable"):
        FSM.load(tmp_path)


def test_undeclared_cycle_errors(tmp_path: Path) -> None:
    """A → B → A without a **Loop:** marker on either edge."""
    _skill_dir(tmp_path)
    _write_state(tmp_path, "intake", _basic_state("intake", "a"))
    _write_state(tmp_path, "finalize", _basic_state("finalize", None))
    _write_state(tmp_path, "a", _basic_state("a", "b"))
    _write_state(
        tmp_path,
        "b",
        "## Description\n\nx\n\n## Skills\n\n- skills/global\n\n"
        "## Next States\n\n### a\n\n**Condition:** repeat.\n\n"
        "**Deliverables:**\n\n- k: whatever.\n\n"
        "### finalize\n\n**Condition:** done.\n\n"
        "**Deliverables:**\n\n- k: whatever.\n",
    )
    with pytest.raises(FSMError, match="undeclared cycle"):
        FSM.load(tmp_path)


def test_declared_cycle_accepted(tmp_path: Path) -> None:
    """Same A → B → A shape but with **Loop:** on the back edge."""
    _skill_dir(tmp_path)
    _write_state(tmp_path, "intake", _basic_state("intake", "a"))
    _write_state(tmp_path, "finalize", _basic_state("finalize", None))
    _write_state(tmp_path, "a", _basic_state("a", "b"))
    _write_state(
        tmp_path,
        "b",
        "## Description\n\nx\n\n## Skills\n\n- skills/global\n\n"
        "## Next States\n\n### a\n\n**Condition:** repeat.\n\n"
        "**Loop:** max_iterations: 2\n\n"
        "**Deliverables:**\n\n- k: whatever.\n\n"
        "### finalize\n\n**Condition:** done.\n\n"
        "**Deliverables:**\n\n- k: whatever.\n",
    )
    fsm = FSM.load(tmp_path)
    assert ("b", "a") in fsm.loop_edges()


def test_no_drain_to_terminal_errors(tmp_path: Path) -> None:
    """State that only loops back → no path to terminal on loop-free subgraph."""
    _skill_dir(tmp_path)
    _write_state(tmp_path, "intake", _basic_state("intake", "a"))
    _write_state(tmp_path, "finalize", _basic_state("finalize", None))
    _write_state(
        tmp_path,
        "a",
        "## Description\n\nx\n\n## Skills\n\n- skills/global\n\n"
        "## Next States\n\n### intake\n\n**Condition:** repeat.\n\n"
        "**Loop:** max_iterations: 2\n\n"
        "**Deliverables:**\n\n- k: whatever.\n",
    )
    # intake → a → intake (declared loop). Neither reaches finalize acyclically.
    with pytest.raises(FSMError, match="no loop-free path"):
        FSM.load(tmp_path)


def test_missing_entry_errors(tmp_path: Path) -> None:
    _skill_dir(tmp_path)
    _write_state(tmp_path, "foo", _basic_state("foo", None))
    with pytest.raises(FSMError, match="entry state"):
        FSM.load(tmp_path)


def test_unknown_state_query_raises() -> None:
    fsm = FSM.load(REPO_ROOT)
    with pytest.raises(FSMError, match="unknown state"):
        fsm.next_states("does_not_exist")
