"""Prompt-caching unit tests.

Live cache-hit measurement lives in test_backends.py (gated on API key) and
the M3-T10 E2E smoke. Here we cover the wire-format transformation and the
context.py block partitioning.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.backends.anthropic import _prepare_system_with_cache
from harness.context import load_state_context


REPO_ROOT = Path(__file__).resolve().parents[2]


# ── _prepare_system_with_cache ────────────────────────────────────────────

def test_string_system_passed_through() -> None:
    """Legacy path: str stays str."""
    out = _prepare_system_with_cache("hello")
    assert out == "hello"


def test_list_of_blocks_gets_cache_control_on_first_four() -> None:
    blocks = [
        {"type": "text", "text": f"block {i}"} for i in range(6)
    ]
    out = _prepare_system_with_cache(blocks)
    assert len(out) == 6
    # First 4 have cache_control
    for i in range(4):
        assert out[i].get("cache_control") == {"type": "ephemeral"}, i
    # Beyond 4, no cache_control
    for i in range(4, 6):
        assert "cache_control" not in out[i], i


def test_existing_cache_control_not_double_added() -> None:
    """If caller already set cache_control on a block, we leave it alone."""
    blocks = [
        {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "b"},
    ]
    out = _prepare_system_with_cache(blocks)
    assert out[0]["cache_control"] == {"type": "ephemeral"}
    assert out[1]["cache_control"] == {"type": "ephemeral"}


def test_input_list_not_mutated() -> None:
    """Caller's blocks list should NOT be modified in place."""
    original = [{"type": "text", "text": "keep me pristine"}]
    _prepare_system_with_cache(original)
    assert original == [{"type": "text", "text": "keep me pristine"}]


def test_non_text_blocks_still_pass_through() -> None:
    """A non-text block (unusual but possible) should not receive cache_control."""
    blocks = [
        {"type": "text", "text": "hi"},
        {"type": "image", "source": {}},  # hypothetical
    ]
    out = _prepare_system_with_cache(blocks)
    assert out[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in out[1]


# ── context.py exposes cacheable blocks ────────────────────────────────────

def test_state_context_produces_system_blocks() -> None:
    ctx = load_state_context(REPO_ROOT, "intake")
    assert ctx.system_blocks, "system_blocks should be populated"
    # Each block is a proper Anthropic content shape
    for b in ctx.system_blocks:
        assert b.get("type") == "text"
        assert isinstance(b.get("text"), str)


def test_system_blocks_contain_claude_md_and_state_md() -> None:
    ctx = load_state_context(REPO_ROOT, "intake")
    joined = "\n\n".join(b["text"] for b in ctx.system_blocks)
    assert "CLAUDE.md" in joined
    assert "## states/intake.md" in joined
    assert "skills/intake" in joined


def test_system_blocks_ordered_for_cache_wins() -> None:
    """Cacheable content (CLAUDE.md, state.md, skills) should be first — those
    are what gets `cache_control` applied by the anthropic backend."""
    ctx = load_state_context(REPO_ROOT, "intake")
    # Preamble is block 0
    assert "runtime agent" in ctx.system_blocks[0]["text"].lower()
    # The last block is the small transition rules chunk
    assert "Transition rules" in ctx.system_blocks[-1]["text"]


def test_terminal_state_blocks_include_terminal_marker() -> None:
    ctx = load_state_context(REPO_ROOT, "finalize")
    joined = "\n\n".join(b["text"] for b in ctx.system_blocks)
    assert "Terminal state" in joined
