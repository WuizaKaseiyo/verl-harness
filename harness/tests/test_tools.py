"""Unit tests for the 5 built-in tools + registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.tools import ToolContext, ToolRegistry, default_registry
from harness.tools.bash import BashTool
from harness.tools.edit import EditTool
from harness.tools.read import ReadTool
from harness.tools.todo import TodoTool
from harness.tools.write import WriteTool


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return ToolContext(
        cwd=tmp_path,
        workdir=tmp_path,
        run_id="TEST",
        workspace=workspace,
    )


# ── bash ─────────────────────────────────────────────────────────────────────

def test_bash_echo(ctx: ToolContext) -> None:
    out = BashTool().execute({"command": "printf hello"}, ctx)
    assert "hello" in out


def test_bash_cwd_honored(ctx: ToolContext) -> None:
    out = BashTool().execute({"command": "pwd"}, ctx)
    assert str(ctx.cwd) in out


def test_bash_exit_code_surfaces(ctx: ToolContext) -> None:
    out = BashTool().execute({"command": "false"}, ctx)
    assert "exit code 1" in out


def test_bash_timeout(ctx: ToolContext) -> None:
    out = BashTool().execute(
        {"command": "sleep 3", "timeout_ms": 300},
        ctx,
    )
    assert "timed out" in out


def test_bash_empty_command_rejected(ctx: ToolContext) -> None:
    from harness.tools import ToolError
    with pytest.raises(ToolError):
        BashTool().execute({"command": ""}, ctx)


def test_bash_truncates_huge_output(ctx: ToolContext) -> None:
    # 200 KB of `x` should truncate to ~100 KB with a marker.
    out = BashTool().execute(
        {"command": "python -c \"import sys; sys.stdout.write('x'*200000)\""},
        ctx,
    )
    assert "truncated" in out
    assert len(out) < 110_000


# ── read ─────────────────────────────────────────────────────────────────────

def test_read_line_numbered(ctx: ToolContext, tmp_path: Path) -> None:
    f = tmp_path / "sample.txt"
    f.write_text("alpha\nbeta\ngamma\n")
    out = ReadTool().execute({"file_path": str(f)}, ctx)
    assert "1\talpha" in out
    assert "2\tbeta" in out
    assert "3\tgamma" in out


def test_read_offset_limit(ctx: ToolContext, tmp_path: Path) -> None:
    f = tmp_path / "many.txt"
    f.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")
    out = ReadTool().execute(
        {"file_path": str(f), "offset": 5, "limit": 2},
        ctx,
    )
    assert "5\tline5" in out
    assert "6\tline6" in out
    assert "line4" not in out
    assert "line7" not in out


def test_read_missing_file_errors(ctx: ToolContext, tmp_path: Path) -> None:
    from harness.tools import ToolError
    with pytest.raises(ToolError, match="file not found"):
        ReadTool().execute({"file_path": str(tmp_path / "nope")}, ctx)


def test_read_refuses_parquet_extension(ctx: ToolContext, tmp_path: Path) -> None:
    """Extension-based binary refusal — cheap short-circuit."""
    from harness.tools import ToolError
    f = tmp_path / "train.parquet"
    f.write_bytes(b"PAR1\x15\x04\x15 " + b"\x00" * 100)
    with pytest.raises(ToolError, match="binary"):
        ReadTool().execute({"file_path": str(f)}, ctx)


def test_read_refuses_by_sniff_when_extension_hidden(
    ctx: ToolContext, tmp_path: Path
) -> None:
    """Binary content with a text-looking extension gets refused via NUL sniff."""
    from harness.tools import ToolError
    f = tmp_path / "sneaky.txt"
    f.write_bytes(b"header line\n" + b"\x00\x01\x02\x00\x00\xff" * 500)
    with pytest.raises(ToolError, match="binary"):
        ReadTool().execute({"file_path": str(f)}, ctx)


def test_read_allows_utf8_with_high_ascii(ctx: ToolContext, tmp_path: Path) -> None:
    """Ensure Chinese / emoji / accented text is NOT flagged as binary."""
    f = tmp_path / "utf8.md"
    f.write_text("你好世界\nCafé au lait\n🚀 rocket\n", encoding="utf-8")
    out = ReadTool().execute({"file_path": str(f)}, ctx)
    assert "你好世界" in out
    assert "Café" in out
    assert "🚀" in out


def test_read_output_capped_at_100kb(ctx: ToolContext, tmp_path: Path) -> None:
    """Text file whose line-numbered output would exceed 100 KB gets capped."""
    f = tmp_path / "huge.txt"
    # 5000 lines × ~30 chars each = ~150 KB before line-numbering
    f.write_text("\n".join("x" * 30 for _ in range(5000)) + "\n")
    out = ReadTool().execute(
        {"file_path": str(f), "limit": 5000}, ctx
    )
    assert len(out) < 105_000
    assert "capped at 100 KB" in out


def test_read_extension_check_case_insensitive(
    ctx: ToolContext, tmp_path: Path
) -> None:
    from harness.tools import ToolError
    f = tmp_path / "Weights.SafeTensors"
    f.write_bytes(b"garbage")
    with pytest.raises(ToolError, match="binary"):
        ReadTool().execute({"file_path": str(f)}, ctx)


# ── write ────────────────────────────────────────────────────────────────────

def test_write_creates_file(ctx: ToolContext, tmp_path: Path) -> None:
    target = tmp_path / "sub" / "new.txt"
    out = WriteTool().execute(
        {"file_path": str(target), "content": "hello world"},
        ctx,
    )
    assert target.read_text() == "hello world"
    assert "created" in out


def test_write_overwrites(ctx: ToolContext, tmp_path: Path) -> None:
    target = tmp_path / "existing.txt"
    target.write_text("old")
    out = WriteTool().execute(
        {"file_path": str(target), "content": "new"},
        ctx,
    )
    assert target.read_text() == "new"
    assert "overwrote" in out


# ── edit ─────────────────────────────────────────────────────────────────────

def test_edit_unique_match(ctx: ToolContext, tmp_path: Path) -> None:
    f = tmp_path / "e.txt"
    f.write_text("one two three")
    EditTool().execute(
        {"file_path": str(f), "old_string": "two", "new_string": "TWO"},
        ctx,
    )
    assert f.read_text() == "one TWO three"


def test_edit_non_unique_errors(ctx: ToolContext, tmp_path: Path) -> None:
    from harness.tools import ToolError
    f = tmp_path / "e.txt"
    f.write_text("x x x")
    with pytest.raises(ToolError, match="matches 3 times"):
        EditTool().execute(
            {"file_path": str(f), "old_string": "x", "new_string": "Y"},
            ctx,
        )
    assert f.read_text() == "x x x"


def test_edit_replace_all(ctx: ToolContext, tmp_path: Path) -> None:
    f = tmp_path / "e.txt"
    f.write_text("x x x")
    EditTool().execute(
        {
            "file_path": str(f),
            "old_string": "x",
            "new_string": "Y",
            "replace_all": True,
        },
        ctx,
    )
    assert f.read_text() == "Y Y Y"


def test_edit_no_match_errors(ctx: ToolContext, tmp_path: Path) -> None:
    from harness.tools import ToolError
    f = tmp_path / "e.txt"
    f.write_text("abc")
    with pytest.raises(ToolError, match="not found"):
        EditTool().execute(
            {"file_path": str(f), "old_string": "xyz", "new_string": "q"},
            ctx,
        )


def test_edit_identical_strings_errors(ctx: ToolContext, tmp_path: Path) -> None:
    from harness.tools import ToolError
    f = tmp_path / "e.txt"
    f.write_text("abc")
    with pytest.raises(ToolError, match="identical"):
        EditTool().execute(
            {"file_path": str(f), "old_string": "abc", "new_string": "abc"},
            ctx,
        )


# ── todo ─────────────────────────────────────────────────────────────────────

def test_todo_replace(ctx: ToolContext) -> None:
    out = TodoTool().execute(
        {
            "todos": [
                {"id": "1", "content": "do a", "status": "pending"},
                {"id": "2", "content": "do b", "status": "in_progress"},
            ]
        },
        ctx,
    )
    assert "2 items" in out
    assert ctx.todo_state["todos"] == [
        {"id": "1", "content": "do a", "status": "pending"},
        {"id": "2", "content": "do b", "status": "in_progress"},
    ]


def test_todo_bad_status_errors(ctx: ToolContext) -> None:
    from harness.tools import ToolError
    with pytest.raises(ToolError, match="status"):
        TodoTool().execute(
            {"todos": [{"id": "1", "content": "x", "status": "unknown"}]},
            ctx,
        )


# ── registry ─────────────────────────────────────────────────────────────────

def test_registry_default_has_all_six() -> None:
    """M1 = 5 tools; M3-T5 added web_fetch → 6 total."""
    reg = default_registry()
    assert set(reg.names()) == {"bash", "read", "edit", "write", "todo", "web_fetch"}


def test_registry_schemas_are_anthropic_shape() -> None:
    reg = default_registry()
    for s in reg.schemas():
        assert set(s.keys()) == {"name", "description", "input_schema"}
        assert s["input_schema"]["type"] == "object"
        assert "properties" in s["input_schema"]


def test_registry_execute_unknown_is_error(ctx: ToolContext) -> None:
    reg = default_registry()
    content, is_error = reg.execute("nope", {}, ctx)
    assert is_error
    assert "unknown tool" in content


def test_registry_execute_wraps_toolerror(ctx: ToolContext, tmp_path: Path) -> None:
    reg = default_registry()
    content, is_error = reg.execute(
        "read",
        {"file_path": str(tmp_path / "missing")},
        ctx,
    )
    assert is_error
    assert "file not found" in content


def test_registry_execute_success(ctx: ToolContext, tmp_path: Path) -> None:
    reg = default_registry()
    target = tmp_path / "roundtrip.txt"
    _, is_error = reg.execute(
        "write",
        {"file_path": str(target), "content": "hi"},
        ctx,
    )
    assert not is_error
    content, is_error = reg.execute("read", {"file_path": str(target)}, ctx)
    assert not is_error
    assert "1\thi" in content


def test_registry_duplicate_name_errors() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry([BashTool(), BashTool()])
