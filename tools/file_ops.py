"""Filesystem built-in tools with workspace-bound path handling."""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any


IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".pytest_cache"}


def _resolve_under_root(root: str | Path, raw_path: str | Path = ".") -> Path:
    root_path = Path(root).resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root_path / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise PermissionError(f"path `{raw_path}` resolves outside root `{root_path}`") from exc
    return resolved


def list_dir(path: str = ".", *, root: str = ".", max_depth: int = 2) -> dict[str, Any]:
    target = _resolve_under_root(root, path)
    if not target.exists() or not target.is_dir():
        raise FileNotFoundError(f"directory not found: {target}")

    rows: list[str] = []

    def walk(current: Path, depth: int, prefix: str = "") -> None:
        if depth > max_depth:
            return
        entries = sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        entries = [item for item in entries if item.name not in IGNORE_DIRS and not item.name.startswith("._")]
        for idx, entry in enumerate(entries):
            is_last = idx == len(entries) - 1
            connector = "└── " if is_last else "├── "
            rows.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                walk(entry, depth + 1, prefix + ("    " if is_last else "│   "))

    walk(target, 1)
    return {"tool": "list_dir", "path": str(target), "tree": rows}


def read_file(path: str, *, root: str = ".", start_line: int = 1, end_line: int = -1) -> dict[str, Any]:
    target = _resolve_under_root(root, path)
    if not target.is_file():
        raise FileNotFoundError(f"file not found: {target}")
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    start = max(1, start_line)
    end = total if end_line == -1 else min(end_line, total)
    if start > total and total:
        raise ValueError(f"start_line {start} exceeds file length {total}")
    selected = lines[start - 1 : end] if total else []
    return {
        "tool": "read_file",
        "path": str(target),
        "start_line": start,
        "end_line": end,
        "total_lines": total,
        "content": "\n".join(selected),
        "numbered_content": "\n".join(f"{start + i:5d} | {line}" for i, line in enumerate(selected)),
    }


def file_create(path: str, *, root: str = ".", content: str = "", overwrite: bool = False) -> dict[str, Any]:
    target = _resolve_under_root(root, path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"file already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"tool": "file_create", "path": str(target), "bytes": len(content.encode("utf-8"))}


def append_file(path: str, *, root: str = ".", content: str = "") -> dict[str, Any]:
    target = _resolve_under_root(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(content)
    return {"tool": "append_file", "path": str(target), "bytes": len(content.encode("utf-8"))}


def mkdir(path: str, *, root: str = ".") -> dict[str, Any]:
    target = _resolve_under_root(root, path)
    target.mkdir(parents=True, exist_ok=True)
    return {"tool": "mkdir", "path": str(target)}


def grep(
    pattern: str,
    *,
    root: str = ".",
    path: str = ".",
    file_pattern: str = "*",
    text: str | None = None,
    max_matches: int = 100,
) -> dict[str, Any]:
    if text is not None:
        regex = re.compile(pattern, re.MULTILINE)
        matches = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append({"line": line_no, "text": line})
                if len(matches) >= max_matches:
                    break
        return {
            "tool": "grep",
            "mode": "text",
            "pattern": pattern,
            "matches": matches,
            "truncated": len(matches) >= max_matches,
        }

    target = _resolve_under_root(root, path)
    if not target.exists() or not target.is_dir():
        raise FileNotFoundError(f"directory not found: {target}")
    regex = re.compile(pattern)
    matches = []
    binary_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".pyc"}
    for current_root, dirs, files in os.walk(target):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
        for filename in files:
            if filename.startswith(".") or not fnmatch.fnmatch(filename, file_pattern):
                continue
            file_path = Path(current_root) / filename
            if file_path.suffix.lower() in binary_suffixes:
                continue
            try:
                lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            for line_no, line in enumerate(lines, start=1):
                if regex.search(line):
                    matches.append(
                        {
                            "path": str(file_path),
                            "relative_path": str(file_path.relative_to(target)),
                            "line": line_no,
                            "text": line.strip(),
                        }
                    )
                    if len(matches) >= max_matches:
                        return {
                            "tool": "grep",
                            "mode": "files",
                            "root": str(target),
                            "pattern": pattern,
                            "file_pattern": file_pattern,
                            "matches": matches,
                            "truncated": True,
                        }
    return {
        "tool": "grep",
        "mode": "files",
        "root": str(target),
        "pattern": pattern,
        "file_pattern": file_pattern,
        "matches": matches,
        "truncated": False,
    }
