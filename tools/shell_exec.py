"""Restricted shell execution built-in tool."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

try:
    from .file_ops import _resolve_under_root
except ImportError:
    from file_ops import _resolve_under_root


DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\s+/",
    r"\brm\s+-rf\s+~",
    r"\brm\s+-rf\s+\$HOME",
    r"\bgit\s+reset\s+--hard\b",
    r"\bsudo\b",
    r">\s*/etc/",
    r">\s*/usr/",
    r">\s*/bin/",
    r">\s*/sbin/",
]


def _check_command(command: str) -> None:
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            raise PermissionError(f"command rejected by policy: matched `{pattern}`")


def shell_exec(
    command: str,
    *,
    root: str = ".",
    cwd: str = ".",
    timeout: int = 60,
    stdin: str | None = None,
    max_output_chars: int = 50000,
) -> dict[str, Any]:
    if not command.strip():
        raise ValueError("command must not be empty")
    _check_command(command)
    work_dir = _resolve_under_root(root, cwd)
    if not work_dir.is_dir():
        raise FileNotFoundError(f"cwd is not a directory: {work_dir}")
    completed = subprocess.run(
        command,
        input=stdin,
        text=True,
        shell=True,
        cwd=work_dir,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    stdout_truncated = len(stdout) > max_output_chars
    stderr_truncated = len(stderr) > max_output_chars
    if stdout_truncated:
        stdout = stdout[:max_output_chars] + "\n... [STDOUT TRUNCATED]"
    if stderr_truncated:
        stderr = stderr[:max_output_chars] + "\n... [STDERR TRUNCATED]"
    return {
        "tool": "shell_exec",
        "capability": "shell.exec",
        "command": command,
        "cwd": str(work_dir),
        "timeout": timeout,
        "exit_code": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }
