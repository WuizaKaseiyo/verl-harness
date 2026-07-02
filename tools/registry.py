#!/usr/bin/env python3
"""verl-harness built-in tool registry.

Exposes the nine built-in tools (filesystem / shell / web) that verl-harness's
`## Required Capabilities` maps to. New tools can be added by extending
BUILTIN_TOOL_REGISTRY + BUILTIN_TOOL_SCHEMAS below.

Invocation:

    python tools/registry.py <tool_name> '<json_args>' \\
        --workspace "<WORKSPACE>" [--output "<WORKSPACE>/<path>"]

    python tools/registry.py --list      # list tool names
    python tools/registry.py --schemas   # dump JSON schemas
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

try:
    from .file_ops import append_file, file_create, grep, list_dir, mkdir, read_file
    from .shell_exec import shell_exec
    from .tool_log import log_tool_call
    from .web import fetch_webpage, search_web
except ImportError:  # Allow `python tools/registry.py ...`
    from file_ops import append_file, file_create, grep, list_dir, mkdir, read_file
    from shell_exec import shell_exec
    from tool_log import log_tool_call
    from web import fetch_webpage, search_web


# ---------------------------------------------------------------------------
# Capability → tool-name index. verl-harness declares filesystem.read/write,
# shell.exec, web.search, web.fetch in its task-overview; this mapping is what
# `run-harness` Step 2.6 consults when the host lacks a native tool.
# ---------------------------------------------------------------------------
CAPABILITY_TOOL_MAP = {
    "filesystem.read":  ["list_dir", "read_file", "grep"],
    "filesystem.write": ["file_create", "append_file", "mkdir"],
    "shell.exec":       ["shell_exec"],
    "web.search":       ["search_web"],
    "web.fetch":        ["fetch_webpage"],
}


BUILTIN_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List a workspace directory as a tree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "root": {"type": "string", "description": "Allowed root directory.", "default": "."},
                    "path": {"type": "string", "description": "Directory path under root.", "default": "."},
                    "max_depth": {"type": "integer", "description": "Maximum recursion depth.", "default": 2},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file under an allowed root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "root": {"type": "string", "description": "Allowed root directory.", "default": "."},
                    "path": {"type": "string", "description": "File path under root."},
                    "start_line": {"type": "integer", "default": 1},
                    "end_line": {"type": "integer", "default": -1},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_create",
            "description": "Create or overwrite a file under an allowed root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "root": {"type": "string", "description": "Allowed root directory.", "default": "."},
                    "path": {"type": "string", "description": "File path under root."},
                    "content": {"type": "string", "default": ""},
                    "overwrite": {"type": "boolean", "default": False},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Append text to a file under an allowed root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "root": {"type": "string", "description": "Allowed root directory.", "default": "."},
                    "path": {"type": "string", "description": "File path under root."},
                    "content": {"type": "string", "default": ""},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mkdir",
            "description": "Create a directory under an allowed root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "root": {"type": "string", "description": "Allowed root directory.", "default": "."},
                    "path": {"type": "string", "description": "Directory path under root."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a regex pattern in files under root, or in a provided text string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "root": {"type": "string", "description": "Allowed root directory.", "default": "."},
                    "path": {"type": "string", "description": "Directory path under root.", "default": "."},
                    "file_pattern": {"type": "string", "description": "Glob pattern for files.", "default": "*"},
                    "pattern": {"type": "string", "description": "Python regular expression."},
                    "text": {"type": "string", "description": "Optional text to search instead of files."},
                    "max_matches": {"type": "integer", "default": 100},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": "Execute a restricted shell command with cwd bounded by root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "root": {"type": "string", "description": "Allowed root directory.", "default": "."},
                    "cwd": {"type": "string", "description": "Working directory under root.", "default": "."},
                    "command": {"type": "string", "description": "Shell command to run."},
                    "timeout": {"type": "integer", "default": 60},
                    "stdin": {"type": "string", "description": "Optional standard input."},
                    "max_output_chars": {"type": "integer", "default": 50000},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web using Serper.dev and return normalized JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {"type": "integer", "default": 10},
                    "location": {"type": "string", "description": "Optional Serper location string."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_webpage",
            "description": "Fetch a URL and extract readable text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "HTTP or HTTPS URL."},
                    "max_chars": {"type": "integer", "default": 15000},
                },
                "required": ["url"],
            },
        },
    },
]


BUILTIN_TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "list_dir":      list_dir,
    "read_file":     read_file,
    "grep":          grep,
    "file_create":   file_create,
    "append_file":   append_file,
    "mkdir":         mkdir,
    "shell_exec":    shell_exec,
    "search_web":    search_web,
    "fetch_webpage": fetch_webpage,
}


def is_builtin_tool(name: str) -> bool:
    return name in BUILTIN_TOOL_REGISTRY


def get_tool_schema(tool_name: str) -> dict[str, Any] | None:
    for schema in BUILTIN_TOOL_SCHEMAS:
        function = schema["function"]
        if function["name"] == tool_name:
            return function["parameters"]
    return None


def get_tools_summary() -> str:
    rows = []
    for schema in BUILTIN_TOOL_SCHEMAS:
        function = schema["function"]
        description = function["description"].split(".")[0]
        rows.append(f"- `{function['name']}` - {description}")
    return "\n".join(rows)


def execute_builtin_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    arguments = dict(arguments)
    workspace = arguments.pop("workspace", None)
    output = arguments.pop("output", None)
    fn = BUILTIN_TOOL_REGISTRY.get(name)
    if fn is None:
        raise KeyError(f"unknown builtin tool `{name}`")
    try:
        result = fn(**arguments)
        output_path = None
        if output:
            output_path = str(Path(output).resolve())
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        log_tool_call(workspace=workspace, tool=name, args=arguments,
                      status="ok", output_path=output_path)
        return result
    except Exception as exc:
        log_tool_call(workspace=workspace, tool=name, args=arguments,
                      status="error", error=str(exc))
        raise


def _parse_json_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON arguments: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("arguments JSON must be an object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="registry.py",
        description="Run a verl-harness built-in tool.",
    )
    parser.add_argument("tool", nargs="?", help="Tool name to execute.")
    parser.add_argument("arguments", nargs="?", help="JSON object with tool arguments.")
    parser.add_argument("--list", action="store_true", help="List tool names and exit.")
    parser.add_argument("--schemas", action="store_true", help="Print tool schemas and exit.")
    parser.add_argument("--workspace", help="Workspace path for tool call logging.")
    parser.add_argument("--output", help="Optional path to write the tool JSON result.")
    args = parser.parse_args(argv)

    if args.list:
        print("\n".join(BUILTIN_TOOL_REGISTRY))
        return 0
    if args.schemas:
        print(json.dumps(BUILTIN_TOOL_SCHEMAS, indent=2, ensure_ascii=False))
        return 0
    if not args.tool:
        parser.error("tool is required unless --list or --schemas is used")

    tool_args = _parse_json_args(args.arguments)
    if args.workspace:
        tool_args["workspace"] = args.workspace
    if args.output:
        tool_args["output"] = args.output
    try:
        result = execute_builtin_tool(args.tool, tool_args)
    except Exception as exc:
        print(f"ERR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
