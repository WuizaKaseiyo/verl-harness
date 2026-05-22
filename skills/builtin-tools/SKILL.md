---
name: builtin-tools
description: Use when a state needs filesystem read/write, shell execution, web search, or web URL fetching and you want to know exactly which FastHarness built-in tool to invoke, what its arguments are, what JSON it returns, and what safety guarantees apply. Covers 9 tools (list_dir, read_file, grep, file_create, append_file, mkdir, shell_exec, search_web, fetch_webpage) bound to 5 capabilities. Called automatically by `run-harness` Step 2.6 when a declared `## Required Capabilities` token has no native host tool, and ad-hoc by any state whose work needs these primitives.
---

# builtin-tools — FastHarness Built-in Tool Reference

A reference for the nine built-in tools that ship with FastHarness. The implementations live in `tools/`; this skill is the agent-facing manual: when to use them, how to invoke them, what they return, what they refuse to do.

## When this skill applies

- You are running a harness, you hit `run-harness` Step 2.6, and a declared capability has no native tool on this host.
- A state's `## Description` calls for a filesystem / shell / web operation and you want a host-agnostic way to do it.
- A user asks you to run one of these tools ad-hoc on their machine.

You do **not** need this skill for capabilities your host already covers natively. Always prefer the native tool — these built-ins are a fallback.

## Decision rule (read this once, apply on every operation)

For any operation you need to perform:

1. **If your host exposes a native tool** for the capability, use it. Native tools are usually faster, better integrated, and have richer error messages.
2. **Else, use a FastHarness built-in tool** from this skill.
3. **If neither is available**, write the missing capability to `<WORKSPACE>/error.md` per `run-harness` Step 2.6 and halt. Do not invent a workaround.

## The nine tools at a glance

| Capability | Tool | Purpose |
|---|---|---|
| `filesystem.read` | `list_dir` | List a directory tree (depth-bounded, skips noisy dirs) |
| `filesystem.read` | `read_file` | Read a file under `root`, optionally a line range |
| `filesystem.read` | `grep` | Regex search across files or a text string |
| `filesystem.write` | `file_create` | Create or overwrite a file under `root` |
| `filesystem.write` | `append_file` | Append to a file under `root` |
| `filesystem.write` | `mkdir` | Create a directory (incl. parents) under `root` |
| `shell.exec` | `shell_exec` | Run a shell command with cwd bounded by `root` and a dangerous-command blacklist |
| `web.search` | `search_web` | Google-via-Serper.dev search (requires `SERPER_API_KEY`) |
| `web.fetch` | `fetch_webpage` | Fetch a URL, extract readable text + links |

For full per-tool parameters and return shapes, read:
- **`filesystem.md`** — the six filesystem tools
- **`shell.md`** — `shell_exec`
- **`web.md`** — `search_web` and `fetch_webpage`

## Invocation patterns

Two equivalent entry points, both dispatching to the same registry:

**CLI** (most common — works from any host that has shell + python3):

```bash
python tools/registry.py <tool_name> '<json_args>' [--workspace <path>] [--output <path>]
```

Examples:

```bash
python tools/registry.py list_dir '{"root":"<WORKSPACE>","path":".","max_depth":2}' --workspace "<WORKSPACE>"
python tools/registry.py search_web '{"query":"finite state machine LLM","max_results":10}' --workspace "<WORKSPACE>" --output "<WORKSPACE>/search/q01.json"
```

**Python import** (only relevant if you are authoring a harness's helper Python somewhere):

```python
from tools.registry import execute_builtin_tool
result = execute_builtin_tool("read_file", {"root": "<WORKSPACE>", "path": "notes.md"})
```

To get the canonical JSON Schema for every tool (useful for function-calling-style hosts):

```bash
python tools/registry.py --list      # tool names only
python tools/registry.py --schemas   # full OpenAI-function-calling JSON Schema for each
```

## Common arguments (every tool accepts these)

- **`workspace`** — pass `--workspace "<WORKSPACE>"` (CLI) or `workspace="<WORKSPACE>"` (Python). Each call appends a JSONL record to `<WORKSPACE>/logs/tool_calls.jsonl` with timestamp, tool name, arguments, status, and (if applicable) output path. **Always pass this from inside a harness** — it is what makes the run auditable.
- **`output`** — pass `--output <path>.json` (CLI) or `output="..."` (Python). The result JSON is written to that path instead of just stdout. Use this for large results (search, fetch) that would clog the conversation.

## Observability

The JSONL log entry written to `<WORKSPACE>/logs/tool_calls.jsonl`:

```json
{"timestamp": "2026-05-13T12:54:55+00:00", "tool": "search_web", "args": {"query": "..."}, "status": "ok", "output_path": "<WORKSPACE>/search/q01.json", "error": null}
```

Failed calls write a record with `"status": "error"` and the exception message in `"error"`. Log writes are **best-effort** — a failure to write the log will not fail the tool call itself.

## Safety guarantees (built into the tools, not opt-in)

- **Filesystem path sandboxing.** Every filesystem tool takes a `root`. Paths that resolve outside `root` are rejected with `PermissionError: path resolves outside root`. Absolute paths are still validated — you cannot escape by passing `/etc/passwd`.
- **Shell command blacklist.** `shell_exec` rejects commands matching `\bsudo\b`, `\brm -rf /`, `\brm -rf ~`, `\bgit reset --hard\b`, redirections to `/etc/`, `/usr/`, `/bin/`, `/sbin/`. See `shell.md` for the exact patterns.
- **Shell cwd sandboxing.** `shell_exec`'s `cwd` must resolve under `root`.
- **Output truncation.** `shell_exec` stdout/stderr default to 50000 chars max; longer output is truncated with a `... [STDOUT TRUNCATED]` marker.
- **Network timeouts.** `search_web` and `fetch_webpage` default to a 30-second timeout per request.

## Dependencies

All nine tools depend only on the Python standard library — no `pip install` required. You need:

- `python3` (>= 3.10) on `$PATH`
- `SERPER_API_KEY` in the environment if and only if you use `search_web`

## Things you must not do

- **Do not bypass `root`** by hoping absolute paths sneak through. They are still validated.
- **Do not call `shell_exec` with raw user input** as the `command`. The blacklist is conservative, not exhaustive; user-supplied commands belong to a different threat model.
- **Do not write tool outputs anywhere except under `<WORKSPACE>/`.** Use the per-state IO log to record the output path.
- **Do not skip `--workspace`** when invoked from a harness. The tool_calls.jsonl is what links these tool invocations to the run that triggered them.
- **Do not use these tools to modify the harness folder itself.** `task-overview.md`, `states/`, and `skills/` are inputs to the run, not outputs. Write to `<WORKSPACE>/` only.
