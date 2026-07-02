# Shell Built-in Tool

One tool covers `shell.exec`. Implementation in `tools/shell_exec.py`, dispatched through `tools/registry.py`.

`shell_exec` is intentionally restricted — it is *not* a substitute for a host's general-purpose `Bash` tool. Use it when there is no native shell tool, and only for commands that operate on workspace files.

---

## `shell_exec` — restricted shell execution

Runs one shell command via `subprocess.run(..., shell=True)`, with three safety layers: a `cwd` bounded by `root`, a regex blacklist of dangerous patterns, and capped stdout/stderr.

**Parameters**

| Name | Type | Default | Required | Description |
|---|---|---|---|---|
| `command` | string | — | ✓ | Shell command to run |
| `root` | string | `.` |   | Allowed root directory |
| `cwd` | string | `.` |   | Working directory under `root` |
| `timeout` | int | `60` |   | Seconds before `subprocess.TimeoutExpired` |
| `stdin` | string | — |   | Optional standard input |
| `max_output_chars` | int | `50000` |   | Truncate each of stdout/stderr beyond this length |

**Returns**

```json
{
  "tool": "shell_exec",
  "capability": "shell.exec",
  "command": "pwd && ls",
  "cwd": "<absolute resolved cwd>",
  "timeout": 60,
  "exit_code": 0,
  "stdout": "...",
  "stderr": "",
  "stdout_truncated": false,
  "stderr_truncated": false
}
```

A non-zero `exit_code` is **not** an exception — the call still returns the dict. Inspect `exit_code` yourself and decide whether to treat it as a failure.

**Native equivalents:** Claude Code `Bash`, Cursor `RunCommand`, any host's shell-execution tool.

**CLI**

```bash
python tools/registry.py shell_exec '{"root":"<WORKSPACE>","cwd":".","command":"pwd && ls -la","timeout":30}' --workspace "<WORKSPACE>"
```

---

## Safety layer 1 — `cwd` sandboxing

`cwd` is resolved via `_resolve_under_root(root, cwd)` (the same gate as filesystem tools). If `cwd` resolves outside `root`, the call raises `PermissionError` before any command runs.

What this means in practice:

- ✓ `root="<WORKSPACE>", cwd="."` — runs in `<WORKSPACE>`.
- ✓ `root="<WORKSPACE>", cwd="deliverables/round-2"` — runs in `<WORKSPACE>/deliverables/round-2`.
- ✗ `root="<WORKSPACE>", cwd="../../etc"` — rejected, escapes `root`.
- ✗ `root="<WORKSPACE>", cwd="/etc"` — rejected, absolute path outside `root`.

But note: `cwd` only controls *where the process starts*. Once running, the command itself can still `cd /elsewhere && cat /etc/passwd`. The blacklist (layer 2) catches the most common dangerous forms; layer 1 alone is not a chroot.

---

## Safety layer 2 — dangerous-command blacklist

The command string is matched (case-insensitive) against this regex list before execution. Any match raises `PermissionError: command rejected by policy: matched <pattern>`.

```
\brm\s+-rf\s+/        # rm -rf /
\brm\s+-rf\s+~        # rm -rf ~
\brm\s+-rf\s+\$HOME   # rm -rf $HOME
\bgit\s+reset\s+--hard\b
\bsudo\b
>\s*/etc/
>\s*/usr/
>\s*/bin/
>\s*/sbin/
```

This is **conservative, not exhaustive**. It catches the most common shoot-self-in-foot patterns but cannot defend against an adversarial command author. **Do not feed user-supplied commands through `shell_exec`.** If you need that threat model, do not use this tool.

---

## Safety layer 3 — output truncation

`stdout` and `stderr` are captured separately and each truncated to `max_output_chars` (default 50000). When truncated, the field gets a trailing `... [STDOUT TRUNCATED]` (or `... [STDERR TRUNCATED]`) marker, and the corresponding `*_truncated` flag in the result is `true`.

This is what keeps a runaway command (`find / | head -10000000`) from blowing up the conversation.

---

## Idioms

**Tail a log into a deliverable:**

```bash
python tools/registry.py shell_exec \
  '{"root":"<WORKSPACE>","cwd":".","command":"tail -20 logs/state_log.md > deliverables/recent.md"}' \
  --workspace "<WORKSPACE>"
```

**Diff two workspace files:**

```bash
python tools/registry.py shell_exec \
  '{"root":"<WORKSPACE>","cwd":".","command":"diff -u a.md b.md","timeout":10}' \
  --workspace "<WORKSPACE>"
```

**Count states in a harness:**

```bash
python tools/registry.py shell_exec \
  '{"root":"<harness_root>","cwd":".","command":"ls states/*.md | wc -l"}' \
  --workspace "<WORKSPACE>"
```

---

## Things you must not do

- **Do not chain `&&` to bypass `cwd` sandboxing.** A command like `cd ../../etc && cat passwd` starts inside `cwd` but the chained `cd` is outside the sandbox layer. The blacklist may or may not catch it. Use `shell_exec` for legitimate workspace commands only.
- **Do not pipe shell output into another `shell_exec` to grow effective output past `max_output_chars`.** If you need more output than the cap allows, redirect to a file under `<WORKSPACE>/` and read that file separately with `read_file`.
- **Do not use `shell_exec` for state-machine logic** (e.g., conditional transitions). Transitions are prompt-driven — keep your reasoning in prose.
