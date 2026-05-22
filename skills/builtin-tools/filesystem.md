# Filesystem Built-in Tools

Six tools cover `filesystem.read` and `filesystem.write`. Implementations live in `tools/file_ops.py` and dispatch through `tools/registry.py`.

**One rule applies to all six:** every tool takes a `root` argument. Relative paths resolve under `root`; absolute paths are still validated to fall inside `root`. Any path that escapes (via `..` or absolute escape) raises `PermissionError: path resolves outside root`. There is no way around this — design your harness assuming `root` is `<WORKSPACE>` and you operate inside it.

Default ignore set (used by `list_dir` and `grep` directory walks): `.git`, `node_modules`, `__pycache__`, `.venv`, `venv`, `.pytest_cache`, plus any name starting with `.` or `._`.

---

## `list_dir` — list a directory tree

Lists a directory under `root` as an ASCII tree, up to `max_depth` levels deep.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `root` | string | `.` | Allowed root directory |
| `path` | string | `.` | Directory under `root` |
| `max_depth` | int | `2` | Recursion depth (1 = root only, 2 = one level into subdirs) |

**Returns**

```json
{
  "tool": "list_dir",
  "path": "<absolute resolved path>",
  "tree": ["├── file_ops.py", "├── registry.py", "└── README.md"]
}
```

**Native equivalents:** Bash `find -maxdepth N` or `ls -R`, dedicated `ListDir` / `Glob` tools.

**CLI**

```bash
python tools/registry.py list_dir '{"root":"<WORKSPACE>","path":".","max_depth":2}' --workspace "<WORKSPACE>"
```

---

## `read_file` — read a file (optionally a line range)

Read a file under `root`. Use `start_line` / `end_line` (1-indexed, `end_line: -1` means EOF) to slice. The result includes both `content` (raw) and `numbered_content` (with right-aligned 5-digit line numbers) — pick whichever fits your downstream use.

**Parameters**

| Name | Type | Default | Required | Description |
|---|---|---|---|---|
| `path` | string | — | ✓ | File path under `root` |
| `root` | string | `.` |   | Allowed root directory |
| `start_line` | int | `1` |   | First line to read (1-indexed) |
| `end_line` | int | `-1` |   | Last line to read; `-1` = end-of-file |

**Returns**

```json
{
  "tool": "read_file",
  "path": "<absolute path>",
  "start_line": 1, "end_line": 42, "total_lines": 100,
  "content": "raw text...",
  "numbered_content": "    1 | first line\n    2 | second line..."
}
```

Errors: `FileNotFoundError` if missing; `ValueError` if `start_line > total_lines`.

**Native equivalents:** Claude Code `Read`, Cursor `read_file`, any host-native file viewer.

**CLI**

```bash
python tools/registry.py read_file '{"root":"<WORKSPACE>","path":"notes.md","start_line":1,"end_line":50}' --workspace "<WORKSPACE>"
```

---

## `grep` — regex search across files or a text string

Two modes:

- **Files mode** (default): walk `root/path`, search `pattern` in every file matching `file_pattern`. Skips the default ignore set and known binary extensions (`.png`, `.jpg`, `.jpeg`, `.gif`, `.ico`, `.pdf`, `.zip`, `.pyc`).
- **Text mode**: pass `text` instead of walking the filesystem. Useful for grep-ing the output of another tool without writing it to disk first.

**Parameters**

| Name | Type | Default | Required | Description |
|---|---|---|---|---|
| `pattern` | string | — | ✓ | Python `re` regular expression |
| `root` | string | `.` |   | Allowed root |
| `path` | string | `.` |   | Directory under `root` to walk |
| `file_pattern` | string | `*` |   | Glob (e.g. `*.md`, `*.py`) |
| `text` | string | — |   | If provided, search this string instead of files |
| `max_matches` | int | `100` |   | Stop after this many matches |

**Returns (files mode)**

```json
{
  "tool": "grep", "mode": "files", "root": "...", "pattern": "...", "file_pattern": "*.md",
  "matches": [{"path": "/abs/path", "relative_path": "subdir/file.md", "line": 12, "text": "matched line"}],
  "truncated": false
}
```

**Returns (text mode)**

```json
{
  "tool": "grep", "mode": "text", "pattern": "...",
  "matches": [{"line": 3, "text": "matched line"}],
  "truncated": false
}
```

**Native equivalents:** Claude Code `Grep` / `Bash grep`, dedicated `SearchCode` / ripgrep wrappers.

**CLI**

```bash
python tools/registry.py grep '{"root":"<WORKSPACE>","path":".","pattern":"TODO","file_pattern":"*.md","max_matches":50}' --workspace "<WORKSPACE>"
```

---

## `file_create` — create or overwrite a file

Creates a file under `root` (with parent directories if needed). Refuses to overwrite an existing file unless `overwrite: true`.

**Parameters**

| Name | Type | Default | Required | Description |
|---|---|---|---|---|
| `path` | string | — | ✓ | File path under `root` |
| `root` | string | `.` |   | Allowed root |
| `content` | string | `""` |   | UTF-8 contents to write |
| `overwrite` | bool | `false` |   | If `false` and file exists, raises `FileExistsError` |

**Returns**

```json
{"tool": "file_create", "path": "<absolute>", "bytes": 1234}
```

**Native equivalents:** Claude Code `Write`, Cursor file-creation tool.

**CLI**

```bash
python tools/registry.py file_create '{"root":"<WORKSPACE>","path":"deliverables/summary.md","content":"# Summary\n\nText...","overwrite":true}' --workspace "<WORKSPACE>"
```

---

## `append_file` — append to a file

Appends UTF-8 text to a file under `root`. Creates the file (and parents) if missing.

**Parameters**

| Name | Type | Default | Required | Description |
|---|---|---|---|---|
| `path` | string | — | ✓ | File path under `root` |
| `root` | string | `.` |   | Allowed root |
| `content` | string | `""` |   | Text to append |

**Returns**

```json
{"tool": "append_file", "path": "<absolute>", "bytes": 56}
```

**Native equivalents:** Claude Code `Edit` (in append mode) or `Bash echo >> file`.

**CLI**

```bash
python tools/registry.py append_file '{"root":"<WORKSPACE>","path":"logs/state_log.md","content":"- [...] #03 entered ...\n"}' --workspace "<WORKSPACE>"
```

This is the right tool for `state_log.md` appends — the file grows one line per state entry, and `append_file` matches that semantics exactly without re-reading the whole file.

---

## `mkdir` — create a directory (recursive, idempotent)

Creates a directory under `root` including any missing parent directories. Idempotent — succeeds even if the directory already exists.

**Parameters**

| Name | Type | Default | Required | Description |
|---|---|---|---|---|
| `path` | string | — | ✓ | Directory path under `root` |
| `root` | string | `.` |   | Allowed root |

**Returns**

```json
{"tool": "mkdir", "path": "<absolute>"}
```

**Native equivalents:** `Bash mkdir -p` or any host directory-creation tool.

**CLI**

```bash
python tools/registry.py mkdir '{"root":"<WORKSPACE>","path":"deliverables/round-2"}' --workspace "<WORKSPACE>"
```

---

## Sandboxing — how `_resolve_under_root()` works

The shared sandbox check, in `tools/file_ops.py`:

1. `root_path = Path(root).resolve()` — turn `root` into an absolute, symlink-resolved path.
2. If `path` is relative, prepend `root_path`.
3. `resolved = path.resolve()` — collapse `..`, follow symlinks.
4. `resolved.relative_to(root_path)` — if this raises `ValueError`, the resolved path is outside `root`, so re-raise as `PermissionError`.

This means every escape vector (absolute path, `..` traversal, symlink) is caught at the same gate. It is the same guarantee a chroot would give you, in 10 lines of stdlib Python.
