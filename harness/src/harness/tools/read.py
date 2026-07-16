"""Read tool — line-numbered file content with offset/limit.

Guards against pathological inputs that would blow the model's context window:
  - refuse binary files (sniff first 8 KB for NUL bytes)
  - cap the returned text at `_MAX_OUTPUT_BYTES`, even when `limit` says more
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.tools.base import Tool, ToolContext, ToolError

_DEFAULT_LIMIT = 2000
# Post-line-numbering output ceiling. Handed back to the agent verbatim, so it
# gates the whole downstream conversation size. Matches BashTool's ceiling.
_MAX_OUTPUT_BYTES = 100 * 1024  # 100 KB
# Bytes sampled at the head of the file for binary detection. Parquet, npz,
# safetensors, images, and archive formats all place either a magic byte or
# a length prefix within the first few KB.
_BINARY_SNIFF_BYTES = 8 * 1024
# Well-known binary extensions we refuse without even opening the file.
_BINARY_EXTENSIONS = frozenset({
    ".parquet", ".arrow", ".feather", ".pkl", ".pickle",
    ".npy", ".npz", ".safetensors", ".pt", ".bin", ".ckpt", ".pth", ".gguf",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".zst",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".pdf",
    ".mp3", ".mp4", ".wav", ".ogg", ".flac", ".webm",
    ".so", ".dll", ".dylib", ".exe", ".class", ".pyc", ".o", ".a",
})


def _looks_binary(sample: bytes) -> bool:
    if not sample:
        return False
    # NUL byte is a near-certain binary marker in any text encoding we support.
    if b"\x00" in sample:
        return True
    # High rate of undecodable bytes = probably binary.
    try:
        sample.decode("utf-8")
        return False
    except UnicodeDecodeError:
        # Retry with latin-1 tolerance and count high-bit noise. >30% high-bit
        # non-printable bytes in the first 8 KB is a strong binary signal.
        noise = sum(1 for b in sample if b < 9 or (13 < b < 32 and b != 27))
        return noise / max(len(sample), 1) > 0.30


class ReadTool(Tool):
    name = "read"
    description = (
        "Read a UTF-8 text file. Output is line-numbered (`{lineno}\\t{line}`), "
        "starting at 1-based line `offset` (default 1) and returning at most "
        "`limit` lines (default 2000). Refuses binary files "
        "(parquet / safetensors / pickles / archives / images / etc.) — use "
        "the `bash` tool with a proper inspector (pyarrow, torch.load, jq) "
        "instead. Overall output is capped at 100 KB regardless of `limit`."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file.",
            },
            "offset": {
                "type": "integer",
                "description": "1-based first line to include (default 1).",
            },
            "limit": {
                "type": "integer",
                "description": f"Max lines to return (default {_DEFAULT_LIMIT}).",
            },
        },
        "required": ["file_path"],
    }

    def execute(self, input: dict[str, Any], ctx: ToolContext) -> str:
        raw_path = input.get("file_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ToolError("`file_path` must be a non-empty string")

        path = Path(raw_path)
        if not path.is_absolute():
            path = (ctx.cwd / path).resolve()

        if not path.exists():
            raise ToolError(f"file not found: {path}")
        if not path.is_file():
            raise ToolError(f"not a file: {path}")

        offset = int(input.get("offset", 1))
        limit = int(input.get("limit", _DEFAULT_LIMIT))
        if offset < 1:
            raise ToolError(f"`offset` must be >= 1, got {offset}")
        if limit < 1:
            raise ToolError(f"`limit` must be >= 1, got {limit}")

        # Extension-based short-circuit — cheaper than opening the file.
        if path.suffix.lower() in _BINARY_EXTENSIONS:
            raise ToolError(
                f"refusing to read binary file (extension {path.suffix!r}): "
                f"{path}. Use the `bash` tool with a proper inspector "
                f"(pyarrow, torch.load, jq, `file`, `head -c`) instead."
            )

        try:
            with path.open("rb") as fp:
                sample = fp.read(_BINARY_SNIFF_BYTES)
        except OSError as e:
            raise ToolError(f"read failed: {e}") from e

        if _looks_binary(sample):
            raise ToolError(
                f"refusing to read binary file (NUL bytes / undecodable UTF-8 "
                f"detected in first {len(sample)} bytes): {path}. Use the "
                f"`bash` tool with a proper inspector instead."
            )

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise ToolError(f"read failed: {e}") from e

        lines = text.splitlines()
        selected = lines[offset - 1 : offset - 1 + limit]

        if not selected:
            return f"[empty range: file has {len(lines)} lines, offset {offset}]"

        rendered_lines: list[str] = []
        rendered_bytes = 0
        capped_at_bytes: int | None = None
        for i, line in enumerate(selected):
            rendered = f"{offset + i}\t{line}"
            # +1 for the joining newline that will follow.
            new_bytes = rendered_bytes + len(rendered) + 1
            if new_bytes > _MAX_OUTPUT_BYTES:
                capped_at_bytes = offset + i
                break
            rendered_lines.append(rendered)
            rendered_bytes = new_bytes

        out = "\n".join(rendered_lines)
        remaining_lines = len(lines) - (offset - 1 + len(rendered_lines))
        if capped_at_bytes is not None:
            out += (
                f"\n[... capped at {_MAX_OUTPUT_BYTES // 1024} KB "
                f"(line {capped_at_bytes} of {len(lines)}); "
                f"raise `offset` or use `bash` to slice further]"
            )
        elif offset - 1 + limit < len(lines):
            out += (
                f"\n[... {remaining_lines} more lines; "
                f"raise limit or offset to continue]"
            )
        return out + "\n"
