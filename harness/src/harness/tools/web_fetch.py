"""Web-fetch tool — pull a public URL's text content.

Used by states that need to consult external references at runtime:
`locate_recipe` (HF Hub model cards), `generate_preprocess` (dataset schemas).

For `text/html` responses we strip scripts/styles/nav and extract visible text
before returning — HF Hub / GitHub pages are ~80% chrome by byte, and returning
the raw HTML into the agent conversation blows past model context limits fast.
Non-HTML content types (json, markdown, plain text) pass through unchanged.

Not a search engine — the agent must supply the exact URL. A future
`web_search` tool would complement this; for M3 we prioritize `web_fetch`
because URL-based lookup is what verl-harness states actually need.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

import httpx

from harness.tools.base import Tool, ToolContext, ToolError

# Default raw-byte cap on the fetched body. We reduced this from 500 KB in
# M3-T10 after a live smoke pulled 275 KB of HF Hub HTML into a single tool
# result and blew past gpt-4o-mini's 128k context.
_DEFAULT_MAX_BYTES = 200_000
# Post-strip cap on the extracted text handed to the agent. HTML→text usually
# shrinks 5-10x; this cap defends against pathological pages (huge JSON blob
# embedded in a <pre>, minified inline JS the stripper couldn't skip).
_DEFAULT_MAX_TEXT_CHARS = 40_000
_DEFAULT_TIMEOUT_S = 30.0
_ALLOWED_SCHEMES = ("http://", "https://")
_USER_AGENT = "verl-harness-runtime/0.1 (+https://github.com/WuizaKaseiyo/verl-harness)"


# ── HTML → text ──────────────────────────────────────────────────────────────

# Tags whose subtree is dropped entirely.
_HTML_SKIP_TAGS = frozenset(
    {"script", "style", "noscript", "svg", "head", "template", "iframe"}
)
# Tags that force a newline before/after their text.
_HTML_BLOCK_TAGS = frozenset(
    {
        "p", "div", "br", "li", "tr", "td", "th",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "section", "article", "header", "footer", "nav", "main",
        "blockquote", "pre", "hr", "ul", "ol",
    }
)


class _HTMLToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag in _HTML_SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _HTML_BLOCK_TAGS:
            self._out.append("\n")

    def handle_startendtag(self, tag: str, attrs) -> None:  # type: ignore[override]
        # e.g. `<br/>`, `<hr/>`, `<img/>` — treat like open+close
        if tag in _HTML_BLOCK_TAGS:
            self._out.append("\n")

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag in _HTML_SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _HTML_BLOCK_TAGS:
            self._out.append("\n")

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth == 0:
            self._out.append(data)

    def get_text(self) -> str:
        raw = "".join(self._out)
        # Collapse runs of horizontal whitespace within each line; keep
        # newlines as paragraph markers so the agent can still see structure.
        lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in raw.split("\n")]
        # Collapse ≥2 blank lines into 1.
        out: list[str] = []
        blank = True  # start suppressing leading blanks
        for line in lines:
            if line:
                out.append(line)
                blank = False
            elif not blank:
                out.append("")
                blank = True
        return "\n".join(out).strip()


def _html_to_text(html: str) -> str:
    parser = _HTMLToText()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        # Malformed HTML — return whatever we managed to extract rather than
        # bubbling an internal parser exception to the agent.
        pass
    return parser.get_text()


def _is_html(content_type: str) -> bool:
    return content_type.lower().split(";", 1)[0].strip() in {"text/html", "application/xhtml+xml"}


# ── tool ─────────────────────────────────────────────────────────────────────


class WebFetchTool(Tool):
    name = "web_fetch"
    description = (
        "Fetch a public URL and return its visible text (HTML is stripped of "
        "scripts / styles / navigation, non-HTML passes through). Use to read "
        "HuggingFace Hub model cards, dataset schemas, verl trainer docs, or "
        "any public web page cited in the state instructions. Does not execute "
        "JavaScript. Not a search engine — supply the exact URL."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "http:// or https:// URL to fetch.",
            },
            "max_bytes": {
                "type": "integer",
                "description": (
                    f"Raw response size cap before HTML stripping. "
                    f"Default {_DEFAULT_MAX_BYTES}."
                ),
            },
            "max_text_chars": {
                "type": "integer",
                "description": (
                    f"Cap on the extracted text (post-strip) handed back to "
                    f"the agent. Default {_DEFAULT_MAX_TEXT_CHARS}."
                ),
            },
        },
        "required": ["url"],
    }

    def execute(self, input: dict[str, Any], ctx: ToolContext) -> str:
        url = input.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ToolError("`url` must be a non-empty string")
        url = url.strip()
        if not url.startswith(_ALLOWED_SCHEMES):
            raise ToolError(
                f"url must start with http:// or https://, got: {url[:60]}"
            )

        try:
            max_bytes = int(input.get("max_bytes", _DEFAULT_MAX_BYTES))
        except (TypeError, ValueError) as e:
            raise ToolError(f"`max_bytes` must be an integer: {e}") from e
        if max_bytes < 1024:
            raise ToolError(f"`max_bytes` must be at least 1024, got {max_bytes}")

        try:
            max_text_chars = int(input.get("max_text_chars", _DEFAULT_MAX_TEXT_CHARS))
        except (TypeError, ValueError) as e:
            raise ToolError(f"`max_text_chars` must be an integer: {e}") from e
        if max_text_chars < 512:
            raise ToolError(
                f"`max_text_chars` must be at least 512, got {max_text_chars}"
            )

        try:
            with httpx.Client(
                follow_redirects=True,
                timeout=_DEFAULT_TIMEOUT_S,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                r = client.get(url)
                r.raise_for_status()
                body = r.text
        except httpx.TimeoutException:
            raise ToolError(
                f"request timed out after {_DEFAULT_TIMEOUT_S}s"
            ) from None
        except httpx.HTTPStatusError as e:
            raise ToolError(
                f"HTTP {e.response.status_code} {e.response.reason_phrase}"
            ) from e
        except httpx.HTTPError as e:
            raise ToolError(f"fetch failed: {type(e).__name__}: {e}") from e

        raw_len = len(body)
        content_type = r.headers.get("content-type", "unknown")
        truncated_bytes = raw_len > max_bytes
        raw = body[:max_bytes]

        if _is_html(content_type):
            text = _html_to_text(raw)
            mode = "html-stripped"
        else:
            text = raw
            mode = "raw"

        truncated_chars = len(text) > max_text_chars
        if truncated_chars:
            text = (
                text[:max_text_chars]
                + f"\n\n[... {len(text) - max_text_chars} more chars truncated; "
                f"raise max_text_chars to fetch more]"
            )
        elif truncated_bytes:
            text += (
                f"\n\n[... {raw_len - max_bytes} more bytes truncated at fetch "
                f"stage; raise max_bytes to fetch more]"
            )

        header = (
            f"# {r.url}\n"
            f"Status: {r.status_code}\n"
            f"Content-Type: {content_type}\n"
            f"Length: {raw_len} bytes raw, {len(text)} chars returned ({mode})\n\n"
        )
        return header + text + "\n"
