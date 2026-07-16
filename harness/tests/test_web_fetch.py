"""web_fetch tool — httpx-based URL fetcher.

Live HTTP is exercised opt-in via VHR_LIVE_HTTP=1 (hits example.com). Unit
tests use httpx MockTransport so they're deterministic + offline.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from harness.tools import ToolContext, ToolError
from harness.tools.web_fetch import WebFetchTool


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ToolContext(cwd=tmp_path, workdir=tmp_path, run_id="T", workspace=ws)


# ── input validation ─────────────────────────────────────────────────────

def test_missing_url_errors(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="url"):
        WebFetchTool().execute({}, ctx)


def test_empty_url_errors(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="non-empty"):
        WebFetchTool().execute({"url": "  "}, ctx)


def test_non_http_scheme_errors(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="http://"):
        WebFetchTool().execute({"url": "ftp://foo/bar"}, ctx)
    with pytest.raises(ToolError, match="http://"):
        WebFetchTool().execute({"url": "file:///etc/passwd"}, ctx)


def test_bad_max_bytes_errors(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="max_bytes"):
        WebFetchTool().execute(
            {"url": "https://example.com", "max_bytes": 100}, ctx
        )
    with pytest.raises(ToolError, match="max_bytes"):
        WebFetchTool().execute(
            {"url": "https://example.com", "max_bytes": "not-an-int"}, ctx
        )


# ── mocked httpx transport ────────────────────────────────────────────────

def _mock(handler):
    """Monkeypatch httpx.Client to use MockTransport for handler."""
    orig = httpx.Client

    class PatchedClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)
    return PatchedClient, orig


def test_ok_response_returned(ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><body>hi world</body></html>",
        )
    patched, _orig = _mock(handler)
    monkeypatch.setattr(httpx, "Client", patched)

    out = WebFetchTool().execute({"url": "https://example.com"}, ctx)
    assert "hi world" in out
    assert "Status: 200" in out
    assert "content-type" in out.lower() or "text/html" in out


def test_truncation_appends_marker(ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    huge = "x" * 100_000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=huge)

    patched, _ = _mock(handler)
    monkeypatch.setattr(httpx, "Client", patched)

    out = WebFetchTool().execute(
        {"url": "https://example.com", "max_bytes": 5000},
        ctx,
    )
    assert "more bytes truncated" in out
    assert out.count("x") <= 5100  # content + a bit of overhead


def test_http_error_surfaces(ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    patched, _ = _mock(handler)
    monkeypatch.setattr(httpx, "Client", patched)

    with pytest.raises(ToolError, match="404"):
        WebFetchTool().execute({"url": "https://example.com/nope"}, ctx)


def test_user_agent_sent(ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    seen_ua: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_ua.append(request.headers.get("user-agent", ""))
        return httpx.Response(200, text="ok")

    patched, _ = _mock(handler)
    monkeypatch.setattr(httpx, "Client", patched)

    WebFetchTool().execute({"url": "https://example.com"}, ctx)
    assert seen_ua
    assert "verl-harness" in seen_ua[0]


# ── HTML → text stripping ─────────────────────────────────────────────────


def test_html_scripts_and_styles_dropped(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """<script> and <style> subtrees must not leak into agent context — they
    can be tens of KB of minified JS that eats the context window."""
    html = (
        "<html><head><style>body{color:red}</style></head>"
        "<body>"
        "<script>var evil='SHOULD_NOT_APPEAR';window.foo=1;</script>"
        "<p>hello world</p>"
        "<script type='application/json'>{\"MORE_JS\":\"nope\"}</script>"
        "<p>keep me too</p>"
        "</body></html>"
    )

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"}, text=html)

    patched, _ = _mock(handler)
    monkeypatch.setattr(httpx, "Client", patched)

    out = WebFetchTool().execute({"url": "https://example.com"}, ctx)
    assert "hello world" in out
    assert "keep me too" in out
    assert "SHOULD_NOT_APPEAR" not in out
    assert "MORE_JS" not in out
    assert "color:red" not in out
    assert "html-stripped" in out


def test_html_navigation_and_svg_dropped(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    html = (
        "<html><body>"
        "<svg><path d='M0 0h100v100H0z'/><text>SVG_TEXT_NOISE</text></svg>"
        "<noscript>Please enable JS</noscript>"
        "<iframe src='//tracker.example.com/frame.html'>iframe body</iframe>"
        "<article>the real content</article>"
        "</body></html>"
    )

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"}, text=html)

    patched, _ = _mock(handler)
    monkeypatch.setattr(httpx, "Client", patched)

    out = WebFetchTool().execute({"url": "https://example.com"}, ctx)
    assert "the real content" in out
    assert "SVG_TEXT_NOISE" not in out
    assert "iframe body" not in out
    assert "Please enable JS" not in out


def test_html_paragraph_breaks_preserved(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    html = "<html><body><p>one</p><p>two</p><h1>three</h1></body></html>"

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"}, text=html)

    patched, _ = _mock(handler)
    monkeypatch.setattr(httpx, "Client", patched)

    out = WebFetchTool().execute({"url": "https://example.com"}, ctx)
    body = out.split("\n\n", 1)[1] if "\n\n" in out else out
    assert "one" in body and "two" in body and "three" in body
    # Paragraphs on separate lines, not concatenated as "onetwothree"
    assert "onetwo" not in body
    assert "twothree" not in body


def test_html_shrinks_bulk_content(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of HTML stripping: 275KB of tags collapses to a few KB
    of text, keeping the model context under control."""
    # Simulate an HF-Hub-shaped page: lots of chrome, a little content.
    chrome = "<script>" + ("x=1;" * 5000) + "</script>" + "<nav>" + ("<a>link</a>" * 500) + "</nav>"
    real = "<article><h1>gsm8k</h1><p>Grade-school math word problems.</p></article>"
    html = f"<html><head><style>{'x{}' * 200}</style></head><body>{chrome}{real}</body></html>"

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"}, text=html)

    patched, _ = _mock(handler)
    monkeypatch.setattr(httpx, "Client", patched)

    out = WebFetchTool().execute({"url": "https://example.com"}, ctx)
    assert "gsm8k" in out
    assert "Grade-school math word problems." in out
    assert "x=1;" not in out
    # The nav-link text may leak (we don't drop <a>) but the whole output should
    # still be dramatically smaller than the raw HTML.
    assert len(out) < len(html) // 5, f"expected >5x shrink, got {len(out)} / {len(html)}"


def test_non_html_content_type_passes_through(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """application/json and text/plain should NOT be html-stripped."""
    body = '{"tag": "<script>not-actually-html</script>", "n": 42}'

    def handler(request):
        return httpx.Response(200, headers={"content-type": "application/json"}, text=body)

    patched, _ = _mock(handler)
    monkeypatch.setattr(httpx, "Client", patched)

    out = WebFetchTool().execute({"url": "https://example.com/x.json"}, ctx)
    assert "<script>not-actually-html</script>" in out
    assert "(raw)" in out or "raw)" in out


def test_html_text_cap_appends_marker(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-strip text over max_text_chars gets a truncation marker."""
    html = "<html><body><p>" + ("word " * 1000) + "</p></body></html>"

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/html"}, text=html)

    patched, _ = _mock(handler)
    monkeypatch.setattr(httpx, "Client", patched)

    out = WebFetchTool().execute(
        {"url": "https://example.com", "max_text_chars": 800}, ctx
    )
    assert "more chars truncated" in out


def test_bad_max_text_chars_errors(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="max_text_chars"):
        WebFetchTool().execute(
            {"url": "https://example.com", "max_text_chars": 100}, ctx
        )


# ── registry integration ─────────────────────────────────────────────────

def test_web_fetch_in_default_registry() -> None:
    from harness.tools import default_registry
    assert "web_fetch" in default_registry().names()


# ── live smoke (opt-in) ──────────────────────────────────────────────────

_LIVE = os.environ.get("VHR_LIVE_HTTP") == "1"


@pytest.mark.skipif(not _LIVE, reason="set VHR_LIVE_HTTP=1 to run")
def test_live_example_com(ctx: ToolContext) -> None:
    out = WebFetchTool().execute({"url": "https://example.com"}, ctx)
    assert "Example Domain" in out
