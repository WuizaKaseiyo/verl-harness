"""Web built-in tools."""
from __future__ import annotations

import html
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any


SERPER_ENDPOINT = "https://google.serper.dev/search"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.links: list[dict[str, str]] = []
        self._chunks: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav", "footer", "header", "aside", "noscript"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "a":
            attrs_dict = {key: value or "" for key, value in attrs}
            href = attrs_dict.get("href")
            if href:
                self.links.append({"href": href, "text": ""})

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer", "header", "aside", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "section", "article", "br", "li", "h1", "h2", "h3"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        text = html.unescape(data).strip()
        if not text:
            return
        if self._in_title and not self.title:
            self.title = text
        if self._skip_depth:
            return
        self._chunks.append(text)
        if self.links and not self.links[-1]["text"]:
            self.links[-1]["text"] = text

    @property
    def text(self) -> str:
        content = " ".join(self._chunks)
        content = re.sub(r"\s*\n\s*", "\n", content)
        content = re.sub(r"[ \t]{2,}", " ", content)
        return content.strip()


def _request(
    url: str,
    *,
    timeout: int = 30,
    data: bytes | None = None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[str, int, str, str]:
    headers = {
        "User-Agent": "FastHarness/0.1",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            return (
                raw.decode(charset, errors="replace"),
                response.status,
                response.geturl(),
                response.headers.get("content-type", ""),
            )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc.reason}") from exc


def search_web(
    query: str,
    *,
    max_results: int = 10,
    location: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    key = api_key or os.environ.get("SERPER_API_KEY")
    if not key:
        raise RuntimeError("SERPER_API_KEY is not set")
    if max_results < 1:
        raise ValueError("max_results must be at least 1")

    payload: dict[str, Any] = {"q": query, "num": max_results}
    if location:
        payload["location"] = location
    body_bytes = json.dumps(payload).encode("utf-8")

    body, _status, _final_url, _content_type = _request(
        SERPER_ENDPOINT,
        data=body_bytes,
        extra_headers={"X-API-KEY": key, "Content-Type": "application/json"},
    )
    data = json.loads(body)
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"Serper error: {data['error']}")
    organic = data.get("organic") or []
    results = []
    for idx, item in enumerate(organic[:max_results], start=1):
        results.append(
            {
                "rank": idx,
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "displayed_url": item.get("displayedLink", ""),
                "snippet": item.get("snippet", ""),
                "source": item.get("source", ""),
                "date": item.get("date", ""),
            }
        )
    return {
        "tool": "search_web",
        "capability": "web.search",
        "provider": "serper",
        "query": query,
        "location": location,
        "requested_results": max_results,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }


def fetch_webpage(url: str, *, max_chars: int = 15000) -> dict[str, Any]:
    body, status, final_url, content_type = _request(url)
    fetched_at = datetime.now(timezone.utc).isoformat()
    if "html" in content_type.lower():
        parser = _TextExtractor()
        parser.feed(body)
        text = parser.text[:max_chars]
        links = parser.links[:100]
        title = parser.title
    else:
        title = ""
        text = body[:max_chars]
        links = []
    return {
        "tool": "fetch_webpage",
        "capability": "web.fetch",
        "url": url,
        "final_url": final_url,
        "status": status,
        "content_type": content_type,
        "title": title,
        "text": text,
        "links": links,
        "fetched_at": fetched_at,
    }
