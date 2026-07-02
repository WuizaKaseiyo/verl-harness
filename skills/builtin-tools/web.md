# Web Built-in Tools

Two tools cover `web.search` and `web.fetch`. Implementations in `tools/web.py`. Both use only the Python standard library (`urllib`, `html.parser`) — no third-party dependencies.

---

## `search_web` — Google search via Serper.dev

Issues a Google search through [Serper.dev](https://serper.dev) and returns the organic results as normalized JSON.

**Prerequisite:** `SERPER_API_KEY` must be in the environment. Without it, the call raises `RuntimeError: SERPER_API_KEY is not set` *before* any network request.

**Parameters**

| Name | Type | Default | Required | Description |
|---|---|---|---|---|
| `query` | string | — | ✓ | Search query |
| `max_results` | int | `10` |   | Number of organic results to return (must be ≥ 1) |
| `location` | string | — |   | Optional Serper location string (e.g., `"United States"` or a city) |
| `api_key` | string | — |   | Override the env var (rarely needed; prefer the env var) |

**Returns**

```json
{
  "tool": "search_web",
  "capability": "web.search",
  "provider": "serper",
  "query": "verl-harness agent workflow",
  "location": null,
  "requested_results": 10,
  "fetched_at": "2026-05-13T12:00:00+00:00",
  "results": [
    {
      "rank": 1,
      "title": "...",
      "url": "https://...",
      "displayed_url": "example.com/...",
      "snippet": "...",
      "source": "",
      "date": ""
    }
  ]
}
```

Errors:

- `RuntimeError("SERPER_API_KEY is not set")` — missing API key.
- `ValueError("max_results must be at least 1")` — bad parameter.
- `RuntimeError("Serper error: ...")` — Serper returned an `error` field (quota, malformed query, etc.).
- `RuntimeError("HTTP <code>: ...")` — non-2xx from Serper (e.g., 401 = bad key, 429 = quota exhausted).
- `RuntimeError("request failed: ...")` — DNS / network failure.

**Native equivalents:** Claude Code `WebSearch`, Cursor's web search tool, dedicated MCP search servers.

**CLI**

```bash
python tools/registry.py search_web \
  '{"query":"finite state machine LLM","max_results":10}' \
  --workspace "<WORKSPACE>" \
  --output "<WORKSPACE>/search/q01.json"
```

Always pass `--output` for `search_web` — full result sets are bulky (10 entries × snippet + URL) and clog the conversation if printed inline.

---

## `fetch_webpage` — fetch a URL and extract readable text

Performs an HTTP GET with `User-Agent: verl-harness/0.1`, follows redirects, and extracts the readable text + links if the response is HTML. Non-HTML responses are returned as raw text (truncated).

**Parameters**

| Name | Type | Default | Required | Description |
|---|---|---|---|---|
| `url` | string | — | ✓ | HTTP or HTTPS URL |
| `max_chars` | int | `15000` |   | Cap on extracted text length |

**Returns**

```json
{
  "tool": "fetch_webpage",
  "capability": "web.fetch",
  "url": "https://example.com/article",
  "final_url": "https://example.com/article",
  "status": 200,
  "content_type": "text/html; charset=utf-8",
  "title": "Article title",
  "text": "Cleaned readable text up to max_chars...",
  "links": [{"href": "https://other.com", "text": "Other site"}],
  "fetched_at": "2026-05-13T12:00:00+00:00"
}
```

**HTML extraction details (the `_TextExtractor` HTMLParser):**

- Skips content inside `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>`, `<aside>`, `<noscript>` — these almost always contain UI chrome, not article text.
- Treats `<p>`, `<div>`, `<section>`, `<article>`, `<br>`, `<li>`, `<h1>-<h3>` as paragraph breaks.
- Captures the first `<title>` as `title`.
- Captures `<a href>` and the anchor text into `links`, up to 100 per page.
- Decodes HTML entities (`&amp;` → `&`, etc.).
- Collapses runs of whitespace.

For non-HTML content types, `title` is empty, `links` is `[]`, and `text` is the raw body truncated to `max_chars`.

Errors:

- `RuntimeError("HTTP <code>: ...")` — non-2xx response.
- `RuntimeError("request failed: ...")` — DNS / connection failure / SSL error.

Timeout is fixed at 30 seconds.

**Native equivalents:** Claude Code `WebFetch`, Cursor's URL fetcher, dedicated MCP browse servers.

**CLI**

```bash
python tools/registry.py fetch_webpage \
  '{"url":"https://example.com/article","max_chars":15000}' \
  --workspace "<WORKSPACE>" \
  --output "<WORKSPACE>/fetches/example.json"
```

---

## When to use `search_web` vs `fetch_webpage`

- **`search_web`** — discovery. You have a topic and want to know what is on the web about it. Returns 10 candidates, you pick the promising ones.
- **`fetch_webpage`** — extraction. You have a specific URL (often from `search_web` results or a citation) and want its content.

Typical pair: `search_web "FSM LLM agent reliability"` → pick top-3 URLs from `results[].url` → `fetch_webpage` each one → grep through the extracted text. Save each `fetch_webpage` JSON under `<WORKSPACE>/fetches/` so a later state can re-read it.

---

## Quotas, costs, and politeness

- **Serper is metered.** Each `search_web` call consumes one search credit. Serper offers 2,500 free credits to start; paid plans scale from there. Plan harness runs accordingly; prefer caching results to `<WORKSPACE>/` and re-reading them on loop-back rather than re-searching.
- **`fetch_webpage` is direct HTTP** — no quota, but the target site has rate limits. Hammering one host from a tight loop is the harness author's responsibility to avoid; this tool has no built-in throttle.
- **No retry built in.** A failed request raises immediately. If you need retries with backoff, wrap the call in a state with an explicit `## Next States` transition back to the same state on error.

---

## Things you must not do

- **Do not call `fetch_webpage` on non-HTTP URLs** (file://, ftp://, javascript:). The underlying `urllib.request.urlopen` will reject them, but you should not even try.
- **Do not feed the raw `text` field of `fetch_webpage` into another LLM call without checking length first.** HTML pages can be massive even after extraction; respect `max_chars` and downstream context budgets.
- **Do not call `search_web` from inside a state's transition condition.** Transitions are evaluated in prose — the search itself belongs to the state's work, not to the decision about where to go next.
