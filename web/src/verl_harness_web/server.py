"""Starlette server for the verl-harness dashboard.

Two layers of endpoints:

- generic harness endpoints: /api/harness, /api/run,
  /api/state/{name}, /api/skill, /api/file
- verl-specific run endpoints: /api/progress, /api/anomalies, /api/job,
  /api/logs, /api/summary
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import (FileResponse, JSONResponse, PlainTextResponse,
                                 StreamingResponse)
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .parser import (parse_harness, parse_state_log, read_anomalies,
                     read_job_info, read_job_log_tail, read_job_status,
                     read_progress, read_progress_csv, read_reflect_state,
                     read_summary)
from .render import (OVERVIEW_NODE, compile_overview, compile_skill_folder,
                     compile_state, harness_to_mermaid)

STATIC_DIR = Path(__file__).parent / "static"


class CacheControlMiddleware:
    """No-cache on every HTTP response; never buffers the SSE stream."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_wrap(message):
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", [])
                           if k.lower() != b"cache-control"]
                headers.append((b"cache-control", b"no-cache"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrap)


def _is_editable(rel: str) -> bool:
    """Editable: task-overview.md, states/*.md, skills/**/*.md."""
    norm = rel.replace("\\", "/").lstrip("/")
    if norm == "task-overview.md":
        return True
    parts = norm.split("/")
    return len(parts) >= 2 and parts[0] in ("states", "skills") and norm.endswith(".md")


def create_app(harness_root: Path, live: bool = True) -> Starlette:
    harness_root = harness_root.resolve()

    def _safe(rel: str) -> Path:
        p = (harness_root / rel).resolve()
        if harness_root not in p.parents and p != harness_root:
            raise ValueError("path escapes harness root")
        return p

    # ---------- pages -------------------------------------------------------
    async def index(_: Request):
        return FileResponse(STATIC_DIR / "index.html")

    async def healthz(_: Request):
        return PlainTextResponse("ok")

    # ---------- read endpoints (generic harness) ----------------------------
    async def api_config(_: Request):
        return JSONResponse({"live": live, "root": str(harness_root)})

    async def api_harness(_: Request):
        h = parse_harness(harness_root)
        return JSONResponse({
            "title": h.title,
            "root": str(h.root),
            "starting_state": h.starting_state,
            "hitl": h.hitl,
            "required_capabilities": h.required_capabilities,
            "state_count": len(h.states),
            "states": sorted(h.states.keys()),
            "terminal_states": sorted(n for n, s in h.states.items() if s.is_terminal),
            "skills": h.list_skill_folders(),
            "mermaid": harness_to_mermaid(h),
            "overview_node": OVERVIEW_NODE,
        })

    def _resolve_run(h, request: Request):
        """Honor ?run_id=<name> if it points at an existing run dir; else latest."""
        rid = (request.query_params.get("run_id") or "").strip()
        if rid:
            cand = h.runs_dir / rid
            if cand.is_dir() and cand.parent == h.runs_dir:
                return cand
        return h.latest_run()

    async def api_runs(_: Request):
        h = parse_harness(harness_root)
        return JSONResponse({"runs": [
            {"id": r.name, "meta": h.run_meta(r)} for r in h.list_runs()
        ]})

    async def api_run(request: Request):
        h = parse_harness(harness_root)
        run = _resolve_run(h, request)
        if run is None:
            return JSONResponse({"run_id": None, "status": "idle",
                                 "entries": [], "current": None, "visited": []})
        meta = h.run_meta(run)
        entries = parse_state_log(h.state_log_path(run))
        # Status fallback chain: meta.json → state_log entries (live) → job_status.md
        # (poller-written terminal) → idle. Without the job_status fallback, completed
        # runs that weren't driven by the FSM state-log appear forever as "idle".
        if meta.get("status"):
            status = meta["status"]
        elif entries:
            status = "running"
        else:
            js = read_job_status(run)
            status = js.get("status", "idle")
        current = entries[-1]["state"] if (entries and status == "running") else None
        return JSONResponse({
            "run_id": run.name,
            "status": status,
            "meta": meta,
            "entries": entries,
            "current": current,
            "last_state": entries[-1]["state"] if entries else None,
            "last_step": entries[-1]["step"] if entries else 0,
            "visited": sorted({e["state"] for e in entries}),
        })

    async def api_state(request: Request):
        name = request.path_params["name"]
        h = parse_harness(harness_root)
        if name == OVERVIEW_NODE:
            return JSONResponse({"name": name, "kind": "overview",
                                 "editable": True,
                                 "file": "task-overview.md",
                                 "compiled": compile_overview(h)})
        if name not in h.states:
            return JSONResponse({"error": "no such state"}, status_code=404)
        st = h.states[name]
        return JSONResponse({
            "name": name,
            "kind": "state",
            "editable": True,
            "file": f"states/{name}.md",
            "compiled": compile_state(h, st),
            "skills": st.skills,
            "is_terminal": st.is_terminal,
        })

    async def api_skill(request: Request):
        path = request.query_params.get("path", "")
        h = parse_harness(harness_root)
        return JSONResponse({
            "path": path,
            "compiled": compile_skill_folder(h, path),
            "files": [f for f, _ in h.read_skill_files(path)],
            "used_by": h.states_using_skill(path),
        })

    # ---------- generic file r/w (states + skills only) ---------------------
    async def api_file(request: Request):
        path = request.query_params.get("path", "")
        try:
            fp = _safe(path)
        except ValueError:
            return JSONResponse({"error": "bad path"}, status_code=400)
        if not fp.exists() or fp.suffix != ".md":
            return JSONResponse({"error": "no such file"}, status_code=404)
        return JSONResponse({"path": path, "content": fp.read_text(encoding="utf-8")})

    async def put_file(request: Request):
        body = await request.json()
        path = body.get("path", "")
        content = body.get("content", "")
        try:
            fp = _safe(path)
        except ValueError:
            return JSONResponse({"error": "bad path"}, status_code=400)
        if not _is_editable(path):
            return JSONResponse(
                {"error": "only state and skill .md files can be edited"},
                status_code=400)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return JSONResponse({"ok": True, "path": path})

    # ---------- verl-specific run endpoints ---------------------------------
    async def api_progress(request: Request):
        h = parse_harness(harness_root)
        run = _resolve_run(h, request)
        if run is None:
            return JSONResponse({"rows": 0, "columns": [], "series": {}})
        # Prefer stdout-derived (live, multi-row) over the static csv
        return JSONResponse(read_progress(run))

    async def api_anomalies(request: Request):
        h = parse_harness(harness_root)
        run = _resolve_run(h, request)
        return JSONResponse({"anomalies": read_anomalies(run) if run else []})

    async def api_job(request: Request):
        h = parse_harness(harness_root)
        run = _resolve_run(h, request)
        if run is None:
            return JSONResponse({"info": {}, "status": {}})
        return JSONResponse({
            "info": read_job_info(run),
            "status": read_job_status(run),
        })

    async def api_logs(request: Request):
        h = parse_harness(harness_root)
        run = _resolve_run(h, request)
        if run is None:
            return JSONResponse({"size": 0, "since": 0, "content": ""})
        since = int(request.query_params.get("since", "0"))
        return JSONResponse(read_job_log_tail(run, since_byte=since))

    async def api_summary(request: Request):
        h = parse_harness(harness_root)
        run = _resolve_run(h, request)
        if run is None:
            return JSONResponse({"summary": ""})
        return JSONResponse({"summary": read_summary(run)})

    async def api_reflect(request: Request):
        h = parse_harness(harness_root)
        run = _resolve_run(h, request)
        if run is None:
            return JSONResponse({"present": False})
        return JSONResponse(read_reflect_state(run))

    # ---------- SSE live stream --------------------------------------------
    async def events(_: Request):
        async def stream():
            yield "event: hello\ndata: connected\n\n"
            if not live:
                return
            from watchfiles import awatch
            try:
                async for _changes in awatch(harness_root):
                    yield "event: changed\ndata: fs\n\n"
            except (asyncio.CancelledError, GeneratorExit):
                return
            except Exception:
                return
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    routes = [
        Route("/", index),
        Route("/healthz", healthz),
        Route("/api/config", api_config),
        Route("/api/harness", api_harness),
        Route("/api/runs", api_runs),
        Route("/api/run", api_run),
        Route("/api/state/{name}", api_state),
        Route("/api/skill", api_skill),
        Route("/api/file", api_file),
        Route("/api/file", put_file, methods=["PUT"]),
        Route("/api/progress", api_progress),
        Route("/api/anomalies", api_anomalies),
        Route("/api/job", api_job),
        Route("/api/logs", api_logs),
        Route("/api/summary", api_summary),
        Route("/api/reflect", api_reflect),
        Route("/events", events),
        Mount("/static", app=StaticFiles(directory=STATIC_DIR), name="static"),
    ]
    return Starlette(routes=routes, middleware=[Middleware(CacheControlMiddleware)])
