"""HITL endpoint tests — /api/hitl/pending + POST /api/hitl/<request_id>.

These pair with harness/src/harness/hitl.py::EventPrompter: the runtime
drops request files under workspace/hitl/requests/, and the dashboard
POSTs decisions back through these endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from verl_harness_web.server import create_app


@pytest.fixture
def harness_root(tmp_path: Path) -> Path:
    """A minimal harness spec dir with states/, skills/, and one run.

    The parser's parse_harness() needs at least a states/ dir and CLAUDE.md;
    the HITL endpoints only need runs/<id>/workspace/hitl/{requests,decisions}/.
    """
    (tmp_path / "states").mkdir()
    (tmp_path / "states" / "intake.md").write_text(
        "# intake\n\n## Description\n\nStub.\n\n"
        "## Skills\n\n- global\n\n"
        "## Hand-off Points\n\n- **stub**. Nothing here.\n\n"
        "## Next States\n\n### finalize\n**Condition:** always.\n**Deliverables:** none.\n"
    )
    (tmp_path / "states" / "finalize.md").write_text(
        "# finalize\n\n## Description\n\nTerminal.\n\n## Skills\n\n- global\n"
    )
    (tmp_path / "skills" / "global").mkdir(parents=True)
    (tmp_path / "skills" / "global" / "default.md").write_text("# global")
    (tmp_path / "CLAUDE.md").write_text("# CLAUDE\n")
    (tmp_path / "task-overview.md").write_text("# overview\n")

    # A run with a workspace, so _resolve_run(latest) returns it
    run = tmp_path / "runs" / "r1"
    (run / "workspace").mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({
        "run_id": "r1", "status": "running", "goal": "t",
        "model": "x/y", "started_at": "2026-07-15T00:00:00Z",
    }))
    return tmp_path


@pytest.fixture
def client(harness_root: Path) -> TestClient:
    app = create_app(harness_root, live=False)
    return TestClient(app)


# ── GET /api/hitl/pending ──────────────────────────────────────────────────


def test_pending_empty_when_no_request_files(client: TestClient) -> None:
    r = client.get("/api/hitl/pending")
    assert r.status_code == 200
    data = r.json()
    assert data["run_id"] == "r1"
    assert data["requests"] == []


def test_pending_returns_no_run_id_when_no_runs(tmp_path: Path) -> None:
    # Harness with no runs/ directory at all
    (tmp_path / "states").mkdir()
    (tmp_path / "states" / "intake.md").write_text(
        "# intake\n\n## Description\n\nStub.\n\n## Skills\n\n- global\n\n"
        "## Hand-off Points\n\n- **x**. y\n\n"
        "## Next States\n\n### finalize\n**Condition:** a.\n**Deliverables:** b.\n"
    )
    (tmp_path / "states" / "finalize.md").write_text(
        "# finalize\n\n## Description\n\nTerminal.\n\n## Skills\n\n- global\n"
    )
    (tmp_path / "skills" / "global").mkdir(parents=True)
    (tmp_path / "skills" / "global" / "default.md").write_text("# global")
    (tmp_path / "CLAUDE.md").write_text("# CLAUDE\n")
    (tmp_path / "task-overview.md").write_text("# overview\n")

    c = TestClient(create_app(tmp_path, live=False))
    data = c.get("/api/hitl/pending").json()
    assert data["run_id"] is None
    assert data["requests"] == []


def _write_request(root: Path, request_id: str, body: dict) -> Path:
    reqs = root / "runs" / "r1" / "workspace" / "hitl" / "requests"
    reqs.mkdir(parents=True, exist_ok=True)
    p = reqs / f"{request_id}.json"
    p.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    return p


def test_pending_lists_request_files(harness_root: Path, client: TestClient) -> None:
    _write_request(harness_root, "abc123", {
        "id": "abc123",
        "state": "intake",
        "title": "Confirm intent",
        "description": "Approve to proceed.",
        "is_always_on": False,
        "condition": None,
        "created_ts": 100.0,
    })
    _write_request(harness_root, "def456", {
        "id": "def456",
        "state": "generate_preprocess",
        "title": "Approve script",
        "description": "Always-on gate.",
        "is_always_on": True,
        "condition": None,
        "created_ts": 200.0,
    })
    data = client.get("/api/hitl/pending").json()
    assert data["run_id"] == "r1"
    ids = [r["id"] for r in data["requests"]]
    # Sorted by created_ts ascending
    assert ids == ["abc123", "def456"]
    assert data["requests"][1]["is_always_on"] is True


def test_pending_skips_malformed_json(harness_root: Path, client: TestClient) -> None:
    _write_request(harness_root, "good", {
        "id": "good", "state": "x", "title": "y", "description": "z",
        "is_always_on": False, "condition": None, "created_ts": 1.0,
    })
    reqs = harness_root / "runs" / "r1" / "workspace" / "hitl" / "requests"
    (reqs / "bad.json").write_text("{not-json")
    ids = [r["id"] for r in client.get("/api/hitl/pending").json()["requests"]]
    assert ids == ["good"]


# ── POST /api/hitl/<request_id> ────────────────────────────────────────────


def test_post_decision_writes_decision_file(harness_root: Path, client: TestClient) -> None:
    _write_request(harness_root, "abc", {
        "id": "abc", "state": "x", "title": "t", "description": "d",
        "is_always_on": False, "condition": None, "created_ts": 1.0,
    })
    r = client.post("/api/hitl/abc", json={"decision": "approve"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["decision"] == "approve"

    dec_path = harness_root / "runs" / "r1" / "workspace" / "hitl" / "decisions" / "abc.json"
    assert dec_path.exists()
    assert json.loads(dec_path.read_text())["decision"] == "approve"


def test_post_decision_rejects_unknown_verb(harness_root: Path, client: TestClient) -> None:
    _write_request(harness_root, "abc", {
        "id": "abc", "state": "x", "title": "t", "description": "d",
        "is_always_on": False, "condition": None, "created_ts": 1.0,
    })
    r = client.post("/api/hitl/abc", json={"decision": "yes-please"})
    assert r.status_code == 400
    assert "decision" in r.json()["error"]


def test_post_decision_404_when_no_matching_request(client: TestClient) -> None:
    r = client.post("/api/hitl/nope", json={"decision": "approve"})
    assert r.status_code == 404


def test_post_decision_rejects_path_traversal(client: TestClient) -> None:
    for bad_id in ("../etc/passwd", "..", "a/b", "a\\b"):
        r = client.post(f"/api/hitl/{bad_id}", json={"decision": "approve"})
        # Either the router 404s bad shapes or our validator 400s the id
        assert r.status_code in (400, 404, 405), (bad_id, r.status_code, r.text)


def test_post_decision_all_three_verbs(harness_root: Path, client: TestClient) -> None:
    for verb in ("approve", "deny", "skip"):
        rid = f"req-{verb}"
        _write_request(harness_root, rid, {
            "id": rid, "state": "x", "title": "t", "description": "d",
            "is_always_on": False, "condition": None, "created_ts": 1.0,
        })
        r = client.post(f"/api/hitl/{rid}", json={"decision": verb})
        assert r.status_code == 200
        dec = json.loads((harness_root / "runs" / "r1" / "workspace" / "hitl"
                          / "decisions" / f"{rid}.json").read_text())
        assert dec["decision"] == verb


def test_post_decision_rejects_invalid_json_body(harness_root: Path, client: TestClient) -> None:
    _write_request(harness_root, "abc", {
        "id": "abc", "state": "x", "title": "t", "description": "d",
        "is_always_on": False, "condition": None, "created_ts": 1.0,
    })
    r = client.post("/api/hitl/abc", content=b"not json",
                    headers={"content-type": "application/json"})
    assert r.status_code == 400
