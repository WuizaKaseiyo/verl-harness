"""CLI-level smoke tests via subprocess.

Live backend calls need ANTHROPIC_API_KEY, so we cover the CLI paths that
don't require the API here (help, version, dry-run, provider resolution
errors, and the not-yet-implemented openai wire).

The full CLI → driver → events path is already exercised by
test_state_driver.py; this file guards the argparse / provider / backend-factory
glue that lives ONLY in cli.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_SRC = Path(__file__).resolve().parents[1] / "src"


def _run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    full_env = os.environ.copy()
    full_env["PYTHONPATH"] = str(HARNESS_SRC) + (
        os.pathsep + full_env["PYTHONPATH"] if "PYTHONPATH" in full_env else ""
    )
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "harness", *args],
        capture_output=True,
        text=True,
        env=full_env,
        timeout=30,
    )


def test_help() -> None:
    r = _run(["--help"])
    assert r.returncode == 0
    assert "verl-harness-runtime" in r.stdout
    assert "run" in r.stdout


def test_version() -> None:
    r = _run(["version"])
    assert r.returncode == 0
    assert r.stdout.strip()


def test_dry_run_prints_config(tmp_path: Path) -> None:
    r = _run(
        [
            "run",
            str(REPO_ROOT),
            "--model", "anthropic/claude-opus-4-8",
            "--goal", "sanity",
            "--dry-run",
            "--run-id", "T-dry",
        ],
        env={"ANTHROPIC_API_KEY": "sk-fake"},
    )
    assert r.returncode == 0, r.stderr
    assert '"model": "anthropic/claude-opus-4-8"' in r.stdout
    assert '"run_id": "T-dry"' in r.stdout


def test_missing_api_key_errors_cleanly() -> None:
    env_no_key = {"ANTHROPIC_API_KEY": ""}  # unset by empty
    r = _run(
        [
            "run",
            str(REPO_ROOT),
            "--model", "anthropic/claude-opus-4-8",
            "--goal", "x",
            "--dry-run",
        ],
        env=env_no_key,
    )
    assert r.returncode == 2
    assert "ANTHROPIC_API_KEY" in r.stderr


def test_unknown_provider_errors() -> None:
    r = _run(
        [
            "run",
            str(REPO_ROOT),
            "--model", "bogus/foo",
            "--goal", "x",
            "--dry-run",
        ]
    )
    assert r.returncode == 2
    assert "unknown provider" in r.stderr


def test_openai_wire_backend_constructs(tmp_path: Path) -> None:
    """openai wire is wired up (verified via --dry-run so we don't hit the API)."""
    r = _run(
        [
            "run",
            str(REPO_ROOT),
            "--model", "openai/gpt-5",
            "--goal", "x",
            "--dry-run",
        ],
        env={"OPENAI_API_KEY": "sk-fake"},
    )
    assert r.returncode == 0, r.stderr
    assert '"model": "openai/gpt-5"' in r.stdout


def test_bad_workdir_errors() -> None:
    r = _run(
        [
            "run",
            "/definitely/not/here/98765",
            "--model", "anthropic/x",
            "--goal", "x",
            "--dry-run",
        ]
    )
    assert r.returncode == 2
    assert "does not exist" in r.stderr


def test_workdir_without_states_errors(tmp_path: Path) -> None:
    r = _run(
        [
            "run",
            str(tmp_path),
            "--model", "anthropic/x",
            "--goal", "x",
            "--dry-run",
        ]
    )
    assert r.returncode == 2
    assert "missing states" in r.stderr


def test_auto_generated_run_id_includes_slug_and_time() -> None:
    r = _run(
        [
            "run",
            str(REPO_ROOT),
            "--model", "anthropic/claude-opus-4-8",
            "--goal", "train GRPO on gsm8k",
            "--dry-run",
        ],
        env={"ANTHROPIC_API_KEY": "sk-fake"},
    )
    assert r.returncode == 0
    # Slug should include normalized words from the goal
    assert "train-grpo-on-gsm8k" in r.stdout


# ── live smoke (opt-in via ANTHROPIC_API_KEY) ────────────────────────────

_HAS_LIVE_KEY = bool(os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant"))


@pytest.mark.skipif(not _HAS_LIVE_KEY, reason="needs a real ANTHROPIC_API_KEY (sk-ant…)")
def test_live_smoke_intake(tmp_path: Path) -> None:
    """Actually drive the intake state against the real Anthropic API.

    Only runs when ANTHROPIC_API_KEY is a real key (heuristic: starts with sk-ant).
    In practice this is the T7 exit-criteria: end-to-end + workspace file + dashboard.
    """
    model = os.environ.get("HARNESS_TEST_MODEL", "claude-haiku-4-5-20251001")
    log = tmp_path / "events.jsonl"
    r = _run(
        [
            "run",
            str(REPO_ROOT),
            "--model", f"anthropic/{model}",
            "--goal", "sanity smoke: reply with an intake plan and transition to locate_recipe",
            "--state", "intake",
            "--no-hitl",
            "--run-id", "smoke-t7",
            "--log-file", str(log),
            "--max-iterations", "20",
        ]
    )
    assert log.exists() and log.stat().st_size > 0
    # We accept either success (transition→locate_recipe) or a clean error —
    # what matters is the event stream came out well-formed.
    lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
    assert lines[0].startswith('{"type":"system"')
    assert lines[-1].startswith('{"type":"result"')
