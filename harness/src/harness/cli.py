"""CLI: `verl-harness-runtime run <workdir> --model <provider/model> --goal <...>`.

Wires the pieces from providers → backend → state driver → stdout event stream.
Circular-import care: state_driver imports RunConfig from here, so backend
resolution happens in _handle_run without touching state_driver at module load.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import os
import re
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    """The resolved run-invocation config that later stages consume."""

    workdir: Path
    model: str
    goal: str
    state: str
    run_id: str
    hitl: bool
    verl_root: Path | None
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        d["workdir"] = str(self.workdir)
        d["verl_root"] = str(self.verl_root) if self.verl_root else None
        return json.dumps(d, indent=2, sort_keys=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verl-harness-runtime",
        description="Drive the verl-harness FSM markdown specs against any LLM backend.",
    )
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    run = sub.add_parser(
        "run",
        help="Drive one FSM state end-to-end (M1: single-state only).",
        description="Drive one FSM state end-to-end.",
    )
    run.add_argument(
        "workdir",
        type=Path,
        help="Path to a verl-harness spec directory (contains states/, skills/, CLAUDE.md).",
    )
    run.add_argument(
        "--model",
        default=None,
        metavar="PROVIDER/MODEL_ID",
        help="e.g. anthropic/claude-opus-4-8, openai/gpt-5, openrouter/anthropic/claude-opus-4. "
             "Optional when --resume-run inherits from meta.json.",
    )
    run.add_argument(
        "--goal",
        default=None,
        help="One-sentence training intent (e.g. 'train GRPO on gsm8k with Qwen2.5-3B'). "
             "Optional when --resume-run inherits from meta.json.",
    )
    run.add_argument(
        "--state",
        default="intake",
        help="State to drive (M1: single state only). Default: intake.",
    )
    run.add_argument(
        "--run-id",
        default=None,
        help="Workspace run id under runs/. Default: auto-generated from goal + timestamp.",
    )
    run.add_argument(
        "--verl-root",
        type=Path,
        default=None,
        help="Absolute path to the verl checkout. Falls back to $VERL_HOME.",
    )
    run.add_argument(
        "--no-hitl",
        dest="hitl",
        action="store_false",
        default=True,
        help="Semi-autonomous mode — skip ordinary hand-off points (always-on gates still fire).",
    )
    run.add_argument(
        "--hitl-channel",
        choices=("stdio", "event"),
        default="stdio",
        help="Where to prompt for hand-off approvals. `stdio` (default) reads "
             "y/n/s from stdin. `event` writes request files under "
             "workspace/hitl/requests/ and polls decisions/ — used by the "
             "dashboard's Approvals tab.",
    )
    run.add_argument(
        "--provider-config",
        type=Path,
        default=None,
        help="Extra provider yaml, merged on top of built-ins. "
             "Default location: ~/.verl-harness/providers.yaml.",
    )
    run.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Cap on tool-loop iterations. Default 100.",
    )
    run.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Cap on assistant output tokens per turn. Default 4096.",
    )
    run.add_argument(
        "--max-states",
        type=int,
        default=50,
        help="Cap on total state entries in orchestrator mode. Default 50.",
    )
    run.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="If set, mirror the JSON event stream to this file (line-delimited).",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse args, resolve provider, print config, exit — do not call the backend.",
    )
    run.add_argument(
        "--resume-run",
        default=None,
        metavar="RUN_ID",
        help="Resume an existing run: read runs/<id>/meta.json + state_log's "
             "last `entered` state, then continue the orchestrator from there. "
             "--goal / --model may be omitted (inherited from meta.json).",
    )
    run.add_argument(
        "--orchestrate",
        dest="orchestrate",
        action="store_true",
        default=True,
        help="Drive the whole FSM until terminal (M2 default).",
    )
    run.add_argument(
        "--single-state",
        dest="orchestrate",
        action="store_false",
        help="Drive one state only (M1 behavior).",
    )

    sub.add_parser("version", help="Print runtime version and exit.")

    return p


def _slug(text: str, limit: int = 24) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit] or "run"


def _resolve_run_id(explicit: str | None, goal: str) -> str:
    if explicit:
        return explicit
    ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{_slug(goal)}-{ts}-{uuid.uuid4().hex[:6]}"


def _resolve_verl_root(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.resolve()
    env = os.environ.get("VERL_HOME")
    return Path(env).resolve() if env else None


class _TeeSink:
    """Write JSON lines to both stdout and (optionally) a log file."""

    def __init__(self, primary, mirror=None):
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        n = self._primary.write(data)
        if self._mirror is not None:
            self._mirror.write(data)
        return n

    def flush(self) -> None:
        try:
            self._primary.flush()
        except Exception:  # pragma: no cover
            pass
        if self._mirror is not None:
            try:
                self._mirror.flush()
            except Exception:  # pragma: no cover
                pass


def _handle_run(args: argparse.Namespace) -> int:
    from harness.backends.anthropic import AnthropicBackend
    from harness.backends.base import Backend
    from harness.orchestrator import orchestrate
    from harness.providers import ProviderError, load_providers, resolve_model_spec
    from harness.resume import ResumeError, flip_status_to_running, load_resume_plan
    from harness.state_driver import drive_state

    workdir = args.workdir.resolve()
    if not workdir.exists():
        print(f"error: workdir does not exist: {workdir}", file=sys.stderr)
        return 2
    if not (workdir / "states").is_dir():
        print(
            f"error: workdir does not look like a verl-harness spec dir "
            f"(missing states/): {workdir}",
            file=sys.stderr,
        )
        return 2

    # ── Resume mode overrides most fields from the existing run's meta.json ──
    resume_state: str | None = None
    resume_run_id: str | None = None
    if args.resume_run:
        workspace_root = workdir / "runs"
        try:
            plan = load_resume_plan(workspace_root, args.resume_run)
        except ResumeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        # Model + goal inherited from meta.json unless CLI overrides
        args.model = args.model or plan.meta.model
        args.goal = args.goal or plan.meta.goal
        resume_state = plan.last_state
        resume_run_id = args.resume_run

    if not args.model or "/" not in args.model:
        print(
            f"error: --model must be '<provider>/<model_id>', got: {args.model!r}",
            file=sys.stderr,
        )
        return 2

    try:
        providers = load_providers(user_config=args.provider_config)
        resolved = resolve_model_spec(args.model, providers=providers)
    except ProviderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    cfg = RunConfig(
        workdir=workdir,
        model=args.model,
        goal=args.goal or "(resumed)",
        state=resume_state or args.state,
        run_id=resume_run_id or _resolve_run_id(args.run_id, args.goal or "resumed"),
        hitl=args.hitl,
        verl_root=_resolve_verl_root(args.verl_root),
    )

    if args.dry_run:
        print(cfg.to_json())
        return 0

    # Build the backend from the resolved model
    backend: Backend
    if resolved.provider.wire == "anthropic":
        backend = AnthropicBackend.from_resolved(resolved)
    elif resolved.provider.wire == "openai":
        from harness.backends.openai import OpenAIBackend
        backend = OpenAIBackend.from_resolved(resolved)
    else:  # pragma: no cover — providers.py validates this
        print(f"error: unknown wire {resolved.provider.wire!r}", file=sys.stderr)
        return 2

    workspace_root = workdir / "runs"

    log_fp = None
    sink = sys.stdout
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fp = args.log_file.open("w", encoding="utf-8")
        sink = _TeeSink(sys.stdout, log_fp)  # type: ignore[assignment]

    session_id = f"vhr-{uuid.uuid4().hex[:12]}"

    # Resolve the hand-off prompter. `stdio` → let downstream default to
    # StdioPrompter. `event` → construct an EventPrompter pointed at this
    # run's workspace; state_driver injects the emitter after it's built.
    hitl_prompter = None
    if args.hitl_channel == "event":
        from harness.hitl import EventPrompter
        run_workspace = workspace_root / cfg.run_id / "workspace"
        hitl_prompter = EventPrompter(workspace=run_workspace)

    # Flip resumed run's meta.status back to running before orchestrator starts
    if args.resume_run:
        flip_status_to_running(
            workspace_root,
            args.resume_run,
            note=f"re-entering at {resume_state} via --resume-run",
        )

    try:
        if args.orchestrate:
            from harness.usage import UsageTotal, format_summary
            orch_result = asyncio.run(
                orchestrate(
                    config=cfg,
                    backend=backend,
                    workspace_root=workspace_root,
                    session_id=session_id,
                    sink=sink,
                    max_iterations_per_state=args.max_iterations,
                    max_tokens=args.max_tokens,
                    max_states=args.max_states,
                    resumed=bool(args.resume_run),
                    hitl_prompter=hitl_prompter,
                )
            )
            print(
                f"\n[orchestrator] {orch_result.reason}: {orch_result.message} "
                f"(visited={'.'.join(orch_result.visited) or 'none'}, "
                f"turns={orch_result.total_turns}, run_id={cfg.run_id})",
                file=sys.stderr,
            )
            # Cost / usage summary
            u = UsageTotal()
            u.add(orch_result.total_usage)
            u.turns = orch_result.total_turns
            print(format_summary(u, cfg.model), file=sys.stderr)
            return 0 if not orch_result.is_error else 1
        else:
            result = asyncio.run(
                drive_state(
                    config=cfg,
                    backend=backend,
                    workspace_root=workspace_root,
                    session_id=session_id,
                    sink=sink,
                    max_iterations=args.max_iterations,
                    max_tokens=args.max_tokens,
                    hitl_prompter=hitl_prompter,
                )
            )
            print(
                f"\n[{result.state_name}] {result.reason}: {result.message} "
                f"(turns={result.turns}, run_id={cfg.run_id})",
                file=sys.stderr,
            )
            return 0 if not result.is_error else 1
    finally:
        if log_fp is not None:
            log_fp.close()


def _handle_version(_args: argparse.Namespace) -> int:
    from harness import __version__

    print(__version__)
    return 0


def main(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _handle_run(args)
    if args.cmd == "version":
        return _handle_version(args)

    parser.print_help(sys.stderr)
    return 2
