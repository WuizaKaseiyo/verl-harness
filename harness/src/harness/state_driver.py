"""Single-state driver — the glue between context.py, loop.py, backends, and events.

Given a RunConfig + a Backend, drives ONE state end-to-end:

  1. Load state context (CLAUDE.md + state.md + skill dirs)
  2. Set up ToolContext + ToolRegistry + EventEmitter
  3. Emit system.init
  4. Inject `transition_to` control tool (+ `schedule_wakeup` for wait-capable states)
  5. Run tool-use loop with contract retries
  6. Validate the transition (or the end_turn for terminal states)
  7. Emit result event
  8. Return StateResult so the orchestrator can chain into the next state
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from harness.backends.base import Backend
from harness.cli import RunConfig
from harness.context import (
    ContextError,
    StateContext,
    Transition,
    build_initial_user_message,
    load_state_context,
)
from harness.contracts import contract_violation_message, missing_deliverables
from harness.events import EventEmitter, tool_result_block
from harness.hitl import HandOffResult, Prompter, evaluate_hand_offs
from harness.loop import LoopResult, run_loop
from harness.tools import ToolContext, ToolRegistry, default_registry


# ── loop-state persistence (for reflect back-edge iteration counting) ─────


def _loop_state_path(workspace: Path) -> Path:
    return workspace / "reflect" / "loop_state.json"


def _load_loop_state(workspace: Path) -> dict[str, int]:
    p = _loop_state_path(workspace)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    edges = data.get("edges") if isinstance(data, dict) else None
    return dict(edges) if isinstance(edges, dict) else {}


def _save_loop_state(workspace: Path, state: dict[str, int]) -> None:
    import os as _os
    p = _loop_state_path(workspace)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"edges": state}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _os.replace(tmp, p)


def _edge_key(source: str, target: str) -> str:
    return f"{source}->{target}"


def _loop_budget_error(source: str, target: str, used: int, cap: int) -> str:
    return (
        f"transition_to({target!r}) rejected — this edge already fired "
        f"{used}/{cap} times (declared **Loop:** max_iterations). "
        f"Pick a different next_state from the state's ## Next States block "
        f"(usually the non-loop exit branch)."
    )


# ── result type ───────────────────────────────────────────────────────────


@dataclass
class StateResult:
    state_name: str
    next_state: str | None
    is_error: bool
    reason: str  # "transition" | "terminal_end" | "no_transition" | "invalid_transition"
                 # | "max_iterations" | "max_tokens" | "context_error"
                 # | "wait" (schedule_wakeup fired — orchestrator sleeps + re-enters)
                 # | "contract_violation" | "hitl_denied" | "loop_budget_exhausted"
    message: str
    turns: int = 0
    final_text: str = ""
    wait_seconds: int | None = None  # only set when reason == "wait"
    wait_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)  # aggregated over this state's turns


# States where the agent may call `schedule_wakeup` to pause + re-enter fresh.
# Matches Claude Code's ScheduleWakeup + auto-resume pattern.
WAIT_CAPABLE_STATES = frozenset({
    "monitor_training",
    "launch_training",
    "sanity_rollout",
})


# ── control-tool schemas ──────────────────────────────────────────────────


def _build_control_schema(allowed: list[str]) -> dict[str, Any]:
    return {
        "name": "transition_to",
        "description": (
            "Call this when the current state's ## Next States deliverables "
            "are fully satisfied. Pick the `next_state` matching the branch's "
            "**Condition:** block. Do NOT call it before the deliverables are done."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "next_state": {
                    "type": "string",
                    "enum": allowed,
                    "description": "Name of the next FSM state.",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional one-line justification.",
                },
            },
            "required": ["next_state"],
        },
    }


def _build_wakeup_schema() -> dict[str, Any]:
    return {
        "name": "schedule_wakeup",
        "description": (
            "Pause this state for `delaySeconds`, then re-enter it fresh "
            "(workspace preserved). Use when polling a long-running external "
            "job — e.g. `squeue` for a slurm job that takes hours. Cap 3600s "
            "per call; call again after wakeup for longer waits. Min 30s."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "delaySeconds": {
                    "type": "integer",
                    "minimum": 30,
                    "maximum": 3600,
                    "description": "Seconds to sleep before re-entering this state.",
                },
                "reason": {
                    "type": "string",
                    "description": "One-line justification recorded in state_log.md.",
                },
            },
            "required": ["delaySeconds", "reason"],
        },
    }


def _find_transition(state_ctx: StateContext, target: str) -> Transition | None:
    for t in state_ctx.transitions:
        if t.target == target:
            return t
    return None


# ── driver ────────────────────────────────────────────────────────────────


async def drive_state(
    *,
    config: RunConfig,
    backend: Backend,
    workspace_root: Path,
    session_id: str,
    sink: IO[str] = sys.stdout,
    max_iterations: int = 100,
    max_tokens: int = 4096,
    max_contract_retries: int = 3,
    hitl_prompter: Prompter | None = None,
    state_md_override: str | None = None,
) -> StateResult:
    """Drive one state. Returns the transition target (or terminal signal).

    `max_contract_retries` — how many times the runtime will reject a
    `transition_to` whose declared workspace deliverables are missing and
    re-enter the loop so the model can write them first.
    """
    aggregated_usage: dict[str, int] = {}

    def _acc(u: dict[str, int]) -> None:
        for k, v in u.items():
            aggregated_usage[k] = aggregated_usage.get(k, 0) + int(v or 0)

    def _make_result(**kwargs: Any) -> StateResult:
        """Wrapper that stamps every StateResult with the accumulated usage."""
        return StateResult(usage=dict(aggregated_usage), **kwargs)

    try:
        state_ctx: StateContext = load_state_context(config.workdir, config.state)
    except ContextError as e:
        emitter = EventEmitter(
            session_id=session_id,
            model=config.model,
            cwd=str(config.workdir),
            sink=sink,
        )
        emitter.system_init(tools=[])
        emitter.result(is_error=True, subtype="context_error", text=str(e))
        return _make_result(
            state_name=config.state,
            next_state=None,
            is_error=True,
            reason="context_error",
            message=str(e),
        )

    workspace = workspace_root / config.run_id / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    tools = default_registry()
    ctx = ToolContext(
        cwd=config.workdir,
        workdir=config.workdir,
        run_id=config.run_id,
        workspace=workspace,
    )

    emitter = EventEmitter(
        session_id=session_id,
        model=config.model,
        cwd=str(config.workdir),
        sink=sink,
    )
    # Wire the emitter into any prompter that exposes an `emitter` slot
    # (EventPrompter) so approval requests + decisions land in events.jsonl.
    if hitl_prompter is not None and hasattr(hitl_prompter, "emitter"):
        hitl_prompter.emitter = emitter
    advertised = tools.names() + (["transition_to"] if not state_ctx.is_terminal else [])
    emitter.system_init(tools=advertised)

    control_schemas: list[dict[str, Any]] = []
    control_tools: set[str] = set()
    if not state_ctx.is_terminal:
        control_schemas = [_build_control_schema(state_ctx.next_states)]
        control_tools = {"transition_to"}
        if config.state in WAIT_CAPABLE_STATES:
            control_schemas.append(_build_wakeup_schema())
            control_tools.add("schedule_wakeup")

    initial_user = build_initial_user_message(
        goal=config.goal,
        run_id=config.run_id,
        workdir=config.workdir,
        verl_root=config.verl_root,
        workspace=workspace,
        state_name=config.state,
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": initial_user}]
    loop_result: LoopResult = LoopResult(reason="", turns=0, final_text="", messages=[])
    total_turns = 0

    # Prefer cacheable block form for anthropic wire; openai backend flattens.
    system_arg: Any = state_ctx.system_blocks or state_ctx.system_prompt

    for attempt in range(max_contract_retries + 1):
        loop_result = await run_loop(
            backend=backend,
            system=system_arg,
            initial_messages=messages,
            tools=tools,
            ctx=ctx,
            emitter=emitter,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            control_tools=control_tools,
            control_schemas=control_schemas,
        )
        total_turns += loop_result.turns
        _acc(loop_result.usage)

        # ── schedule_wakeup short-circuit ───────────────────────────────
        if (
            not state_ctx.is_terminal
            and loop_result.reason == "control"
            and loop_result.control_call is not None
            and loop_result.control_call.get("name") == "schedule_wakeup"
        ):
            seconds_raw = loop_result.control_call["input"].get("delaySeconds")
            reason_raw = loop_result.control_call["input"].get("reason", "")
            try:
                seconds = int(seconds_raw)
            except (TypeError, ValueError):
                seconds = 60
            seconds = max(30, min(3600, seconds))
            reason_txt = str(reason_raw)[:200] if reason_raw else "(no reason given)"
            emitter.result(
                is_error=False,
                subtype="wait",
                text=f"schedule_wakeup {seconds}s: {reason_txt}",
            )
            return _make_result(
                state_name=config.state,
                next_state=None,
                is_error=False,
                reason="wait",
                message=f"wakeup after {seconds}s: {reason_txt}",
                turns=total_turns,
                final_text=loop_result.final_text,
                wait_seconds=seconds,
                wait_reason=reason_txt,
            )

        # ── contracts + loop budget check for transition_to ──────────────
        if (
            not state_ctx.is_terminal
            and loop_result.reason == "control"
            and loop_result.control_call is not None
        ):
            target = loop_result.control_call["input"].get("next_state")
            transition = (
                _find_transition(state_ctx, target)
                if isinstance(target, str)
                else None
            )
            if transition is not None:
                missing = missing_deliverables(workspace, config.state, transition)
                loop_error: str | None = None
                if transition.loop_max_iterations is not None:
                    loop_state = _load_loop_state(workspace)
                    used = loop_state.get(_edge_key(config.state, target), 0)
                    if used >= transition.loop_max_iterations:
                        loop_error = _loop_budget_error(
                            config.state,
                            target,
                            used,
                            transition.loop_max_iterations,
                        )

                if missing or loop_error:
                    if attempt >= max_contract_retries:
                        subtype = (
                            "loop_budget_exhausted"
                            if loop_error
                            else "contract_violation"
                        )
                        text = (
                            loop_error
                            or f"missing after {max_contract_retries} retries: {missing}"
                        )
                        emitter.result(is_error=True, subtype=subtype, text=text)
                        return _make_result(
                            state_name=config.state,
                            next_state=None,
                            is_error=True,
                            reason=subtype,
                            message=(
                                loop_error
                                or f"transition_to({target}) blocked — missing "
                                   f"deliverables after {max_contract_retries} "
                                   f"retries: {', '.join(missing)}"
                            ),
                            turns=total_turns,
                            final_text=loop_result.final_text,
                        )
                    # Feed a rejection tool_result back and re-enter the loop
                    reason_text = loop_error or contract_violation_message(
                        config.state, target, missing
                    )
                    reject = tool_result_block(
                        tool_use_id=loop_result.control_call["id"],
                        content=reason_text,
                        is_error=True,
                    )
                    emitter.user_tool_result([reject])
                    messages = loop_result.messages + [
                        {"role": "user", "content": [reject]}
                    ]
                    continue

                # ── HITL: hand-off approval before accepting transition ─
                state_md = state_md_override
                if state_md is None:
                    state_md = (
                        config.workdir / "states" / f"{config.state}.md"
                    ).read_text()
                hop_result: HandOffResult = evaluate_hand_offs(
                    state_name=config.state,
                    state_md=state_md,
                    hitl=config.hitl,
                    prompter=hitl_prompter,
                    workspace=workspace,
                )
                if not hop_result.approved:
                    denied_str = (
                        ", ".join(hop_result.denied_titles) or "(unspecified)"
                    )
                    emitter.result(
                        is_error=True,
                        subtype="hitl_denied",
                        text=f"user denied hand-off: {denied_str}",
                    )
                    return _make_result(
                        state_name=config.state,
                        next_state=None,
                        is_error=True,
                        reason="hitl_denied",
                        message=f"hand-off denied: {denied_str}",
                        turns=total_turns,
                        final_text=loop_result.final_text,
                    )

                # ── accept: increment loop counter if applicable ───────
                if transition.loop_max_iterations is not None:
                    ls = _load_loop_state(workspace)
                    ls[_edge_key(config.state, target)] = (
                        ls.get(_edge_key(config.state, target), 0) + 1
                    )
                    _save_loop_state(workspace, ls)
        # Any other outcome falls through to the interpretation block below.
        break

    # ── interpret loop outcome ────────────────────────────────────────────
    if state_ctx.is_terminal:
        if loop_result.reason == "end_turn":
            emitter.result(is_error=False, text="terminal state completed")
            return _make_result(
                state_name=config.state,
                next_state=None,
                is_error=False,
                reason="terminal_end",
                message="terminal state completed",
                turns=total_turns,
                final_text=loop_result.final_text,
            )
        emitter.result(
            is_error=True,
            subtype="terminal_" + loop_result.reason,
            text=f"terminal state halted with {loop_result.reason}",
        )
        return _make_result(
            state_name=config.state,
            next_state=None,
            is_error=True,
            reason=loop_result.reason,
            message=f"terminal state halted with {loop_result.reason}",
            turns=total_turns,
        )

    if loop_result.reason == "control":
        assert loop_result.control_call is not None
        target = loop_result.control_call["input"].get("next_state")
        if target not in state_ctx.next_states:
            emitter.result(
                is_error=True,
                subtype="invalid_transition",
                text=f"transition_to({target!r}) not in {state_ctx.next_states}",
            )
            return _make_result(
                state_name=config.state,
                next_state=None,
                is_error=True,
                reason="invalid_transition",
                message=f"illegal target {target!r}; allowed: {state_ctx.next_states}",
                turns=total_turns,
                final_text=loop_result.final_text,
            )
        emitter.result(is_error=False, text=f"transition→{target}")
        return _make_result(
            state_name=config.state,
            next_state=target,
            is_error=False,
            reason="transition",
            message=f"ready for {target}",
            turns=total_turns,
            final_text=loop_result.final_text,
        )

    # Non-terminal state ran out of loop without calling transition_to
    subtype_map = {
        "end_turn": "no_transition",
        "max_iterations": "max_iterations",
        "max_tokens": "max_tokens",
    }
    subtype = subtype_map.get(loop_result.reason, loop_result.reason)
    emitter.result(
        is_error=True,
        subtype=subtype,
        text=f"halted without transition ({loop_result.reason})",
    )
    return _make_result(
        state_name=config.state,
        next_state=None,
        is_error=True,
        reason=subtype,
        message=f"halted without transition ({loop_result.reason})",
        turns=total_turns,
        final_text=loop_result.final_text,
    )
