"""Human-in-the-loop layer.

Parses each state file's `## Hand-off Points` block (also accepts the older
`## Human Checkpoints` heading for back-compat), classifies each point as
either ordinary or always-on, and prompts the user via stdio when appropriate.

The 4 always-on gates called out in `skills/global/scientific_principles.md`:
  - generate_preprocess: script approval
  - configure_reward:    custom/shaped reward approval
  - sanity_rollout:      sanity report approval
  - launch_training:     cost gate (threshold-based)

Classification is done by keyword match against the hand-off description text
(`**Always-on**` or `**Threshold-based**` markers), so if a state file adds
another always-on gate later, we detect it automatically.

Two Prompter channels ship in-tree:
  - StdioPrompter — reads y/n/s from stdin (default; CLI/terminal use)
  - EventPrompter — writes `workspace/hitl/requests/<uuid>.json` and blocks
    polling `workspace/hitl/decisions/<uuid>.json` (dashboard-driven; the
    web server writes the decision file in response to the user clicking
    Approve / Deny / Skip)
"""

from __future__ import annotations

import json
import re
import sys
import time
import uuid as _uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Protocol

from harness.context import _extract_section  # type: ignore[attr-defined]


# ── data types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HandOffPoint:
    state: str
    title: str
    description: str
    is_always_on: bool
    # `**Always-on for KEY ∈ {v1, v2}**` embeds a workspace-checkable condition.
    # None when the hand-off is unconditional or the condition can't be parsed.
    condition: str | None = None


class PromptDecision:
    """Truthy tri-state — approve, deny, or skip."""

    APPROVE = "approve"
    DENY = "deny"
    SKIP = "skip"


# ── parsing ────────────────────────────────────────────────────────────────

_HAND_OFF_HEADINGS = ("Hand-off Points", "Human Checkpoints")
_BULLET_TITLE = re.compile(
    r"^-\s+\*\*(?P<title>[^*]+?)\*\*\s*(?:\.\s*)?(?P<rest>.*?)$",
    re.MULTILINE,
)
# Always-on classification — mirrors CLAUDE.md's "four always-on" list.
#
# Three marker shapes count as always-on:
#   `**Always-on**:` / `.`         — unconditional
#   `**Always-on for X**`           — X is a workspace-checkable predicate;
#                                     runtime evaluates X and skips when it doesn't hold
#   `**Threshold-based**`           — cost gate; agent-side threshold logic; runtime always fires
#
# One shape is treated as ordinary (skippable):
#   `**Always-on** when Y ...`      — Y is a runtime state we don't evaluate today
_UNCONDITIONAL_ALWAYS_ON = re.compile(r"\*\*Always-on\*\*\s*[:.]")
_FOR_CLAUSE_ALWAYS_ON = re.compile(r"\*\*Always-on for (?P<cond>[^*]+?)\*\*")
_THRESHOLD_BASED = re.compile(r"\*\*Threshold-based\*\*")
# M3-T7: `**Always-on** when the matching skill emits a halt condition (dpo/rm ...)`
# captures the parenthetical enumeration for workspace-conditional evaluation.
_WHEN_PAREN_ALWAYS_ON = re.compile(
    r"\*\*Always-on\*\*\s+when\b[^(]*\(([^)]*?)\)",
    re.IGNORECASE,
)


def _is_always_on(description: str) -> bool:
    return bool(
        _UNCONDITIONAL_ALWAYS_ON.search(description)
        or _FOR_CLAUSE_ALWAYS_ON.search(description)
        or _THRESHOLD_BASED.search(description)
        or _WHEN_PAREN_ALWAYS_ON.search(description)
    )


def _extract_condition(description: str) -> str | None:
    """Return the raw text of the always-on condition (for-clause or when-parens), or None."""
    m = _FOR_CLAUSE_ALWAYS_ON.search(description)
    if m is not None:
        return m.group("cond").strip()
    m2 = _WHEN_PAREN_ALWAYS_ON.search(description)
    if m2 is not None:
        return m2.group(1).strip()
    return None


# ── condition evaluation ───────────────────────────────────────────────────

# `KEY ∈ {v1, v2, ...}` — the shape used across the spec (unicode ∈ or ASCII `in`).
_SET_COND = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_.]*)\s*(?:∈|in)\s*\{(?P<vals>[^}]+)\}"
)
# `KEY == v` or `KEY = v` or `KEY: v`
_EQ_COND = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_.]*)\s*(?:==|=|:)\s*(?P<val>[A-Za-z0-9_./-]+)"
)

# Where the runtime looks for the KEY's value.
_INTENT_FILE = Path("intake") / "training_intent.md"
_INTENT_KV = re.compile(r"^-?\s*(?P<key>[A-Za-z_][A-Za-z0-9_.]*)\s*:\s*(?P<val>.+?)\s*$")


def _read_intent_value(workspace: Path, key: str) -> str | None:
    """Grep `key: value` out of workspace/intake/training_intent.md."""
    p = workspace / _INTENT_FILE
    if not p.is_file():
        return None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _INTENT_KV.match(line)
        if m and m.group("key") == key:
            # Strip inline comments (`value  # note ...`)
            return m.group("val").split("#", 1)[0].strip()
    return None


# Common intake-field keywords that hint at which key to check when the
# condition text doesn't spell out the key explicitly. Used by the M3-T7
# `**Always-on** when ... (X/Y/Z...)` parser — the parenthetical often lists
# algorithm names without saying "algorithm: X".
_ALGO_HINTS = {
    "dpo", "rm", "gdpo", "ppo", "grpo", "sft", "distill", "rloo", "remax", "dapo",
    "gspo", "cispo", "gmpo", "sapo", "dppo", "gpg", "mtp", "otb",
    "on_policy_distillation", "reinforce_plus_plus",
}
_REWARD_HINTS = {"rule", "model", "custom", "shaped"}


def _guess_key_for_values(values: set[str]) -> str | None:
    """When the condition names values without a key, guess which intake field
    they belong to."""
    if any(v in _ALGO_HINTS for v in values):
        return "algorithm"
    if any(v in _REWARD_HINTS for v in values):
        return "reward_kind"
    return None


def _parse_slash_list(text: str) -> set[str]:
    """Return the set of tokens in a `A/B/C ...` slash-separated list.

    Stops at the first non-identifier gap (space, comma) — so
    `dpo/rm without first-class trainer` yields {`dpo`, `rm`}.
    """
    m = re.match(r"([A-Za-z_][A-Za-z0-9_/]+)", text.strip())
    if m is None:
        return set()
    return {tok for tok in m.group(1).split("/") if tok}


def condition_met(condition: str, workspace: Path) -> bool | None:
    """Evaluate a hand-off's always-on condition against the workspace.

    Returns:
      - True    if condition holds (fire the gate)
      - False   if condition explicitly does not hold (skip the gate)
      - None    if the runtime can't evaluate the condition (still prompt to be safe)

    Handles four forms:
      - `KEY ∈ {v1, v2}`  (from `**Always-on for KEY ∈ {v1, v2}**`)
      - `KEY == v`
      - `A/B/C...`  (from `**Always-on** when ... (A/B/C...)` — key is inferred)
    """
    # 1) Set-membership form
    m = _SET_COND.search(condition)
    if m is not None:
        key = m.group("key")
        raw_vals = m.group("vals")
        values = {v.strip().strip("`'\"") for v in raw_vals.split(",") if v.strip()}
        actual = _read_intent_value(workspace, key)
        if actual is None:
            return None
        return actual.strip("`'\"") in values

    # 2) Equality form
    m = _EQ_COND.search(condition)
    if m is not None:
        key = m.group("key")
        expected = m.group("val")
        actual = _read_intent_value(workspace, key)
        if actual is None:
            return None
        return actual == expected

    # 3) Slash-list form (from the `**Always-on** when ... (A/B/C...)` marker)
    values = _parse_slash_list(condition)
    if values:
        key = _guess_key_for_values(values)
        if key is None:
            return None
        actual = _read_intent_value(workspace, key)
        if actual is None:
            return None
        return actual.strip("`'\"") in values

    return None


def parse_handoff_points(state_md: str, state_name: str) -> list[HandOffPoint]:
    """Return each `- **Title.** rest` bullet under `## Hand-off Points`."""
    section: str | None = None
    for heading in _HAND_OFF_HEADINGS:
        section = _extract_section(state_md, heading)
        if section is not None:
            break
    if section is None:
        return []

    # Split on the bullet-title regex to preserve multi-line rest bodies.
    matches = list(_BULLET_TITLE.finditer(section))
    if not matches:
        return []

    points: list[HandOffPoint] = []
    for i, m in enumerate(matches):
        title = m.group("title").strip(" .")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section)
        rest = (m.group("rest") + section[start:end]).strip()
        points.append(
            HandOffPoint(
                state=state_name,
                title=title,
                description=rest,
                is_always_on=_is_always_on(rest),
                condition=_extract_condition(rest),
            )
        )
    return points


# ── prompter interface ────────────────────────────────────────────────────


class Prompter(Protocol):
    def ask(self, hop: HandOffPoint) -> str:
        """Return one of PromptDecision.{APPROVE,DENY,SKIP}."""


@dataclass
class StdioPrompter:
    """Prompts on stderr, reads y/n/s from stdin."""

    err: IO[str] = sys.stderr
    inp: IO[str] = sys.stdin

    def ask(self, hop: HandOffPoint) -> str:
        gate_label = "ALWAYS-ON GATE" if hop.is_always_on else "hand-off"
        self.err.write(
            f"\n[{gate_label}] {hop.state} — {hop.title}\n"
            f"    {hop.description}\n"
            f"    Approve? [y/n/s(kip)] "
        )
        self.err.flush()
        raw = self.inp.readline().strip().lower()
        if raw in ("y", "yes"):
            return PromptDecision.APPROVE
        if raw in ("s", "skip"):
            return PromptDecision.SKIP
        return PromptDecision.DENY


@dataclass
class EventPrompter:
    """File-based two-way channel for the dashboard approval UI.

    On each `ask()`:
      1. Writes `workspace/hitl/requests/<uuid>.json` with the hand-off details.
      2. Optionally emits a stream-json `approval_request` event via `emitter`
         (for observability — the dashboard itself reads the file, not the event).
      3. Blocks polling `workspace/hitl/decisions/<uuid>.json` every
         `poll_interval` seconds. `time.sleep` is signal-interruptible so
         SIGTERM / SIGINT propagate normally.
      4. On decision file appearing: reads it, deletes both files, emits
         `approval_decision` if `emitter` is set, returns the decision string.

    Malformed / missing / unknown decision → DENY (safest default).
    """

    workspace: Path
    poll_interval: float = 0.5
    emitter: Any | None = None  # EventEmitter; typed loosely to avoid import cycle

    def ask(self, hop: HandOffPoint) -> str:
        request_id = _uuid.uuid4().hex
        requests_dir = self.workspace / "hitl" / "requests"
        decisions_dir = self.workspace / "hitl" / "decisions"
        requests_dir.mkdir(parents=True, exist_ok=True)
        decisions_dir.mkdir(parents=True, exist_ok=True)

        req_path = requests_dir / f"{request_id}.json"
        dec_path = decisions_dir / f"{request_id}.json"

        request = {
            "id": request_id,
            "state": hop.state,
            "title": hop.title,
            "description": hop.description,
            "is_always_on": hop.is_always_on,
            "condition": hop.condition,
            "created_ts": time.time(),
        }
        req_path.write_text(
            json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if self.emitter is not None:
            try:
                self.emitter.approval_request(request)
            except Exception:
                pass  # emitter is best-effort; never let it break the gate

        while not dec_path.exists():
            time.sleep(self.poll_interval)

        decision = PromptDecision.DENY
        try:
            data = json.loads(dec_path.read_text(encoding="utf-8"))
            raw = data.get("decision")
            if raw in (
                PromptDecision.APPROVE,
                PromptDecision.DENY,
                PromptDecision.SKIP,
            ):
                decision = raw
        except (json.JSONDecodeError, OSError):
            pass  # keep DENY fallback

        for p in (req_path, dec_path):
            try:
                p.unlink()
            except OSError:
                pass

        if self.emitter is not None:
            try:
                self.emitter.approval_decision(request_id, decision)
            except Exception:
                pass

        return decision


@dataclass
class AutoApprovePrompter:
    """Head-less mode used by tests and `--no-hitl` where the runtime auto-approves
    the point. Never denies."""

    def ask(self, hop: HandOffPoint) -> str:
        return PromptDecision.APPROVE


@dataclass
class AutoDenyPrompter:
    """For tests — always says no."""

    def ask(self, hop: HandOffPoint) -> str:
        return PromptDecision.DENY


# ── driver-facing entry point ─────────────────────────────────────────────


@dataclass
class HandOffResult:
    approved: bool
    denied_titles: list[str]
    approved_titles: list[str]
    skipped_titles: list[str]


def evaluate_hand_offs(
    *,
    state_name: str,
    state_md: str,
    hitl: bool,
    prompter: Prompter | None = None,
    workspace: Path | None = None,
) -> HandOffResult:
    """Evaluate every hand-off point for a state.

    Ordering:
      - `hitl=True`  → every point prompts
      - `hitl=False` → only always-on points prompt; the rest auto-pass silently

    For `**Always-on for KEY ∈ {…}**` hand-offs, the runtime evaluates the
    condition against `workspace/intake/training_intent.md`. When the condition
    is explicitly false (e.g. reward_kind=rule while the gate is for
    custom/shaped), the gate is downgraded to ordinary and skipped under
    `--no-hitl`. When the condition can't be evaluated (workspace file not yet
    written, condition unparseable), we err on the side of prompting.

    Returns (all_approved, [denied], [approved], [skipped]). If ANY prompted
    point returns DENY, `approved` is False and the state should treat that as
    a user reject.
    """
    prompter = prompter or StdioPrompter()
    approved: list[str] = []
    denied: list[str] = []
    skipped: list[str] = []
    for hop in parse_handoff_points(state_md, state_name):
        # Runtime-condition downgrade for `**Always-on for X**` gates
        effective_always_on = hop.is_always_on
        if hop.is_always_on and hop.condition and workspace is not None:
            met = condition_met(hop.condition, workspace)
            if met is False:
                effective_always_on = False  # condition not met → treat as ordinary

        if not hitl and not effective_always_on:
            skipped.append(hop.title)
            continue
        decision = prompter.ask(hop)
        if decision == PromptDecision.APPROVE:
            approved.append(hop.title)
        elif decision == PromptDecision.SKIP:
            skipped.append(hop.title)
        else:
            denied.append(hop.title)
    return HandOffResult(
        approved=not denied,
        denied_titles=denied,
        approved_titles=approved,
        skipped_titles=skipped,
    )
