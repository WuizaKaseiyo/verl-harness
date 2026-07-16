"""Token usage aggregation + cost estimation.

Per-turn usage arrives in RawEvent.usage; the state driver and orchestrator
accumulate into `UsageTotal` and, at end-of-run, format a cost summary using
prices from `pricing.yaml` (overrideable via `~/.verl-harness/pricing.yaml`).
"""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class UsageTotal:
    """Running total across a full run (or any sub-window)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    turns: int = 0

    def add(self, other: dict[str, int] | "UsageTotal") -> None:
        if isinstance(other, UsageTotal):
            self.input_tokens += other.input_tokens
            self.output_tokens += other.output_tokens
            self.cache_read_input_tokens += other.cache_read_input_tokens
            self.cache_creation_input_tokens += other.cache_creation_input_tokens
            self.turns += other.turns
        else:
            self.input_tokens += int(other.get("input_tokens", 0) or 0)
            self.output_tokens += int(other.get("output_tokens", 0) or 0)
            self.cache_read_input_tokens += int(
                other.get("cache_read_input_tokens", 0) or 0
            )
            self.cache_creation_input_tokens += int(
                other.get("cache_creation_input_tokens", 0) or 0
            )

    def bump_turn(self) -> None:
        self.turns += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "turns": self.turns,
        }


# ── pricing table ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelPrice:
    """USD per MILLION tokens (Anthropic-style four-tier)."""

    input: float
    output: float
    cache_read: float | None = None
    cache_creation: float | None = None


def _load_pricing_yaml(source_text: str) -> dict[str, ModelPrice]:
    data = yaml.safe_load(source_text) or {}
    raw = data.get("models") or {}
    out: dict[str, ModelPrice] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        out[name] = ModelPrice(
            input=float(entry.get("input", 0.0)),
            output=float(entry.get("output", 0.0)),
            cache_read=(
                float(entry["cache_read"]) if "cache_read" in entry else None
            ),
            cache_creation=(
                float(entry["cache_creation"]) if "cache_creation" in entry else None
            ),
        )
    return out


def load_pricing(user_config: Path | None = None) -> dict[str, ModelPrice]:
    """Load built-in pricing.yaml plus optional user overrides."""
    text = importlib.resources.files("harness").joinpath("pricing.yaml").read_text()
    merged = _load_pricing_yaml(text)
    default_user = Path.home() / ".verl-harness" / "pricing.yaml"
    for p in (default_user, user_config):
        if p is None or not p.exists():
            continue
        merged.update(_load_pricing_yaml(p.read_text()))
    return merged


def _resolve_model_key(model_spec: str) -> list[str]:
    """From `openrouter/anthropic/claude-opus-4-8` → tries
    ['openrouter/anthropic/claude-opus-4-8', 'anthropic/claude-opus-4-8',
     'claude-opus-4-8'] in that order.
    """
    parts = model_spec.split("/")
    return ["/".join(parts[i:]) for i in range(len(parts))]


def estimate_cost(
    usage: UsageTotal,
    model_spec: str,
    pricing: dict[str, ModelPrice] | None = None,
) -> tuple[float | None, ModelPrice | None]:
    """Return (dollars, matched_price) or (None, None) if the model is unknown."""
    table = pricing if pricing is not None else load_pricing()
    matched: ModelPrice | None = None
    for key in _resolve_model_key(model_spec):
        if key in table:
            matched = table[key]
            break
    if matched is None:
        return (None, None)

    # Regular billed tokens = input - (cache_read + cache_creation)
    normal_input = max(
        0,
        usage.input_tokens
        - usage.cache_read_input_tokens
        - usage.cache_creation_input_tokens,
    )
    dollars = (
        normal_input * matched.input
        + usage.output_tokens * matched.output
        + usage.cache_read_input_tokens * (matched.cache_read or matched.input)
        + usage.cache_creation_input_tokens
        * (matched.cache_creation or matched.input)
    ) / 1_000_000.0
    return (dollars, matched)


def format_summary(usage: UsageTotal, model_spec: str) -> str:
    """Human-readable one-line summary for CLI stderr."""
    dollars, matched = estimate_cost(usage, model_spec)
    parts = [
        f"in={usage.input_tokens}",
        f"out={usage.output_tokens}",
    ]
    if usage.cache_read_input_tokens:
        parts.append(f"cache_read={usage.cache_read_input_tokens}")
    if usage.cache_creation_input_tokens:
        parts.append(f"cache_write={usage.cache_creation_input_tokens}")
    parts.append(f"turns={usage.turns}")
    total = " · ".join(parts)
    if dollars is not None:
        return f"[usage] {total} · ≈ ${dollars:.4f} ({model_spec})"
    return f"[usage] {total} · cost: pricing table has no entry for {model_spec}"
