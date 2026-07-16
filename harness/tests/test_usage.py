"""Usage aggregation + cost estimation tests."""

from __future__ import annotations

from pathlib import Path

from harness.usage import (
    UsageTotal,
    estimate_cost,
    format_summary,
    load_pricing,
)


# ── UsageTotal ────────────────────────────────────────────────────────────

def test_usage_add_from_dict() -> None:
    u = UsageTotal()
    u.add({"input_tokens": 100, "output_tokens": 50})
    u.add({"input_tokens": 20})
    assert u.input_tokens == 120
    assert u.output_tokens == 50


def test_usage_add_cache_fields() -> None:
    u = UsageTotal()
    u.add({
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 60,
        "cache_creation_input_tokens": 10,
    })
    assert u.cache_read_input_tokens == 60
    assert u.cache_creation_input_tokens == 10


def test_usage_add_other_usagetotal() -> None:
    a = UsageTotal(input_tokens=100, output_tokens=50, turns=2)
    b = UsageTotal(input_tokens=200, output_tokens=100, turns=3)
    a.add(b)
    assert a.input_tokens == 300
    assert a.turns == 5


def test_usage_to_dict_roundtrip() -> None:
    u = UsageTotal(input_tokens=100, output_tokens=50, turns=1)
    d = u.to_dict()
    assert d["input_tokens"] == 100
    assert d["output_tokens"] == 50
    assert d["turns"] == 1


# ── pricing table ─────────────────────────────────────────────────────────

def test_load_pricing_has_known_models() -> None:
    pricing = load_pricing()
    assert "claude-haiku-4-5" in pricing
    assert "gpt-4o-mini" in pricing
    p = pricing["claude-haiku-4-5"]
    assert p.input > 0
    assert p.output > 0


def test_load_pricing_user_override(tmp_path: Path) -> None:
    user_yaml = tmp_path / "pricing.yaml"
    user_yaml.write_text(
        "version: 1\n"
        "models:\n"
        "  my-proxy-model:\n"
        "    input: 0.5\n"
        "    output: 1.5\n"
    )
    pricing = load_pricing(user_config=user_yaml)
    assert "my-proxy-model" in pricing
    assert pricing["my-proxy-model"].input == 0.5


# ── estimate_cost ─────────────────────────────────────────────────────────

def test_estimate_cost_known_model() -> None:
    u = UsageTotal(input_tokens=1_000_000, output_tokens=1_000_000)
    dollars, price = estimate_cost(u, "gpt-4o-mini")
    assert dollars is not None
    assert price is not None
    # gpt-4o-mini: $0.15/M in, $0.60/M out → $0.75 total
    assert 0.7 < dollars < 0.8


def test_estimate_cost_via_openrouter_prefix() -> None:
    """openrouter/openai/gpt-4o-mini should resolve to the gpt-4o-mini price."""
    u = UsageTotal(input_tokens=1_000_000, output_tokens=1_000_000)
    dollars, price = estimate_cost(u, "openrouter/openai/gpt-4o-mini")
    assert dollars is not None
    assert 0.7 < dollars < 0.8


def test_estimate_cost_unknown_model() -> None:
    u = UsageTotal(input_tokens=100_000)
    dollars, price = estimate_cost(u, "unknown-model-xyz")
    assert dollars is None
    assert price is None


def test_estimate_cost_with_cache_discount() -> None:
    """Cache-read tokens should be billed at cache_read rate, not input rate."""
    u = UsageTotal(
        input_tokens=1_000_000,
        output_tokens=0,
        cache_read_input_tokens=900_000,
        cache_creation_input_tokens=0,
    )
    # claude-haiku-4-5: $1.00/M input, $0.10/M cache_read
    # 100k normal * $1.00/M + 900k * $0.10/M = $0.10 + $0.09 = $0.19
    dollars, _ = estimate_cost(u, "claude-haiku-4-5")
    assert dollars is not None
    assert 0.18 < dollars < 0.20


# ── format_summary ────────────────────────────────────────────────────────

def test_format_summary_known_model() -> None:
    u = UsageTotal(input_tokens=1000, output_tokens=100, turns=3)
    s = format_summary(u, "gpt-4o-mini")
    assert "in=1000" in s
    assert "out=100" in s
    assert "turns=3" in s
    assert "$" in s


def test_format_summary_unknown_model() -> None:
    u = UsageTotal(input_tokens=1000, output_tokens=100, turns=3)
    s = format_summary(u, "nope-nope")
    assert "cost: pricing table has no entry" in s


def test_format_summary_shows_cache_fields_when_nonzero() -> None:
    u = UsageTotal(
        input_tokens=1000,
        output_tokens=100,
        cache_read_input_tokens=500,
        cache_creation_input_tokens=200,
        turns=2,
    )
    s = format_summary(u, "claude-haiku-4-5")
    assert "cache_read=500" in s
    assert "cache_write=200" in s
