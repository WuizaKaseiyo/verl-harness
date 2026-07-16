"""Live smoke tests, one per provider.

Each test skips cleanly when its provider's env var is unset. When the env is
set, sends a trivial `say hi` prompt and asserts on the streamed events.

Model choice per provider is overrideable via `VHR_LIVE_MODEL_<PROVIDER>` env,
so users can pick cheaper / faster models. Defaults picked to be widely
available and inexpensive at time of writing.

Run with a single provider:
  ANTHROPIC_API_KEY=... python -m pytest tests/test_live_providers.py -k anthropic -v
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import pytest

from harness.backends.anthropic import AnthropicBackend
from harness.backends.base import RawEvent
from harness.backends.openai import OpenAIBackend
from harness.providers import load_providers


@dataclass(frozen=True)
class LiveCase:
    provider: str
    default_model: str
    wire: str  # "anthropic" or "openai"


# NOTE: model ids are the tail after `<provider>/` in --model.
CASES = [
    LiveCase("anthropic",  "claude-haiku-4-5-20251001",  "anthropic"),
    LiveCase("openai",     "gpt-4o-mini",                "openai"),
    LiveCase("openrouter", "openai/gpt-4o-mini",         "openai"),
    LiveCase("deepseek",   "deepseek-chat",              "openai"),
    LiveCase("qwen",       "qwen3-max",                  "openai"),
    LiveCase("groq",       "llama-3.3-70b-versatile",    "openai"),
    LiveCase("together",   "meta-llama/Llama-3.3-70B-Instruct-Turbo", "openai"),
]


def _resolve_model(case: LiveCase) -> str:
    env_key = f"VHR_LIVE_MODEL_{case.provider.upper()}"
    return os.environ.get(env_key, case.default_model)


def _make_backend(case: LiveCase, api_key: str, base_url: str):
    if case.wire == "anthropic":
        return AnthropicBackend(
            model_id=_resolve_model(case), api_key=api_key, base_url=base_url
        )
    return OpenAIBackend(
        model_id=_resolve_model(case), api_key=api_key, base_url=base_url
    )


async def _one_turn(backend) -> list[RawEvent]:
    events: list[RawEvent] = []
    async for e in backend.stream(
        system="You reply in exactly one word.",
        messages=[{"role": "user", "content": "Say hi."}],
        tools=[],
        max_tokens=32,
    ):
        events.append(e)
    return events


@pytest.mark.parametrize("case", CASES, ids=[c.provider for c in CASES])
def test_live_provider_hello(case: LiveCase) -> None:
    providers = load_providers()
    if case.provider not in providers:
        pytest.skip(f"provider {case.provider!r} not in providers table")
    profile = providers[case.provider]
    env_var = profile.api_key_env
    api_key = os.environ.get(env_var) if env_var else None
    if env_var and not api_key:
        pytest.skip(f"{env_var} not set — provider {case.provider!r} live-smoke skipped")

    backend = _make_backend(case, api_key or "sk-no-key-needed", profile.base_url)
    events = asyncio.run(_one_turn(backend))

    assert events, "backend yielded no events"
    assert events[-1].kind == "message_stop"
    stop_reason = events[-1].stop_reason
    assert stop_reason in {"end_turn", "max_tokens", "stop_sequence", "stop"}
    output_tokens = events[-1].usage.get("output_tokens", 0)
    assert output_tokens > 0, f"{case.provider}: no output tokens billed"

    text = "".join(e.text for e in events if e.kind == "text_delta")
    assert text.strip(), f"{case.provider}: empty text response"


def test_all_providers_have_default_model_declared() -> None:
    """Guard against regressions where a provider is added to providers.yaml
    but forgotten in the live-smoke matrix."""
    providers = load_providers()
    live_providers = {c.provider for c in CASES}
    # vllm intentionally excluded (needs a running local server)
    expected = set(providers) - {"vllm"}
    missing = expected - live_providers
    assert not missing, f"providers without live-smoke coverage: {missing}"
