"""Rate-limit retry — verify SDK is configured with our max_retries + timeout.

Wire-level 429 simulation requires respx or similar HTTP-level mocking; that
lives in the E2E smoke where it's easy to observe. Here we cover the config
surface and the client instantiation shape.
"""

from __future__ import annotations

from harness.backends.anthropic import (
    DEFAULT_MAX_RETRIES as ANTHROPIC_MAX_RETRIES,
    DEFAULT_TIMEOUT_S as ANTHROPIC_TIMEOUT_S,
    AnthropicBackend,
)
from harness.backends.openai import (
    DEFAULT_MAX_RETRIES as OPENAI_MAX_RETRIES,
    DEFAULT_TIMEOUT_S as OPENAI_TIMEOUT_S,
    OpenAIBackend,
)


# ── defaults ──────────────────────────────────────────────────────────────

def test_anthropic_default_retries() -> None:
    """Runtime picks a max_retries that survives a rate-limit window."""
    assert ANTHROPIC_MAX_RETRIES >= 5
    assert ANTHROPIC_TIMEOUT_S >= 60


def test_openai_default_retries() -> None:
    assert OPENAI_MAX_RETRIES >= 5
    assert OPENAI_TIMEOUT_S >= 60


# ── clients configured with our settings ─────────────────────────────────

def test_anthropic_client_uses_max_retries() -> None:
    b = AnthropicBackend(model_id="test-model", api_key="sk-fake")
    # anthropic SDK exposes max_retries via the client's private attr;
    # both anthropic 0.34+ store it on the client itself.
    assert getattr(b._client, "max_retries", 0) >= 5


def test_openai_client_uses_max_retries() -> None:
    b = OpenAIBackend(
        model_id="test", api_key="sk-fake", base_url="https://example.com/v1"
    )
    assert getattr(b._client, "max_retries", 0) >= 5


def test_backends_accept_custom_retry_config() -> None:
    b1 = AnthropicBackend(model_id="m", api_key="k", max_retries=2, timeout_s=30.0)
    assert getattr(b1._client, "max_retries", 0) == 2

    b2 = OpenAIBackend(model_id="m", api_key="k", base_url="https://x/v1", max_retries=3, timeout_s=45.0)
    assert getattr(b2._client, "max_retries", 0) == 3
