"""Provider table + `<provider>/<model_id>` spec resolution."""

from __future__ import annotations

import pytest

from harness.providers import (
    Provider,
    ProviderError,
    load_providers,
    resolve_model_spec,
)


def test_load_builtin_has_key_providers() -> None:
    providers = load_providers()
    assert "anthropic" in providers
    assert "openai" in providers
    assert "openrouter" in providers
    assert "deepseek" in providers
    assert "vllm" in providers

    ant = providers["anthropic"]
    assert ant.wire == "anthropic"
    assert ant.api_key_env == "ANTHROPIC_API_KEY"

    orouter = providers["openrouter"]
    assert orouter.wire == "openai"


def test_resolve_simple_spec() -> None:
    r = resolve_model_spec(
        "anthropic/claude-opus-4-8",
        env={"ANTHROPIC_API_KEY": "sk-test"},
    )
    assert r.provider.name == "anthropic"
    assert r.model_id == "claude-opus-4-8"
    assert r.api_key == "sk-test"


def test_resolve_openrouter_nested_model_id() -> None:
    # OpenRouter models look like `anthropic/claude-opus-4` — the split is
    # only at the FIRST '/', so the nested `/` stays inside model_id.
    r = resolve_model_spec(
        "openrouter/anthropic/claude-opus-4",
        env={"OPENROUTER_API_KEY": "sk-test"},
    )
    assert r.provider.name == "openrouter"
    assert r.model_id == "anthropic/claude-opus-4"


def test_resolve_vllm_no_api_key() -> None:
    r = resolve_model_spec("vllm/Qwen/Qwen2.5-72B-Instruct", env={})
    assert r.provider.name == "vllm"
    assert r.model_id == "Qwen/Qwen2.5-72B-Instruct"
    assert r.api_key is None


def test_missing_slash_errors() -> None:
    with pytest.raises(ProviderError, match="must be"):
        resolve_model_spec("just-a-name", env={})


def test_empty_model_id_errors() -> None:
    with pytest.raises(ProviderError, match="empty model_id"):
        resolve_model_spec("anthropic/", env={})


def test_unknown_provider_errors() -> None:
    with pytest.raises(ProviderError, match="unknown provider"):
        resolve_model_spec("nope/foo", env={})


def test_missing_env_key_errors() -> None:
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        resolve_model_spec("anthropic/x", env={})


def test_user_override(tmp_path) -> None:
    user_yaml = tmp_path / "providers.yaml"
    user_yaml.write_text(
        "version: 1\n"
        "providers:\n"
        "  my-proxy:\n"
        "    base_url: https://proxy.internal/v1\n"
        "    api_key_env: MY_PROXY_KEY\n"
        "    wire: openai\n"
    )
    providers = load_providers(user_config=user_yaml)
    assert "my-proxy" in providers
    assert providers["my-proxy"].base_url == "https://proxy.internal/v1"
    assert "anthropic" in providers  # built-in preserved


def test_wire_validation(tmp_path) -> None:
    user_yaml = tmp_path / "providers.yaml"
    user_yaml.write_text(
        "version: 1\nproviders:\n  bad:\n    base_url: x\n    wire: cohere\n"
    )
    with pytest.raises(ProviderError, match="invalid wire"):
        load_providers(user_config=user_yaml)


def test_provider_immutable() -> None:
    p = Provider(name="x", base_url="u", api_key_env=None, wire="openai")
    with pytest.raises(Exception):
        p.name = "y"  # type: ignore[misc]
