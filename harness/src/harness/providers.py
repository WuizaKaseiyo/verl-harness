"""Provider profile loader — resolves `<provider>/<model_id>` model specs.

Built-in profiles live in `providers.yaml` next to this module. A user config
at `~/.verl-harness/providers.yaml` (or a path passed via `--provider-config`)
is deep-merged on top.

The `<provider>/<model_id>` spec splits at the FIRST `/`. That means
OpenRouter's `openrouter/anthropic/claude-opus-4` cleanly resolves to the
`openrouter` provider with model_id `anthropic/claude-opus-4`.
"""

from __future__ import annotations

import importlib.resources
import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Provider:
    """A resolved provider profile — where and how to talk to a backend."""

    name: str
    base_url: str
    api_key_env: str | None
    wire: str  # "anthropic" | "openai"


@dataclass(frozen=True)
class ResolvedModel:
    """A parsed `<provider>/<model_id>` spec, with API key looked up from env."""

    provider: Provider
    model_id: str
    api_key: str | None  # None only when provider.api_key_env is None


class ProviderError(Exception):
    """Raised for unknown provider, malformed spec, or missing API key."""


_VALID_WIRES = {"anthropic", "openai"}


def _load_builtin() -> dict[str, Provider]:
    """Load the packaged built-in providers yaml."""
    text = importlib.resources.files("harness").joinpath("providers.yaml").read_text()
    return _parse(text, source="<built-in>")


def _load_user(path: Path) -> dict[str, Provider]:
    if not path.exists():
        return {}
    return _parse(path.read_text(), source=str(path))


def _parse(text: str, *, source: str) -> dict[str, Provider]:
    data = yaml.safe_load(text) or {}
    raw_providers = data.get("providers") or {}
    out: dict[str, Provider] = {}
    for name, entry in raw_providers.items():
        if not isinstance(entry, dict):
            raise ProviderError(f"{source}: provider {name!r} must be a mapping")
        wire = entry.get("wire")
        if wire not in _VALID_WIRES:
            raise ProviderError(
                f"{source}: provider {name!r} has invalid wire "
                f"{wire!r} (expected one of {sorted(_VALID_WIRES)})"
            )
        base_url = entry.get("base_url")
        if not base_url:
            raise ProviderError(f"{source}: provider {name!r} missing base_url")
        out[name] = Provider(
            name=name,
            base_url=base_url,
            api_key_env=entry.get("api_key_env"),
            wire=wire,
        )
    return out


def load_providers(user_config: Path | None = None) -> dict[str, Provider]:
    """Load built-in providers, then overlay a user config on top."""
    merged = _load_builtin()
    default_user = Path.home() / ".verl-harness" / "providers.yaml"
    for path in (default_user, user_config):
        if path is None:
            continue
        merged.update(_load_user(path))
    return merged


def resolve_model_spec(
    spec: str,
    providers: dict[str, Provider] | None = None,
    *,
    env: dict[str, str] | None = None,
) -> ResolvedModel:
    """Split `<provider>/<model_id>` and resolve.

    Env override is exposed for tests; default reads from `os.environ`.
    """
    if "/" not in spec:
        raise ProviderError(
            f"model spec {spec!r} must be '<provider>/<model_id>'"
        )
    provider_name, model_id = spec.split("/", 1)
    if not model_id:
        raise ProviderError(f"model spec {spec!r} has empty model_id")

    table = providers if providers is not None else load_providers()
    if provider_name not in table:
        raise ProviderError(
            f"unknown provider {provider_name!r}. Known: {sorted(table)}"
        )
    provider = table[provider_name]

    api_key: str | None = None
    if provider.api_key_env:
        source = env if env is not None else os.environ
        api_key = source.get(provider.api_key_env)
        if not api_key:
            raise ProviderError(
                f"provider {provider_name!r} needs env var "
                f"{provider.api_key_env}, but it is unset"
            )
    return ResolvedModel(provider=provider, model_id=model_id, api_key=api_key)
