from __future__ import annotations

import pytest

from acsi.replay.routing import (
    DEFAULT_LOCAL_BASE_URL,
    LOCAL_BASE_URL_ENV,
    UnknownProviderError,
    provider_route,
    required_api_key_env,
)


def test_openai_routes_to_bare_model_and_openai_key() -> None:
    route = provider_route("openai", "gpt-5.4-mini")

    assert route.provider == "openai"
    assert route.litellm_model == "gpt-5.4-mini"
    assert route.api_key_env == "OPENAI_API_KEY"
    assert route.api_base is None


def test_google_routes_via_gemini_ai_studio_not_vertex() -> None:
    route = provider_route("google", "gemini-3.5-flash")

    assert route.litellm_model == "gemini/gemini-3.5-flash"
    assert "vertex" not in route.litellm_model
    assert route.api_key_env == "GEMINI_API_KEY"


def test_anthropic_routes_to_bare_model_and_anthropic_key() -> None:
    route = provider_route("anthropic", "claude-sonnet-5")

    assert route.litellm_model == "claude-sonnet-5"
    assert route.api_key_env == "ANTHROPIC_API_KEY"


def test_local_routes_to_openai_compatible_base_url_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(LOCAL_BASE_URL_ENV, raising=False)
    route = provider_route("local", "llama3")

    assert route.litellm_model == "openai/llama3"
    assert route.api_key_env is None
    assert route.api_base == DEFAULT_LOCAL_BASE_URL


def test_local_base_url_is_overridable_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOCAL_BASE_URL_ENV, "http://ollama.internal:1234/v1")

    assert provider_route("local", "llama3").api_base == "http://ollama.internal:1234/v1"


def test_redundant_provider_prefix_is_stripped() -> None:
    assert provider_route("openai", "openai/gpt-judge").litellm_model == "gpt-judge"
    assert provider_route("google", "google/gemini-judge").litellm_model == "gemini/gemini-judge"
    assert provider_route("local", "local:llama3").litellm_model == "openai/llama3"


def test_unknown_provider_is_actionable() -> None:
    with pytest.raises(UnknownProviderError, match="Unknown provider 'cohere'"):
        provider_route("cohere", "command-r")


def test_required_api_key_env_table() -> None:
    assert required_api_key_env("openai") == "OPENAI_API_KEY"
    assert required_api_key_env("google") == "GEMINI_API_KEY"
    assert required_api_key_env("anthropic") == "ANTHROPIC_API_KEY"
    assert required_api_key_env("local") is None
