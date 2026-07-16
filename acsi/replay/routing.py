from __future__ import annotations

import os
from dataclasses import dataclass

# Pinned July 2026 provider → litellm routing. Each provider family maps to the
# litellm model-string shape and the single environment variable that must hold
# its credential. `local` is an OpenAI-compatible endpoint (Ollama) and needs no
# API key, only a base URL (which has a working default), so its key env is None.
DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"
LOCAL_BASE_URL_ENV = "ACSI_LOCAL_JUDGE_URL"

PROVIDER_API_KEY_ENV: dict[str, str | None] = {
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "local": None,
}

KNOWN_PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "google", "local")


class UnknownProviderError(ValueError):
    pass


@dataclass(frozen=True)
class ProviderRoute:
    """Resolved litellm routing for a single (provider, model)."""

    provider: str
    litellm_model: str
    api_key_env: str | None
    api_base: str | None = None


def provider_route(provider: str, model: str) -> ProviderRoute:
    """Map a provider family and model id onto its litellm route.

    - openai   → bare model id, OPENAI_API_KEY.
    - google   → "gemini/{model}" via Google AI Studio, GEMINI_API_KEY
      (Google AI Studio, NOT vertex_ai — there are no GCP credentials here).
    - anthropic → bare model id, ANTHROPIC_API_KEY.
    - local    → OpenAI-compatible "openai/{model}" against ACSI_LOCAL_JUDGE_URL.
    """
    bare = _strip_provider_prefix(provider, model)
    if provider == "openai":
        return ProviderRoute("openai", bare, "OPENAI_API_KEY")
    if provider == "google":
        return ProviderRoute("google", f"gemini/{bare}", "GEMINI_API_KEY")
    if provider == "anthropic":
        return ProviderRoute("anthropic", bare, "ANTHROPIC_API_KEY")
    if provider == "local":
        return ProviderRoute(
            "local",
            f"openai/{bare}",
            None,
            api_base=os.environ.get(LOCAL_BASE_URL_ENV, DEFAULT_LOCAL_BASE_URL),
        )
    raise UnknownProviderError(
        f"Unknown provider '{provider}'. Known providers: {', '.join(KNOWN_PROVIDERS)}."
    )


def required_api_key_env(provider: str) -> str | None:
    """Return the env var that must hold this provider's credential, or None."""
    if provider not in PROVIDER_API_KEY_ENV:
        raise UnknownProviderError(
            f"Unknown provider '{provider}'. Known providers: {', '.join(KNOWN_PROVIDERS)}."
        )
    return PROVIDER_API_KEY_ENV[provider]


def _strip_provider_prefix(provider: str, model: str) -> str:
    # SPEC-NOTE: judge entries may carry either the pinned {provider, model} shape
    # (bare model id) or a legacy "{provider}/{model}" / "{provider}:{model}" string.
    # Strip a redundant provider prefix so both resolve to the same litellm route.
    for separator in ("/", ":"):
        prefix = f"{provider}{separator}"
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model
