from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from acsi.judge.clients import judge_entry_provider
from acsi.replay.clients import (
    CompletionClient,
    CompletionRequest,
    RateLimitError,
    ReplayClientError,
)
from acsi.replay.params import transform_params
from acsi.replay.routing import required_api_key_env
from acsi.replay.runner import estimate_call_cost_usd
from acsi.schemas import WorkloadManifest

PREFLIGHT_PROMPT = "ping"
PREFLIGHT_MAX_TOKENS = 1


@dataclass(frozen=True)
class ModelCheck:
    role: str
    provider: str
    requested_model: str
    served_model: str | None
    latency_ms: int | None
    ok: bool
    error: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "provider": self.provider,
            "requested_model": self.requested_model,
            "served_model": self.served_model,
            "latency_ms": self.latency_ms,
            "ok": self.ok,
            "error": self.error,
        }


@dataclass(frozen=True)
class PreflightReport:
    ok: bool
    missing_keys: list[str]
    required_keys: dict[str, str]
    checks: list[ModelCheck]
    estimated_cost_usd: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.ok else "error",
            "ok": self.ok,
            "missing_keys": self.missing_keys,
            "required_keys": self.required_keys,
            "checks": [check.to_payload() for check in self.checks],
            "estimated_cost_usd": self.estimated_cost_usd,
        }


def collect_targets(manifest: WorkloadManifest) -> list[tuple[str, str, str]]:
    """Return (role, provider, model) for baseline, candidate, and every judge."""
    targets: list[tuple[str, str, str]] = [
        ("baseline", manifest.baseline.provider, manifest.baseline.model),
        ("candidate", manifest.candidate.provider, manifest.candidate.model),
    ]
    for entry in manifest.judging.judges or []:
        targets.append(("judge", judge_entry_provider(entry), entry.model))
    return targets


def run_preflight(
    manifest: WorkloadManifest,
    *,
    client: CompletionClient,
    env: Mapping[str, str] | None = None,
    fake: bool = False,
) -> PreflightReport:
    """Verify credentials and, when present, make one minimal completion per model.

    Never prints or returns secret values — only environment variable NAMES.
    """
    env_map = env if env is not None else os.environ
    targets = collect_targets(manifest)

    required_keys: dict[str, str] = {}
    missing: set[str] = set()
    for provider in _distinct(t[1] for t in targets):
        key_env = required_api_key_env(provider)
        if key_env is None:
            continue
        required_keys[provider] = key_env
        if not env_map.get(key_env):
            missing.add(key_env)

    if missing:
        # Keys absent — report by name and stop before any provider call.
        return PreflightReport(
            ok=False,
            missing_keys=sorted(missing),
            required_keys=dict(sorted(required_keys.items())),
            checks=[],
            estimated_cost_usd=0.0,
        )

    checks: list[ModelCheck] = []
    estimated_cost = 0.0
    seen: set[tuple[str, str]] = set()
    for role, provider, model in targets:
        if (provider, model) in seen:
            continue
        seen.add((provider, model))
        check, cost = _probe_model(role, provider, model, client=client, fake=fake)
        checks.append(check)
        estimated_cost += cost

    ok = all(check.ok for check in checks)
    return PreflightReport(
        ok=ok,
        missing_keys=[],
        required_keys=dict(sorted(required_keys.items())),
        checks=checks,
        estimated_cost_usd=estimated_cost,
    )


def _probe_model(
    role: str,
    provider: str,
    model: str,
    *,
    client: CompletionClient,
    fake: bool,
) -> tuple[ModelCheck, float]:
    params, _transforms = transform_params(provider, model, {"max_tokens": PREFLIGHT_MAX_TOKENS})
    request = CompletionRequest(
        provider=provider,
        model=model,
        system=None,
        messages=[{"role": "user", "content": PREFLIGHT_PROMPT}],
        params=params,
    )
    try:
        response = client.complete(request)
    except ReplayClientError as exc:
        return (
            ModelCheck(
                role=role,
                provider=provider,
                requested_model=model,
                served_model=None,
                latency_ms=None,
                ok=False,
                error=_error_line(exc, provider, model),
            ),
            0.0,
        )
    cost = estimate_call_cost_usd(
        provider,
        model,
        response.usage.get("input_tokens", 0),
        response.usage.get("output_tokens", 0),
        fake=fake,
    )
    return (
        ModelCheck(
            role=role,
            provider=provider,
            requested_model=model,
            served_model=response.served_model,
            latency_ms=response.latency_ms,
            ok=True,
            error=None,
        ),
        cost,
    )


def _error_line(exc: ReplayClientError, provider: str, model: str) -> str:
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        key_env = required_api_key_env(provider) or "the provider credential"
        return (
            f"{provider}/{model}: authentication failed (HTTP {status}); "
            f"check that {key_env} holds a valid key."
        )
    if status == 404:
        # retired_model_message already carries the retired-model hint.
        return f"{provider}/{model}: {exc}"
    if status == 429 or isinstance(exc, RateLimitError):
        return (
            f"{provider}/{model}: rate limited or over quota (HTTP 429); "
            "retry later or raise the provider quota."
        )
    return f"{provider}/{model}: {exc}"


def _distinct(values: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered
