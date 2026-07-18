from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Protocol

from acsi.replay.routing import provider_route


class ReplayClientError(RuntimeError):
    retryable = False
    # Provider-supplied hint (Retry-After / RetryInfo), in seconds, when present.
    retry_after_s: float | None = None


class RateLimitError(ReplayClientError):
    retryable = True


class TransientError(ReplayClientError):
    retryable = True


class PermanentError(ReplayClientError):
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        run_level: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.run_level = run_level
        self.status_code = status_code


@dataclass(frozen=True)
class CompletionRequest:
    provider: str
    model: str
    system: str | None
    messages: list[dict[str, str]]
    params: dict[str, Any] = field(default_factory=dict)
    sample_index: int = 0

    @property
    def prompt_text(self) -> str:
        parts = [self.system or ""]
        parts.extend(message.get("content", "") for message in self.messages)
        return "\n".join(part for part in parts if part)


@dataclass(frozen=True)
class CompletionResponse:
    text: str | None
    tool_calls: list[dict[str, Any]] | None
    finish_reason: str
    usage: dict[str, int]
    latency_ms: int
    served_model: str


class CompletionClient(Protocol):
    def complete(self, request: CompletionRequest) -> CompletionResponse: ...


ResponseTransform = Callable[[str, str], str]
PromptPredicate = Callable[[str], bool]


@dataclass(frozen=True)
class RegressionRule:
    predicate: PromptPredicate
    transform: ResponseTransform


class FakeClient:
    def __init__(
        self,
        *,
        seed: int = 42,
        noise: float = 0.0,
        regressions: Sequence[RegressionRule] | None = None,
        fail_rate_limit_every: int | None = None,
        retired_models: set[str] | None = None,
        rejected_prompt_predicate: PromptPredicate | None = None,
        served_model_override: str | None = None,
    ) -> None:
        self.seed = seed
        self.noise = noise
        self.regressions = tuple(regressions or ())
        self.fail_rate_limit_every = fail_rate_limit_every
        self.retired_models = retired_models or set()
        self.rejected_prompt_predicate = rejected_prompt_predicate
        self.served_model_override = served_model_override
        self.call_count = 0

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.call_count += 1
        if self.fail_rate_limit_every and self.call_count % self.fail_rate_limit_every == 0:
            raise RateLimitError("Fake provider rate limit.")
        if request.model in self.retired_models:
            raise PermanentError(
                retired_model_message(request.model),
                run_level=True,
                status_code=404,
            )

        prompt = request.prompt_text
        if self.rejected_prompt_predicate and self.rejected_prompt_predicate(prompt):
            raise PermanentError("Provider rejected this prompt.", run_level=False)

        prompt_hash = prompt_sha256(prompt)
        text = self._base_text(prompt, prompt_hash, request.sample_index)
        for regression in self.regressions:
            if regression.predicate(prompt):
                text = regression.transform(prompt, text)

        input_tokens = plausible_token_count(prompt)
        output_tokens = plausible_token_count(text)
        latency_ms = 120 + int(prompt_hash[:4], 16) % 80
        return CompletionResponse(
            text=text,
            tool_calls=None,
            finish_reason="stop",
            usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
            latency_ms=latency_ms,
            served_model=self.served_model_override or request.model,
        )

    def _base_text(self, prompt: str, prompt_hash: str, sample_index: int) -> str:
        variant = "summary"
        if self._noise_decision(prompt_hash, sample_index):
            variant = "paraphrased summary"
        if "json" in prompt.lower():
            return _json_summary(prompt_hash, variant)
        return f"{variant}: {prompt_hash[:16]}"

    def _noise_decision(self, prompt_hash: str, sample_index: int) -> bool:
        if self.noise <= 0:
            return False
        if self.noise >= 1:
            return True
        digest_input = f"{self.seed}:{prompt_hash}:{sample_index}".encode()
        digest = hashlib.sha256(digest_input).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        return value < self.noise


class LiveClient:
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        try:
            from acsi.replay.litellm_env import import_litellm

            litellm = import_litellm()
        except ImportError as exc:
            raise PermanentError("Install litellm to use live replay.", run_level=True) from exc

        route = provider_route(request.provider, request.model)
        completion_kwargs: dict[str, Any] = {
            "model": route.litellm_model,
            "messages": _litellm_messages(request),
            **request.params,
        }
        if route.api_base:
            completion_kwargs["api_base"] = route.api_base

        started = time.perf_counter()
        try:
            raw_response = litellm.completion(**completion_kwargs)
        except Exception as exc:
            raise map_litellm_error(exc, request.model) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        choice = raw_response.choices[0]
        message = choice.message
        usage = getattr(raw_response, "usage", None)
        return CompletionResponse(
            text=getattr(message, "content", None),
            tool_calls=getattr(message, "tool_calls", None),
            finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
            usage={
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            },
            latency_ms=latency_ms,
            served_model=str(getattr(raw_response, "model", request.model) or request.model),
        )


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _json_summary(prompt_hash: str, variant: str) -> str:
    return json.dumps(
        {
            "availability": "fixture availability",
            "candidate": prompt_hash[:12],
            "next_step": "schedule coordinator screen",
            "risks": [],
            "role_fit": variant,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def plausible_token_count(text: str | None) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def retired_model_message(model: str) -> str:
    return (
        f"Model {model} returned 404; it may be retired. "
        "Rerun with --degraded to certify against stored outputs."
    )


def live_client_keys_present() -> bool:
    return any(
        os.environ.get(name)
        for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
    )


def _litellm_messages(request: CompletionRequest) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.extend(request.messages)
    return messages


def map_litellm_error(exc: Exception, model: str) -> ReplayClientError:
    status_code = getattr(exc, "status_code", None)
    retry_after = _retry_after_seconds(exc)
    if _is_timeout(exc) or status_code == 408:
        return _with_retry_after(TransientError(f"request timed out: {exc}"), retry_after)
    if status_code == 429:
        return _with_retry_after(RateLimitError(str(exc)), retry_after)
    if status_code in {500, 502, 503, 504}:
        return _with_retry_after(TransientError(str(exc)), retry_after)
    if status_code in {401, 403, 404}:
        message = retired_model_message(model) if status_code == 404 else str(exc)
        return PermanentError(message, run_level=True, status_code=status_code)
    return PermanentError(str(exc), run_level=False, status_code=status_code)


def _with_retry_after(error: ReplayClientError, retry_after_s: float | None) -> ReplayClientError:
    error.retry_after_s = retry_after_s
    return error


def _is_timeout(exc: Exception) -> bool:
    return type(exc).__name__ in {"Timeout", "APITimeoutError"}


def _retry_after_seconds(exc: Exception) -> float | None:
    """Best-effort extraction of a provider retry hint (seconds).

    Reads an explicit `retry_after` attribute or an HTTP `Retry-After` header.
    Only integer-second forms are honored; HTTP-date forms are ignored.
    """
    direct = getattr(exc, "retry_after", None)
    if isinstance(direct, int | float):
        return float(direct)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = None
    with suppress(AttributeError, TypeError):
        raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
