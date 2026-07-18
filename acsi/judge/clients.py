from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from acsi.judge.rubric import CandidateOutcome
from acsi.replay.clients import (
    CompletionRequest,
    CompletionResponse,
    PermanentError,
    map_litellm_error,
    plausible_token_count,
)
from acsi.replay.routing import provider_route
from acsi.schemas import JudgeModelConfig, WorkloadManifest

Oracle = Callable[[str], CandidateOutcome]
ClassifierOracle = Callable[[str], bool]
Ordering = Literal["candidate_a", "candidate_b"]


class JudgeConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class JudgeSpec:
    model: str
    family: str
    provider: str = ""
    litellm_model: str = ""
    api_base: str | None = None


class FakeJudge:
    def __init__(
        self,
        *,
        model: str = "openai/fake-judge",
        seed: int = 42,
        oracle: Oracle | None = None,
        classifier_oracle: ClassifierOracle | None = None,
        error_rate: float = 0.0,
        positional_bias: float = 0.0,
        malformed_calls: set[int] | None = None,
        malformed_attempts: set[tuple[str, str, int]] | None = None,
    ) -> None:
        self.model = model
        self.seed = seed
        self.uses_default_oracle = oracle is None
        self.oracle = oracle or (lambda _pair_id: "equivalent")
        self.classifier_oracle = classifier_oracle or (lambda _pair_id: True)
        self.error_rate = error_rate
        self.positional_bias = positional_bias
        self.malformed_calls = malformed_calls or set()
        self.malformed_attempts = malformed_attempts or set()
        self.call_count = 0

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.call_count += 1
        pair_id = str(request.params.get("pair_id", "unknown"))
        ordering = str(request.params.get("ordering", "single"))
        attempt = int(request.params.get("attempt", request.sample_index))
        if (
            self.call_count in self.malformed_calls
            or (pair_id, ordering, attempt) in self.malformed_attempts
        ):
            text = "{malformed"
        elif request.params.get("mode") == "classifier":
            text = self._classifier_text(pair_id, ordering)
        else:
            text = self._pairwise_text(pair_id, _ordering_value(ordering), request.prompt_text)

        return CompletionResponse(
            text=text,
            tool_calls=None,
            finish_reason="stop",
            usage={
                "input_tokens": plausible_token_count(request.prompt_text),
                "output_tokens": plausible_token_count(text),
            },
            latency_ms=25 + int(_hash_bytes(self.seed, pair_id, ordering)[0]),
            served_model=self.model,
        )

    def _pairwise_text(self, pair_id: str, ordering: Ordering, prompt_text: str) -> str:
        outcome = self.oracle(pair_id)
        if self.uses_default_oracle and "ACSI_DEMO_BROKEN_JSON" in prompt_text:
            outcome = "worse_critical"
        if _draw(self.seed, pair_id, ordering, "error") < self.error_rate:
            outcome = _wrong_outcome(outcome)
        if _draw(self.seed, pair_id, ordering, "position") < self.positional_bias:
            verdict = "a_better"
            severity = "minor"
        else:
            verdict, severity = _position_verdict(outcome, ordering)
        return json.dumps(
            {
                "reason": "Deterministic fake judge decision.",
                "severity_if_worse": severity,
                "verdict": verdict,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _classifier_text(self, pair_id: str, ordering: str) -> str:
        passed = self.classifier_oracle(pair_id)
        if _draw(self.seed, pair_id, ordering, "classifier_error") < self.error_rate:
            passed = not passed
        return json.dumps(
            {"pass": passed, "reason": "Deterministic fake classifier decision."},
            sort_keys=True,
            separators=(",", ":"),
        )


class LiveJudge:
    DEFAULT_TIMEOUT_S = 120.0

    def __init__(
        self,
        litellm_model: str,
        *,
        api_base: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.litellm_model = litellm_model
        self.api_base = api_base
        self.timeout_s = timeout_s

    @classmethod
    def from_spec(cls, spec: JudgeSpec, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> LiveJudge:
        return cls(spec.litellm_model, api_base=spec.api_base, timeout_s=timeout_s)

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        try:
            from acsi.replay.litellm_env import import_litellm

            litellm = import_litellm()
        except ImportError as exc:
            raise PermanentError(
                "Install litellm to use live judge calls.",
                run_level=True,
            ) from exc

        started = time.perf_counter()
        kwargs: dict[str, object] = {"timeout": self.timeout_s}
        if self.api_base:
            kwargs["api_base"] = self.api_base
        try:
            raw_response = litellm.completion(
                model=self.litellm_model,
                messages=[{"role": "user", "content": request.prompt_text}],
                **kwargs,
            )
        except Exception as exc:
            raise map_litellm_error(exc, self.litellm_model) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        choice = raw_response.choices[0]
        usage = getattr(raw_response, "usage", None)
        text = getattr(choice.message, "content", None)
        return CompletionResponse(
            text=text,
            tool_calls=None,
            finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
            usage={
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            },
            latency_ms=latency_ms,
            served_model=str(
                getattr(raw_response, "model", self.litellm_model) or self.litellm_model
            ),
        )


def select_judge_panel(manifest: WorkloadManifest) -> list[JudgeSpec]:
    configured = manifest.judging.judges
    if not configured:
        raise JudgeConfigurationError(
            "judging.judges is required; add explicit judge model strings to the manifest."
        )
    excluded = {manifest.baseline.provider, manifest.candidate.provider}
    allowed = set(manifest.judging.families_allowed) - excluded
    panel = [
        _build_judge_spec(judge)
        for judge in configured
        if judge_entry_provider(judge) in allowed
    ]
    if len(panel) < manifest.judging.min_judges:
        excluded_text = ", ".join(
            f"{family} (baseline/candidate)" for family in sorted(excluded)
        )
        raise JudgeConfigurationError(
            "Not enough eligible judges after family exclusion. "
            f"Excluded families: {excluded_text}. "
            f"Allowed remaining families: {', '.join(sorted(allowed)) or 'none'}. "
            f"Need {manifest.judging.min_judges}; configured eligible {len(panel)}."
        )
    return panel


def judge_entry_provider(entry: JudgeModelConfig) -> str:
    # Prefer the explicit pinned {provider, model} field; fall back to parsing a
    # legacy "{provider}/{model}" string.
    return entry.provider or judge_family(entry.model)


def _build_judge_spec(entry: JudgeModelConfig) -> JudgeSpec:
    provider = judge_entry_provider(entry)
    route = provider_route(provider, entry.model)
    return JudgeSpec(
        model=entry.model,
        family=provider,
        provider=provider,
        litellm_model=route.litellm_model,
        api_base=route.api_base,
    )


def judge_family(model: str) -> str:
    if "/" in model:
        return model.split("/", 1)[0]
    return model.split(":", 1)[0]


def _position_verdict(
    outcome: CandidateOutcome,
    ordering: Ordering,
) -> tuple[str, str | None]:
    candidate_position = "a" if ordering == "candidate_a" else "b"
    other_position = "b" if candidate_position == "a" else "a"
    if outcome == "equivalent":
        return "equivalent", None
    if outcome == "candidate_better":
        return f"{candidate_position}_better", None
    severity = "critical" if outcome == "worse_critical" else "minor"
    return f"{other_position}_better", severity


def _wrong_outcome(outcome: CandidateOutcome) -> CandidateOutcome:
    alternatives: list[CandidateOutcome] = [
        "equivalent",
        "candidate_better",
        "worse_minor",
        "worse_critical",
    ]
    alternatives.remove(outcome if outcome != "unresolved" else "equivalent")
    return alternatives[0]


def _ordering_value(value: str) -> Ordering:
    return "candidate_a" if value == "candidate_a" else "candidate_b"


def _draw(seed: int, pair_id: str, ordering: str, channel: str) -> float:
    digest = _hash_bytes(seed, pair_id, ordering, channel)
    return int.from_bytes(digest[:8], "big") / 2**64


def _hash_bytes(seed: int, pair_id: str, ordering: str, channel: str = "") -> bytes:
    return hashlib.sha256(f"{seed}:{pair_id}:{ordering}:{channel}".encode()).digest()
