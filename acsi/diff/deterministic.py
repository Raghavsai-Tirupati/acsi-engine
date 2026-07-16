from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

REFUSAL_MARKERS = (
    "i can't",
    "i cannot",
    "i am unable",
    "i'm unable",
    "cannot comply",
    "can't comply",
    "not able to",
    "i won't",
    "i will not",
    "sorry, but i",
)
REFUSAL_FINISH_REASONS = {"content_filter", "safety", "refusal"}


@dataclass(frozen=True)
class DiffResponse:
    text: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    latency_ms: int | None = None


@dataclass(frozen=True)
class DeterministicPairResult:
    equivalent: bool
    reason: str | None = None


def deterministic_pair_equivalence(
    baseline: DiffResponse,
    candidate: DiffResponse,
) -> DeterministicPairResult:
    if baseline.tool_calls is not None or candidate.tool_calls is not None:
        if canonical_json(baseline.tool_calls) == canonical_json(candidate.tool_calls):
            return DeterministicPairResult(equivalent=True, reason="tool_calls_equal")
        return DeterministicPairResult(equivalent=False, reason=None)

    baseline_text = baseline.text or ""
    candidate_text = candidate.text or ""
    if normalized_text(baseline_text) == normalized_text(candidate_text):
        return DeterministicPairResult(equivalent=True, reason="normalized_text_equal")

    baseline_json = parse_json(baseline_text)
    candidate_json = parse_json(candidate_text)
    if (
        baseline_json.parsed
        and candidate_json.parsed
        and canonical_json(baseline_json.value) == canonical_json(candidate_json.value)
    ):
        return DeterministicPairResult(equivalent=True, reason="canonical_json_equal")

    return DeterministicPairResult(equivalent=False, reason=None)


def normalized_text(value: str) -> str:
    return " ".join(value.strip().split())


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class JsonParseResult:
    parsed: bool
    value: Any | None = None
    error: str | None = None


def parse_json(value: str | None) -> JsonParseResult:
    if value is None:
        return JsonParseResult(parsed=False, error="missing text")
    try:
        return JsonParseResult(parsed=True, value=json.loads(value))
    except json.JSONDecodeError as exc:
        return JsonParseResult(parsed=False, error=str(exc))


def json_valid(response: DiffResponse) -> bool:
    return parse_json(response.text).parsed


def is_refusal(response: DiffResponse) -> bool:
    finish_reason = (response.finish_reason or "").lower()
    if finish_reason in REFUSAL_FINISH_REASONS:
        return True
    text = (response.text or "").lower()
    return any(marker in text for marker in REFUSAL_MARKERS)


def length_chars(response: DiffResponse) -> int:
    return len(response.text or "")


def latency_ms(response: DiffResponse) -> int | None:
    return response.latency_ms


def regex_matches(response: DiffResponse, pattern: str) -> bool:
    return re.search(pattern, response.text or "") is not None
