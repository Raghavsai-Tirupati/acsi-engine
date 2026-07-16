from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema

from acsi.diff.deterministic import (
    DiffResponse,
    is_refusal,
    json_valid,
    length_chars,
    parse_json,
)
from acsi.schemas import AssertionConfig, Severity

AssertionStatus = Literal["passed", "failed", "deferred_to_judge"]


@dataclass(frozen=True)
class AssertionPair:
    trace_id: str
    baseline: DiffResponse
    candidate: DiffResponse


@dataclass
class AssertionEvaluation:
    assertion_id: str
    severity: Severity
    status: AssertionStatus
    pass_count: int = 0
    fail_count: int = 0
    failing_trace_ids: list[str] = field(default_factory=list)

    def record(self, trace_id: str, passed: bool, max_failures: int) -> None:
        if passed:
            self.pass_count += 1
            return
        self.fail_count += 1
        if len(self.failing_trace_ids) < max_failures:
            self.failing_trace_ids.append(trace_id)


def evaluate_assertion(
    assertion: AssertionConfig,
    pairs: list[AssertionPair],
    *,
    max_failures: int = 20,
) -> AssertionEvaluation:
    if assertion.type == "judge_classifier":
        return AssertionEvaluation(
            assertion_id=assertion.id,
            severity=assertion.severity,
            status="deferred_to_judge",
        )

    evaluation = AssertionEvaluation(
        assertion_id=assertion.id,
        severity=assertion.severity,
        status="passed",
    )
    if assertion.type == "latency_p95_ms":
        _evaluate_latency_p95(assertion, pairs, evaluation, max_failures)
    else:
        for pair in pairs:
            evaluation.record(
                pair.trace_id,
                _evaluate_pair(assertion, pair),
                max_failures,
            )
    if evaluation.fail_count:
        evaluation.status = "failed"
    return evaluation


def evaluate_assertions(
    assertions: list[AssertionConfig],
    pairs: list[AssertionPair],
    *,
    max_failures: int = 20,
) -> list[AssertionEvaluation]:
    return [
        evaluate_assertion(assertion, pairs, max_failures=max_failures)
        for assertion in assertions
    ]


def _evaluate_pair(assertion: AssertionConfig, pair: AssertionPair) -> bool:
    response = pair.candidate
    if assertion.type == "contains":
        expected = str(_config_value(assertion, "value", "text", default=""))
        return expected in (response.text or "")
    if assertion.type == "not_contains":
        forbidden = str(_config_value(assertion, "value", "text", default=""))
        return forbidden not in (response.text or "")
    if assertion.type == "regex":
        pattern = str(_config_value(assertion, "pattern", default=""))
        return re.search(pattern, response.text or "") is not None
    if assertion.type == "json_valid":
        return json_valid(response)
    if assertion.type == "json_schema":
        return _json_schema_passes(assertion, response)
    if assertion.type == "numeric_field_equal":
        return _numeric_field_equal(assertion, response)
    if assertion.type == "length_range":
        return _length_range(assertion, response)
    if assertion.type == "refusal":
        return not (is_refusal(response) and not is_refusal(pair.baseline))
    raise ValueError(f"Unsupported assertion type for deterministic evaluation: {assertion.type}")


def _evaluate_latency_p95(
    assertion: AssertionConfig,
    pairs: list[AssertionPair],
    evaluation: AssertionEvaluation,
    max_failures: int,
) -> None:
    max_latency = assertion.max
    if max_latency is None:
        raise ValueError("latency_p95_ms assertion requires max.")
    latencies = [
        pair.candidate.latency_ms
        for pair in pairs
        if pair.candidate.latency_ms is not None
    ]
    if not latencies:
        evaluation.fail_count = 1
        return
    p95 = float(np.percentile(np.asarray(latencies, dtype=np.float64), 95))
    if p95 <= max_latency:
        evaluation.pass_count = len(latencies)
    else:
        evaluation.fail_count = 1
        evaluation.failing_trace_ids = [
            pair.trace_id
            for pair in pairs
            if pair.candidate.latency_ms is not None and pair.candidate.latency_ms > max_latency
        ][:max_failures]


def _json_schema_passes(assertion: AssertionConfig, response: DiffResponse) -> bool:
    schema = _config_value(assertion, "schema", default=None)
    if schema is None:
        raise ValueError("json_schema assertion requires inline schema for M3.")
    parsed = parse_json(response.text)
    if not parsed.parsed:
        return False
    try:
        validate_json_schema(parsed.value, schema)
    except JsonSchemaValidationError:
        return False
    return True


def _numeric_field_equal(assertion: AssertionConfig, response: DiffResponse) -> bool:
    field = str(_config_value(assertion, "field", "json_path", default=""))
    expected = _config_value(assertion, "expected", "value", default=None)
    parsed = parse_json(response.text)
    if not parsed.parsed:
        return False
    value = _lookup_path(parsed.value, field)
    return (
        isinstance(value, int | float)
        and isinstance(expected, int | float)
        and value == expected
    )


def _length_range(assertion: AssertionConfig, response: DiffResponse) -> bool:
    length = length_chars(response)
    if assertion.min_chars is not None and length < assertion.min_chars:
        return False
    return not (assertion.max_chars is not None and length > assertion.max_chars)


def _lookup_path(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _config_value(assertion: AssertionConfig, *keys: str, default: Any) -> Any:
    extras = assertion.model_extra or {}
    for key in keys:
        if key in extras:
            return extras[key]
    return default
