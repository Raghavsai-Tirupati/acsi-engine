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
    failure_reasons: dict[str, str] = field(default_factory=dict)

    def record(
        self,
        trace_id: str,
        passed: bool,
        max_failures: int,
        reason: str | None = None,
    ) -> None:
        if passed:
            self.pass_count += 1
            return
        self.fail_count += 1
        if len(self.failing_trace_ids) < max_failures:
            self.failing_trace_ids.append(trace_id)
            if reason is not None:
                self.failure_reasons[trace_id] = reason


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
            passed, reason = _evaluate_pair(assertion, pair)
            evaluation.record(pair.trace_id, passed, max_failures, reason=reason)
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


def _evaluate_pair(assertion: AssertionConfig, pair: AssertionPair) -> tuple[bool, str | None]:
    """Return (passed, failure_reason). The reason carries the validator message
    on failure so cluster naming and cert exemplars can be precise."""
    response = pair.candidate
    if assertion.type == "contains":
        expected = str(_config_value(assertion, "value", "text", default=""))
        if expected in (response.text or ""):
            return True, None
        return False, f"missing required substring {expected!r}"
    if assertion.type == "not_contains":
        forbidden = str(_config_value(assertion, "value", "text", default=""))
        if forbidden not in (response.text or ""):
            return True, None
        return False, f"forbidden substring present {forbidden!r}"
    if assertion.type == "regex":
        pattern = str(_config_value(assertion, "pattern", default=""))
        if re.search(pattern, response.text or "") is not None:
            return True, None
        return False, f"pattern {pattern!r} did not match"
    if assertion.type == "json_valid":
        if json_valid(response):
            return True, None
        return False, "response is not valid JSON"
    if assertion.type == "json_schema":
        return _json_schema_passes(assertion, response)
    if assertion.type == "numeric_field_equal":
        return _numeric_field_equal(assertion, response)
    if assertion.type == "length_range":
        return _length_range(assertion, response)
    if assertion.type == "refusal":
        if not (is_refusal(response) and not is_refusal(pair.baseline)):
            return True, None
        return False, "candidate refused but baseline did not"
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
        over = [
            pair
            for pair in pairs
            if pair.candidate.latency_ms is not None and pair.candidate.latency_ms > max_latency
        ][:max_failures]
        evaluation.failing_trace_ids = [pair.trace_id for pair in over]
        for pair in over:
            evaluation.failure_reasons[pair.trace_id] = (
                f"latency {pair.candidate.latency_ms}ms exceeds max {max_latency:.0f}ms "
                f"(suite p95 {p95:.0f}ms)"
            )


def _json_schema_passes(
    assertion: AssertionConfig,
    response: DiffResponse,
) -> tuple[bool, str | None]:
    schema = _config_value(assertion, "schema", default=None)
    if schema is None:
        raise ValueError("json_schema assertion requires an inline schema.")
    parsed = parse_json(response.text)
    if not parsed.parsed:
        return False, "response is not valid JSON"
    try:
        validate_json_schema(parsed.value, schema)
    except JsonSchemaValidationError as exc:
        return False, _schema_error_reason(exc)
    return True, None


def _schema_error_reason(exc: JsonSchemaValidationError) -> str:
    # Prefer a stable, mechanism-bearing message (path + validator + limit) over
    # jsonschema's default, which for maxLength quotes the entire offending value
    # and so fragments clustering. The limit (e.g. 400) is retained; the variable
    # value is not.
    field = ".".join(str(part) for part in exc.absolute_path) or "<root>"
    validator = getattr(exc, "validator", None)
    limit = getattr(exc, "validator_value", None)
    stable = {
        "maxLength": f"{field}: exceeds maxLength {limit}",
        "minLength": f"{field}: below minLength {limit}",
        "maximum": f"{field}: exceeds maximum {limit}",
        "minimum": f"{field}: below minimum {limit}",
        "maxItems": f"{field}: exceeds maxItems {limit}",
        "minItems": f"{field}: below minItems {limit}",
        "required": f"{field}: missing a required property",
        "additionalProperties": f"{field}: unexpected additional properties",
        "enum": f"{field}: not one of the allowed values",
        "type": f"{field}: expected type {limit}",
    }
    if validator in stable:
        return stable[validator]
    path = ".".join(str(part) for part in exc.absolute_path)
    return f"{path}: {exc.message}" if path else exc.message


def _numeric_field_equal(
    assertion: AssertionConfig,
    response: DiffResponse,
) -> tuple[bool, str | None]:
    field = str(_config_value(assertion, "field", "json_path", default=""))
    expected = _config_value(assertion, "expected", "value", default=None)
    parsed = parse_json(response.text)
    if not parsed.parsed:
        return False, "response is not valid JSON"
    value = _lookup_path(parsed.value, field)
    if (
        isinstance(value, int | float)
        and isinstance(expected, int | float)
        and value == expected
    ):
        return True, None
    return False, f"{field}={value!r} != expected {expected!r}"


def _length_range(
    assertion: AssertionConfig,
    response: DiffResponse,
) -> tuple[bool, str | None]:
    length = length_chars(response)
    if assertion.min_chars is not None and length < assertion.min_chars:
        return False, f"length {length} < min {assertion.min_chars}"
    if assertion.max_chars is not None and length > assertion.max_chars:
        return False, f"length {length} > max {assertion.max_chars}"
    return True, None


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
