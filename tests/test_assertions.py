from __future__ import annotations

from acsi.diff.assertions import AssertionPair, evaluate_assertion
from acsi.diff.deterministic import DiffResponse
from acsi.schemas import AssertionConfig, Severity


def test_assertion_engine_text_and_regex_cases() -> None:
    pairs = [_pair("t1", "baseline", "hello safe world")]

    contains = _assertion("contains", value="safe")
    not_contains = _assertion("not_contains", value="unsafe")
    regex = _assertion("regex", pattern=r"safe\s+world")

    assert evaluate_assertion(contains, pairs).status == "passed"
    assert evaluate_assertion(not_contains, pairs).status == "passed"
    assert evaluate_assertion(regex, pairs).status == "passed"

    failed = evaluate_assertion(_assertion("contains", value="missing"), pairs)
    assert failed.status == "failed"
    assert failed.fail_count == 1
    assert failed.failing_trace_ids == ["t1"]
    assert failed.severity == Severity.CRITICAL


def test_assertion_engine_json_cases() -> None:
    pairs = [_pair("t1", "{}", '{"score": 7, "name": "A"}')]

    assert evaluate_assertion(_assertion("json_valid"), pairs).status == "passed"
    assert (
        evaluate_assertion(
            _assertion(
                "json_schema",
                schema={
                    "type": "object",
                    "required": ["score"],
                    "properties": {"score": {"type": "number"}},
                },
            ),
            pairs,
        ).status
        == "passed"
    )
    assert (
        evaluate_assertion(
            _assertion("numeric_field_equal", field="score", expected=7),
            pairs,
        ).status
        == "passed"
    )
    assert (
        evaluate_assertion(_assertion("json_valid"), [_pair("bad", "{}", "{bad")]).status
        == "failed"
    )


def test_assertion_engine_length_latency_and_refusal() -> None:
    pairs = [
        AssertionPair(
            trace_id="t1",
            baseline=DiffResponse(text="answer", latency_ms=50),
            candidate=DiffResponse(text="answer", latency_ms=100),
        ),
        AssertionPair(
            trace_id="t2",
            baseline=DiffResponse(text="answer", latency_ms=50),
            candidate=DiffResponse(text="Sorry, but I cannot comply.", latency_ms=200),
        ),
    ]

    assert (
        evaluate_assertion(_assertion("length_range", min_chars=1, max_chars=40), pairs).status
        == "passed"
    )
    assert evaluate_assertion(_assertion("latency_p95_ms", max=250), pairs).status == "passed"

    refusal = evaluate_assertion(_assertion("refusal"), pairs)
    assert refusal.status == "failed"
    assert refusal.failing_trace_ids == ["t2"]


def test_judge_classifier_is_deferred() -> None:
    result = evaluate_assertion(
        _assertion("judge_classifier", severity=Severity.MINOR),
        [_pair("t1", "a", "b")],
    )

    assert result.status == "deferred_to_judge"
    assert result.severity == Severity.MINOR
    assert result.pass_count == 0
    assert result.fail_count == 0


def test_assertion_failures_carry_validator_reasons() -> None:
    schema = _assertion(
        "json_schema",
        schema={
            "type": "object",
            "properties": {"summary": {"type": "string", "maxLength": 5}},
        },
    )
    schema_eval = evaluate_assertion(schema, [_pair("t1", "{}", '{"summary": "way too long"}')])
    assert schema_eval.status == "failed"
    assert "summary" in schema_eval.failure_reasons["t1"]

    length = _assertion("length_range", max_chars=3)
    length_eval = evaluate_assertion(length, [_pair("t2", "ok", "far too long")])
    assert length_eval.failure_reasons["t2"] == "length 12 > max 3"

    fence = _assertion("not_contains", value="```")
    fence_eval = evaluate_assertion(fence, [_pair("t3", "ok", "here is ``` a fence")])
    assert "```" in fence_eval.failure_reasons["t3"]


def _pair(trace_id: str, baseline: str, candidate: str) -> AssertionPair:
    return AssertionPair(
        trace_id=trace_id,
        baseline=DiffResponse(text=baseline),
        candidate=DiffResponse(text=candidate),
    )


def _assertion(
    assertion_type: str,
    severity: Severity = Severity.CRITICAL,
    **extra,
) -> AssertionConfig:
    return AssertionConfig(
        id=f"{assertion_type}-assertion",
        type=assertion_type,
        severity=severity,
        **extra,
    )
