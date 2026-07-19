from __future__ import annotations

import pytest

from acsi.judge.rubric import (
    JudgeParseError,
    PairwiseJudgment,
    map_position_verdict,
    parse_classifier_judgment,
    parse_pairwise_judgment,
    reconcile_position_outcomes,
    render_pairwise_rubric,
)


def test_pairwise_rubric_is_blinded() -> None:
    rendered = render_pairwise_rubric(
        "Summarize the application.",
        "Response one.",
        "Response two.",
    )

    lowered = rendered.lower()
    assert "baseline" not in lowered
    assert "candidate" not in lowered
    assert "Response A" in rendered
    assert "Response B" in rendered


def test_pairwise_parse_is_schema_validated() -> None:
    parsed = parse_pairwise_judgment(
        '{"verdict":"a_better","severity_if_worse":"minor","reason":"A is clearer."}'
    )

    assert parsed.verdict == "a_better"
    assert parsed.severity_if_worse == "minor"

    with pytest.raises(JudgeParseError):
        parse_pairwise_judgment('{"verdict":"bad","severity_if_worse":null,"reason":"x"}')


def test_pairwise_parse_recovers_fenced_and_prefixed_json() -> None:
    # gemini-style ```json fence around a valid verdict (run 0a716021 lost 848/851
    # gemini verdicts to exactly this wrapping).
    fenced = parse_pairwise_judgment(
        '```json\n{\n  "verdict": "a_better",\n  '
        '"severity_if_worse": "minor",\n  "reason": "A is clearer."\n}\n```'
    )
    assert fenced.verdict == "a_better"
    assert fenced.severity_if_worse == "minor"

    # Bare ``` fence with a leading preamble line.
    prefixed = parse_pairwise_judgment(
        'Here is my verdict:\n```\n{"verdict":"b_better",'
        '"severity_if_worse":"critical","reason":"B wins."}\n```'
    )
    assert prefixed.verdict == "b_better"
    assert prefixed.severity_if_worse == "critical"


def test_pairwise_parse_coerces_string_null_severity() -> None:
    parsed = parse_pairwise_judgment(
        '```json\n{"verdict":"b_better","severity_if_worse":"null","reason":"B wins."}\n```'
    )
    assert parsed.verdict == "b_better"
    assert parsed.severity_if_worse is None


def test_pairwise_parse_still_rejects_unparseable_and_invalid() -> None:
    with pytest.raises(JudgeParseError):
        parse_pairwise_judgment("no json object here at all")
    with pytest.raises(JudgeParseError):
        parse_pairwise_judgment('```json\n{"verdict":"bad","severity_if_worse":null,"reason":"x"}\n```')


def test_classifier_parse_recovers_fenced_json() -> None:
    parsed = parse_classifier_judgment('```json\n{"pass": true, "reason": "Meets it."}\n```')
    assert parsed.passed


def test_position_verdict_maps_back_to_candidate_outcome() -> None:
    assert (
        map_position_verdict(
            PairwiseJudgment("a_better", "critical", "A is better."),
            candidate_position="a",
        )
        == "candidate_better"
    )
    assert (
        map_position_verdict(
            PairwiseJudgment("a_better", "critical", "A is better."),
            candidate_position="b",
        )
        == "worse_critical"
    )


def test_reconcile_same_harm_class_resolves_conservatively() -> None:
    # Same side of the harm boundary: reconcile, do not abstain.
    assert reconcile_position_outcomes("candidate_better", "equivalent") == ("equivalent", None)
    assert reconcile_position_outcomes("equivalent", "candidate_better") == ("equivalent", None)
    assert reconcile_position_outcomes("candidate_better", "candidate_better") == (
        "candidate_better",
        None,
    )
    # Both worse: milder unless both agree critical.
    assert reconcile_position_outcomes("worse_minor", "worse_critical") == ("worse_minor", None)
    assert reconcile_position_outcomes("worse_critical", "worse_critical") == (
        "worse_critical",
        None,
    )
    assert reconcile_position_outcomes("equivalent", "equivalent") == ("equivalent", None)


def test_reconcile_cross_harm_class_abstains() -> None:
    outcome, reason = reconcile_position_outcomes("candidate_better", "worse_minor")
    assert outcome is None
    assert reason == "position_inconsistency"
    outcome, reason = reconcile_position_outcomes("equivalent", "worse_critical")
    assert outcome is None
    assert reason == "position_inconsistency"


def test_classifier_parse_is_schema_validated() -> None:
    parsed = parse_classifier_judgment('{"pass":true,"reason":"Meets criterion."}')

    assert parsed.passed
    with pytest.raises(JudgeParseError):
        parse_classifier_judgment('{"pass":"yes","reason":"bad"}')
