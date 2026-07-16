from __future__ import annotations

import pytest

from acsi.judge.rubric import (
    JudgeParseError,
    PairwiseJudgment,
    map_position_verdict,
    parse_classifier_judgment,
    parse_pairwise_judgment,
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


def test_classifier_parse_is_schema_validated() -> None:
    parsed = parse_classifier_judgment('{"pass":true,"reason":"Meets criterion."}')

    assert parsed.passed
    with pytest.raises(JudgeParseError):
        parse_classifier_judgment('{"pass":"yes","reason":"bad"}')
