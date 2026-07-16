from __future__ import annotations

from pathlib import Path

import pytest

from acsi.diff.deterministic import DiffResponse
from acsi.judge.clients import FakeJudge
from acsi.judge.ensemble import aggregate_pair_outcomes
from acsi.judge.runner import (
    CandidatePair,
    JudgeInterrupted,
    JudgeRunConfig,
    run_classifier_assertion,
    run_pairwise_judging,
    select_for_judging,
    write_judge_artifacts,
)
from acsi.replay.store import ReplayStore


def test_selection_returns_det_unequal_below_tau() -> None:
    pairs = [
        _pair("equal", deterministic_equal=True, similarity=0.1),
        _pair("within", deterministic_equal=False, similarity=0.95),
        _pair("beyond", deterministic_equal=False, similarity=0.72),
    ]

    assert [pair.pair_id for pair in select_for_judging(pairs, 0.9)] == ["beyond"]


def test_oracle_fake_judges_reach_high_ensemble_accuracy(tmp_path: Path) -> None:
    oracle = _oracle(100)
    pairs = [_pair(f"p{i}") for i in range(100)]
    result = run_pairwise_judging(
        pairs,
        {
            "openai/fake-a": FakeJudge(seed=1, oracle=oracle, error_rate=0.05),
            "google/fake-b": FakeJudge(seed=2, oracle=oracle, error_rate=0.05),
        },
        store=ReplayStore(tmp_path / "judge.sqlite"),
        config=JudgeRunConfig(run_id="run-1"),
    )
    outcomes = aggregate_pair_outcomes(_votes(result.judgments))

    assert _accuracy(outcomes, oracle) >= 0.9
    assert result.stats["ensemble"]["krippendorff_alpha"] is not None


def test_positional_bias_inconsistency_is_reported_and_ensemble_stays_accurate(
    tmp_path: Path,
) -> None:
    oracle = _oracle(80)
    pairs = [_pair(f"p{i}") for i in range(80)]
    result = run_pairwise_judging(
        pairs,
        {
            "openai/biased": FakeJudge(seed=3, oracle=oracle, positional_bias=0.9),
            "google/honest": FakeJudge(seed=4, oracle=oracle),
        },
        store=ReplayStore(tmp_path / "judge.sqlite"),
        config=JudgeRunConfig(run_id="run-1"),
    )
    outcomes = aggregate_pair_outcomes(_votes(result.judgments))

    biased = result.stats["judges"]["openai/biased"]
    assert biased["position_inconsistency_rate"] > 0.5
    assert _accuracy(outcomes, oracle) >= 0.9


def test_parse_retry_succeeds_then_double_failure_abstains(tmp_path: Path) -> None:
    def oracle(_pair_id: str) -> str:
        return "equivalent"

    judge = FakeJudge(
        oracle=oracle,
        malformed_attempts={
            ("retry", "candidate_a", 0),
            ("retry", "candidate_b", 0),
            ("fail", "candidate_a", 0),
            ("fail", "candidate_a", 1),
            ("fail", "candidate_b", 0),
            ("fail", "candidate_b", 1),
        },
    )
    result = run_pairwise_judging(
        [_pair("retry"), _pair("fail")],
        {"openai/fake": judge},
        store=ReplayStore(tmp_path / "judge.sqlite"),
        config=JudgeRunConfig(run_id="run-1"),
    )
    outcomes = aggregate_pair_outcomes(_votes(result.judgments))

    assert outcomes["retry"] == "equivalent"
    assert outcomes["fail"] == "unresolved"
    assert result.stats["judges"]["openai/fake"]["parse_failures"] == 1


def test_classifier_assertion_majority_and_tie_policy(tmp_path: Path) -> None:
    passed = run_classifier_assertion(
        assertion_id="a1",
        pair_id="p1",
        prompt="prompt",
        response="response",
        criterion="must pass",
        judge_clients={
            "openai/a": FakeJudge(classifier_oracle=lambda _pair_id: True),
            "google/b": FakeJudge(classifier_oracle=lambda _pair_id: True),
        },
        store=ReplayStore(tmp_path / "pass.sqlite"),
        config=JudgeRunConfig(run_id="run-1"),
    )
    tied = run_classifier_assertion(
        assertion_id="a1",
        pair_id="p2",
        prompt="prompt",
        response="response",
        criterion="must pass",
        judge_clients={
            "openai/a": FakeJudge(classifier_oracle=lambda _pair_id: True),
            "google/b": FakeJudge(classifier_oracle=lambda _pair_id: False),
        },
        store=ReplayStore(tmp_path / "tie.sqlite"),
        config=JudgeRunConfig(run_id="run-1"),
    )

    assert passed.passed
    assert not passed.queued_for_review
    assert not tied.passed
    assert tied.queued_for_review


def test_checkpoint_resume_matches_uninterrupted_artifacts(tmp_path: Path) -> None:
    oracle = _oracle(6)
    pairs = [_pair(f"p{i}") for i in range(6)]
    judges = {
        "openai/a": FakeJudge(seed=1, oracle=oracle),
        "google/b": FakeJudge(seed=2, oracle=oracle),
    }
    control = run_pairwise_judging(
        pairs,
        judges,
        store=ReplayStore(tmp_path / "control.sqlite"),
        config=JudgeRunConfig(run_id="run-1"),
    )
    write_judge_artifacts(tmp_path / "control", control)

    interrupted_store = ReplayStore(tmp_path / "resume.sqlite")
    with pytest.raises(JudgeInterrupted):
        run_pairwise_judging(
            pairs,
            {
                "openai/a": FakeJudge(seed=1, oracle=oracle),
                "google/b": FakeJudge(seed=2, oracle=oracle),
            },
            store=interrupted_store,
            config=JudgeRunConfig(run_id="run-1", interrupt_after_dispatches=12),
        )

    resumed = run_pairwise_judging(
        pairs,
        {
            "openai/a": FakeJudge(seed=1, oracle=oracle),
            "google/b": FakeJudge(seed=2, oracle=oracle),
        },
        store=interrupted_store,
        config=JudgeRunConfig(run_id="run-1"),
    )
    write_judge_artifacts(tmp_path / "resume", resumed)

    assert resumed.cache_hits == 12
    assert resumed.dispatched == 12
    assert (tmp_path / "resume" / "judgments.jsonl").read_bytes() == (
        tmp_path / "control" / "judgments.jsonl"
    ).read_bytes()


def _pair(
    pair_id: str,
    *,
    deterministic_equal: bool = False,
    similarity: float = 0.5,
) -> CandidatePair:
    return CandidatePair(
        pair_id=pair_id,
        trace_id=pair_id,
        prompt=f"Prompt {pair_id}",
        baseline=DiffResponse(text=f"old {pair_id}"),
        candidate=DiffResponse(text=f"new {pair_id}"),
        deterministic_equal=deterministic_equal,
        similarity=similarity,
    )


def _oracle(count: int):
    labels = ["equivalent", "candidate_better", "worse_minor", "worse_critical"]
    return lambda pair_id: labels[int(pair_id.removeprefix("p")) % len(labels)]


def _votes(rows: list[dict[str, object]]):
    votes: dict[str, dict[str, object]] = {}
    for row in rows:
        votes.setdefault(str(row["pair_id"]), {})[str(row["judge"])] = row["outcome"]
    return votes


def _accuracy(outcomes, oracle) -> float:
    correct = 0
    for pair_id, outcome in outcomes.items():
        correct += int(outcome == oracle(pair_id))
    return correct / len(outcomes)
