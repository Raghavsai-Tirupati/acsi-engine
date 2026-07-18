from __future__ import annotations

import json
from pathlib import Path

from acsi.diff.deterministic import DiffResponse
from acsi.judge.clients import JudgeSpec, LiveJudge
from acsi.judge.ensemble import aggregate_pair_outcomes, majority_outcome
from acsi.judge.runner import CandidatePair, JudgeRunConfig, run_pairwise_judging
from acsi.replay.clients import (
    CompletionRequest,
    CompletionResponse,
    PermanentError,
    RateLimitError,
    TransientError,
    map_litellm_error,
)
from acsi.replay.store import ReplayStore

VERDICT = json.dumps(
    {"reason": "ok", "severity_if_worse": None, "verdict": "equivalent"},
    sort_keys=True,
    separators=(",", ":"),
)


class ScriptedJudge:
    """Raise the queued exceptions on successive calls, then return a valid verdict."""

    def __init__(self, errors: list[Exception], *, model: str = "openai/scripted") -> None:
        self.errors = list(errors)
        self.model = model
        self.calls = 0

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return CompletionResponse(
            text=VERDICT,
            tool_calls=None,
            finish_reason="stop",
            usage={"input_tokens": 1, "output_tokens": 1},
            latency_ms=1,
            served_model=self.model,
        )


def _pair(pair_id: str) -> CandidatePair:
    return CandidatePair(
        pair_id=pair_id,
        trace_id=pair_id,
        prompt=f"Prompt {pair_id}",
        baseline=DiffResponse(text=f"old {pair_id}"),
        candidate=DiffResponse(text=f"new {pair_id}"),
        deterministic_equal=False,
        similarity=0.5,
    )


def _config(tmp_path: Path, **kwargs) -> tuple[JudgeRunConfig, list[float]]:
    sleeps: list[float] = []
    config = JudgeRunConfig(
        run_id="run-1",
        base_backoff_s=0.001,
        sleep=sleeps.append,
        **kwargs,
    )
    return config, sleeps


def _outcomes(result) -> dict[str, str]:
    votes: dict[str, dict[str, object]] = {}
    for row in result.judgments:
        votes.setdefault(str(row["pair_id"]), {})[str(row["judge"])] = row["outcome"]
    return aggregate_pair_outcomes(votes)


def test_transient_errors_are_retried_then_succeed(tmp_path: Path) -> None:
    config, sleeps = _config(tmp_path)
    judge = ScriptedJudge([TransientError("503"), RateLimitError("429")])

    result = run_pairwise_judging(
        [_pair("p")],
        {"openai/scripted": judge},
        store=ReplayStore(tmp_path / "j.sqlite"),
        config=config,
    )

    assert _outcomes(result)["p"] == "equivalent"
    assert len(sleeps) == 2  # two transient failures backed off before success


def test_non_transient_error_fails_fast_without_retry(tmp_path: Path) -> None:
    config, sleeps = _config(tmp_path)
    judge = ScriptedJudge([PermanentError("bad request", status_code=400)])

    result = run_pairwise_judging(
        [_pair("p")],
        {"openai/scripted": judge},
        store=ReplayStore(tmp_path / "j.sqlite"),
        config=config,
    )

    assert sleeps == []  # no backoff for a non-retryable error
    assert _outcomes(result)["p"] == "unresolved"
    row = next(r for r in result.judgments if r["pair_id"] == "p")
    assert row["abstain_reason"] == "judge_error"
    assert row["error"]


def test_exhausted_retries_record_judge_error_and_run_continues(tmp_path: Path) -> None:
    config, _ = _config(tmp_path, max_attempts=4)
    always_fail = ScriptedJudge([TransientError("503")] * 100)

    result = run_pairwise_judging(
        [_pair("a"), _pair("b")],
        {"openai/scripted": always_fail},
        store=ReplayStore(tmp_path / "j.sqlite"),
        config=config,
    )

    # The run did not raise; both pairs abstained with judge_error.
    assert {r["pair_id"] for r in result.judgments} == {"a", "b"}
    assert all(r["abstain_reason"] == "judge_error" for r in result.judgments)
    assert result.stats["judges"]["openai/scripted"]["call_errors"] == 2


def test_pair_below_min_judges_is_unresolved(tmp_path: Path) -> None:
    config, _ = _config(tmp_path, max_attempts=2)
    healthy = ScriptedJudge([], model="openai/healthy")
    broken = ScriptedJudge([TransientError("503")] * 100, model="google/broken")

    result = run_pairwise_judging(
        [_pair("p")],
        {"openai/healthy": healthy, "google/broken": broken},
        store=ReplayStore(tmp_path / "j.sqlite"),
        config=config,
    )
    votes: dict[str, dict[str, object]] = {}
    for row in result.judgments:
        votes.setdefault(str(row["pair_id"]), {})[str(row["judge"])] = row["outcome"]

    # One judge voted, the other errored: below a two-judge floor -> unresolved.
    assert aggregate_pair_outcomes(votes, min_valid=2)["p"] == "unresolved"
    # ...but at the default floor the single healthy verdict still counts.
    assert aggregate_pair_outcomes(votes, min_valid=1)["p"] == "equivalent"


def test_retry_after_hint_is_honored_and_capped(tmp_path: Path) -> None:
    config, sleeps = _config(tmp_path, max_attempts=4, max_retry_after_s=60.0)
    over_cap = TransientError("busy")
    over_cap.retry_after_s = 120.0
    under_cap = TransientError("busy")
    under_cap.retry_after_s = 5.0
    judge = ScriptedJudge([over_cap, under_cap])

    run_pairwise_judging(
        [_pair("p")],
        {"openai/scripted": judge},
        store=ReplayStore(tmp_path / "j.sqlite"),
        config=config,
    )

    assert sleeps == [60.0, 5.0]


def test_progress_emits_per_pair_lines_and_summary(tmp_path: Path) -> None:
    messages: list[str] = []
    config = JudgeRunConfig(run_id="run-1", progress=messages.append)
    judge = ScriptedJudge([])

    run_pairwise_judging(
        [_pair("p")],
        {"openai/scripted": judge},
        store=ReplayStore(tmp_path / "j.sqlite"),
        config=config,
    )

    assert any("judging pair 1/1 [openai/scripted] attempt 1" in m for m in messages)
    assert any(m.startswith("judged 1 pairs;") for m in messages)


def test_min_judges_floor_capped_by_present_votes() -> None:
    # Single present valid vote: floor capped at 1, so it counts.
    assert majority_outcome(["worse_critical"], min_valid=2) == "worse_critical"
    # Two present votes, one abstained: floor is 2, one valid -> unresolved.
    assert majority_outcome(["worse_critical", "unresolved"], min_valid=2) == "unresolved"


def test_live_judge_carries_timeout_and_error_mapping() -> None:
    spec = JudgeSpec(
        model="gpt-5.4-mini",
        family="openai",
        provider="openai",
        litellm_model="gpt-5.4-mini",
    )
    judge = LiveJudge.from_spec(spec, timeout_s=45.0)
    assert judge.timeout_s == 45.0

    class Timeout(Exception):
        pass

    assert isinstance(map_litellm_error(Timeout(), "m"), TransientError)

    rate = type("RateErr", (Exception,), {"status_code": 429})()
    mapped = map_litellm_error(rate, "m")
    assert isinstance(mapped, RateLimitError)

    bad = type("BadErr", (Exception,), {"status_code": 400})()
    assert isinstance(map_litellm_error(bad, "m"), PermanentError)
    assert map_litellm_error(bad, "m").retryable is False
