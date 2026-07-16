from __future__ import annotations

import os

import pytest

from acsi.judge.clients import (
    FakeJudge,
    JudgeConfigurationError,
    LiveJudge,
    judge_family,
    select_judge_panel,
)
from acsi.judge.rubric import parse_pairwise_judgment
from acsi.replay.clients import CompletionRequest
from acsi.schemas import WorkloadManifest


def test_family_exclusion_builds_panel_without_baseline_candidate_family() -> None:
    manifest = _manifest(
        judges=[
            {"model": "anthropic/claude-judge"},
            {"model": "openai/gpt-judge"},
            {"model": "google/gemini-judge"},
            {"model": "local/llama-judge"},
        ]
    )

    panel = select_judge_panel(manifest)

    assert [judge.family for judge in panel] == ["openai", "google", "local"]
    assert "anthropic" not in {judge.family for judge in panel}


def test_family_exclusion_aborts_with_actionable_error() -> None:
    manifest = _manifest(judges=[{"model": "anthropic/claude-judge"}])

    with pytest.raises(JudgeConfigurationError, match="Excluded families: anthropic"):
        select_judge_panel(manifest)


def test_missing_explicit_judges_abort() -> None:
    manifest = _manifest(judges=None)

    with pytest.raises(JudgeConfigurationError, match="judging.judges is required"):
        select_judge_panel(manifest)


def test_fake_judge_uses_oracle_and_position_mapping() -> None:
    judge = FakeJudge(oracle=lambda _pair_id: "worse_critical")

    response = judge.complete(_request(pair_id="p1", ordering="candidate_b"))
    judgment = parse_pairwise_judgment(response.text)

    assert judgment.verdict == "a_better"
    assert judgment.severity_if_worse == "critical"


def test_fake_judge_can_emit_malformed_configured_calls() -> None:
    judge = FakeJudge(malformed_calls={1})

    assert judge.complete(_request()).text == "{malformed"
    assert parse_pairwise_judgment(judge.complete(_request()).text).verdict == "equivalent"


def test_judge_family_uses_model_prefix() -> None:
    assert judge_family("openai/gpt-4o-mini") == "openai"
    assert judge_family("local:llama3") == "local"


@pytest.mark.skipif(
    os.environ.get("ACSI_TEST_LIVE_JUDGE") != "1",
    reason="set ACSI_TEST_LIVE_JUDGE=1 and provider keys/env to smoke-test LiveJudge",
)
def test_live_judge_smoke() -> None:
    response = LiveJudge(os.environ.get("ACSI_TEST_LIVE_JUDGE_MODEL", "local/test")).complete(
        _request()
    )

    assert response.finish_reason


def _request(pair_id: str = "p1", ordering: str = "candidate_a") -> CompletionRequest:
    return CompletionRequest(
        provider="judge",
        model="fake",
        system=None,
        messages=[{"role": "user", "content": "judge this"}],
        params={
            "attempt": 0,
            "mode": "pairwise",
            "ordering": ordering,
            "pair_id": pair_id,
        },
        sample_index=0,
    )


def _manifest(judges: list[dict[str, str]] | None) -> WorkloadManifest:
    payload = {
        "assertions": [],
        "baseline": {"provider": "anthropic", "model": "claude-old"},
        "budget": {"max_usd": 1.0, "use_batch_api": False},
        "candidate": {"provider": "anthropic", "model": "claude-new"},
        "judging": {
            "families_allowed": ["openai", "google", "local"],
            "judges": judges,
            "min_judges": 2,
        },
        "privacy": {"egress": "hosted_api", "scrub": True},
        "sampling": {"k_baseline": 2, "n": 10, "seed": 42, "stratify_by": []},
        "thresholds": {"confidence": 0.95, "epsilon_pp": 2.0, "max_critical": 0},
        "workload": "demo",
    }
    if judges is None:
        del payload["judging"]["judges"]
    return WorkloadManifest.model_validate(payload)
