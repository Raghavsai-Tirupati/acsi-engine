from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from acsi.importers.jsonl import import_jsonl_paths
from acsi.replay.clients import FakeClient
from acsi.replay.runner import (
    ReplayAbortError,
    ReplayConfig,
    ReplayInterrupted,
    replay,
    write_responses_jsonl,
)
from acsi.replay.store import ReplayStore
from acsi.schemas import ProviderModel, TraceRecord


def test_replay_resume_uses_cache_and_matches_control_hash(tmp_path: Path) -> None:
    traces = _fixture_traces(8)
    model = ProviderModel(provider="anthropic", model="claude-sonnet-5")

    control_store = ReplayStore(tmp_path / "control.sqlite")
    control_client = FakeClient(seed=123, noise=0.25)
    asyncio.run(
        replay(
            traces,
            model,
            1,
            client=control_client,
            store=control_store,
            config=ReplayConfig(run_id="same-run", seed=123, concurrency=2),
        )
    )
    control_hash = write_responses_jsonl(control_store, "same-run", tmp_path / "control.jsonl")

    resume_store = ReplayStore(tmp_path / "resume.sqlite")
    interrupted_client = FakeClient(seed=123, noise=0.25)
    with pytest.raises(ReplayInterrupted):
        asyncio.run(
            replay(
                traces,
                model,
                1,
                client=interrupted_client,
                store=resume_store,
                config=ReplayConfig(
                    run_id="same-run",
                    seed=123,
                    concurrency=1,
                    interrupt_after_dispatches=4,
                    resume_command="acsi replay --run-id same-run",
                ),
            )
        )

    resume_client = FakeClient(seed=123, noise=0.25)
    resumed = asyncio.run(
        replay(
            traces,
            model,
            1,
            client=resume_client,
            store=resume_store,
            config=ReplayConfig(run_id="same-run", seed=123, concurrency=2),
        )
    )
    resumed_hash = write_responses_jsonl(resume_store, "same-run", tmp_path / "resume.jsonl")

    assert resumed.cache_hits == 4
    assert resume_client.call_count == len(traces) - 4
    assert resumed_hash == control_hash


def test_replay_strips_sonnet_sampling_params_before_dispatch(tmp_path: Path) -> None:
    traces = _fixture_traces(1)
    model = ProviderModel(provider="anthropic", model="claude-sonnet-5")
    client = RecordingFakeClient(seed=1)

    result = asyncio.run(
        replay(
            traces,
            model,
            1,
            client=client,
            store=ReplayStore(tmp_path / "params.sqlite"),
            config=ReplayConfig(run_id="params-run", seed=1),
        )
    )

    assert "temperature" not in client.seen_params[0]
    assert result.param_transforms[0].path == "params.temperature"
    assert "HTTP 400" in result.param_transforms[0].reason


def test_replay_budget_halts_with_resumable_checkpoint(tmp_path: Path) -> None:
    traces = _fixture_traces(6)
    model = ProviderModel(provider="anthropic", model="claude-sonnet-5")
    store = ReplayStore(tmp_path / "budget.sqlite")

    halted = asyncio.run(
        replay(
            traces,
            model,
            1,
            client=FakeClient(seed=5),
            store=store,
            config=ReplayConfig(
                run_id="budget-run",
                seed=5,
                concurrency=1,
                max_cost_usd=0.000002,
            ),
        )
    )

    assert halted.halted_reason and "--max-cost" in halted.halted_reason
    assert 0 < halted.completed < len(traces)
    assert store.total_cost("budget-run") <= 0.000002

    resumed = asyncio.run(
        replay(
            traces,
            model,
            1,
            client=FakeClient(seed=5),
            store=store,
            config=ReplayConfig(run_id="budget-run", seed=5, concurrency=1),
        )
    )
    assert resumed.cache_hits == halted.completed
    assert resumed.completed == len(traces)


def test_replay_retries_rate_limits_without_duplicate_rows(tmp_path: Path) -> None:
    traces = _fixture_traces(5)
    model = ProviderModel(provider="anthropic", model="claude-sonnet-5")
    store = ReplayStore(tmp_path / "retry.sqlite")

    result = asyncio.run(
        replay(
            traces,
            model,
            1,
            client=FakeClient(seed=9, fail_rate_limit_every=3),
            store=store,
            config=ReplayConfig(run_id="retry-run", seed=9, concurrency=1, base_backoff_seconds=0),
        )
    )

    assert result.retry_count > 0
    assert store.status_counts("retry-run") == {"done": len(traces)}
    assert len(store.done_calls("retry-run")) == len(traces)


def test_replay_model_404_aborts_with_retired_model_message(tmp_path: Path) -> None:
    traces = _fixture_traces(1)
    model = ProviderModel(provider="anthropic", model="retired-model")

    with pytest.raises(ReplayAbortError) as exc_info:
        asyncio.run(
            replay(
                traces,
                model,
                1,
                client=FakeClient(retired_models={"retired-model"}),
                store=ReplayStore(tmp_path / "abort.sqlite"),
                config=ReplayConfig(run_id="abort-run"),
            )
        )
    assert str(exc_info.value) == (
        "Model retired-model returned 404; it may be retired. "
        "Rerun with --degraded to certify against stored outputs."
    )


class RecordingFakeClient(FakeClient):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seen_params: list[dict[str, object]] = []

    def complete(self, request):
        self.seen_params.append(dict(request.params))
        return super().complete(request)


def _fixture_traces(count: int) -> list[TraceRecord]:
    fixture = Path(__file__).parent / "fixtures" / "synthetic_traces.jsonl"
    return import_jsonl_paths([fixture]).records[:count]
