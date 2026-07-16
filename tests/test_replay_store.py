from __future__ import annotations

from pathlib import Path

from acsi.replay.clients import CompletionResponse
from acsi.replay.store import ReplayStore


def test_replay_store_persists_done_rows_and_closes_connections(tmp_path: Path) -> None:
    db_path = tmp_path / "replay.sqlite"
    store = ReplayStore(db_path)
    store.initialize()
    store.write_done(
        run_id="run-1",
        trace_id="trace-1",
        sample_index=0,
        model="claude-sonnet-5",
        params_hash="params",
        prompt_hash="prompt",
        response=CompletionResponse(
            text="ok",
            tool_calls=None,
            finish_reason="stop",
            usage={"input_tokens": 4, "output_tokens": 2},
            latency_ms=123,
            served_model="claude-sonnet-5",
        ),
        cost_usd=0.01,
        retry_count=1,
    )

    reopened = ReplayStore(db_path)
    cached = reopened.get_done("run-1", "trace-1", 0)

    assert cached is not None
    assert cached.response and cached.response["text"] == "ok"
    assert cached.usage == {"input_tokens": 4, "output_tokens": 2}
    assert cached.retry_count == 1
    assert reopened.total_cost("run-1") == 0.01
    db_path.unlink()
    assert not db_path.exists()


def test_replay_store_records_trace_level_errors(tmp_path: Path) -> None:
    db_path = tmp_path / "replay.sqlite"
    store = ReplayStore(db_path)
    store.initialize()

    store.write_error(
        run_id="run-1",
        trace_id="trace-1",
        sample_index=0,
        model="candidate",
        params_hash="params",
        prompt_hash="prompt",
        error="content rejected",
        retry_count=0,
    )

    assert store.status_counts("run-1") == {"error": 1}
    assert store.done_calls("run-1") == []


def test_replay_store_isolates_phases(tmp_path: Path) -> None:
    db_path = tmp_path / "replay.sqlite"
    store = ReplayStore(db_path)
    store.initialize()

    response = CompletionResponse(
        text="ok",
        tool_calls=None,
        finish_reason="stop",
        usage={"input_tokens": 1, "output_tokens": 1},
        latency_ms=5,
        served_model="judge-model",
    )
    for phase, cost in (("replay", 0.01), ("judge", 0.02)):
        store.write_done(
            run_id="run-1",
            trace_id="same-id",
            sample_index=0,
            model="model",
            params_hash="params",
            prompt_hash="prompt",
            response=response,
            cost_usd=cost,
            retry_count=0,
            phase=phase,
        )

    assert store.get_done("run-1", "same-id", 0).phase == "replay"
    assert store.get_done("run-1", "same-id", 0, phase="judge").phase == "judge"
    assert store.total_cost("run-1") == 0.01
    assert store.total_cost("run-1", phase="judge") == 0.02
