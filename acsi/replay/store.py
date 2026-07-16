from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acsi.replay.clients import CompletionResponse


@dataclass(frozen=True)
class StoredCall:
    run_id: str
    trace_id: str
    sample_index: int
    model: str
    params_hash: str
    prompt_hash: str
    status: str
    response: dict[str, Any] | None
    usage: dict[str, int]
    cost_usd: float
    served_model: str | None
    error: str | None
    retry_count: int


class ReplayStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                create table if not exists calls (
                    run_id text not null,
                    trace_id text not null,
                    sample_index integer not null,
                    model text not null,
                    params_hash text not null,
                    prompt_hash text not null,
                    status text not null,
                    response_json text,
                    usage_json text not null,
                    cost_usd real not null,
                    served_model text,
                    error text,
                    retry_count integer not null default 0,
                    updated_at text not null default current_timestamp,
                    primary key (run_id, trace_id, sample_index)
                )
                """
            )

    def get_done(self, run_id: str, trace_id: str, sample_index: int) -> StoredCall | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                """
                select run_id, trace_id, sample_index, model, params_hash, prompt_hash, status,
                       response_json, usage_json, cost_usd, served_model, error, retry_count
                from calls
                where run_id = ? and trace_id = ? and sample_index = ? and status = 'done'
                """,
                (run_id, trace_id, sample_index),
            ).fetchone()
        return _stored_call_from_row(row) if row else None

    def write_done(
        self,
        *,
        run_id: str,
        trace_id: str,
        sample_index: int,
        model: str,
        params_hash: str,
        prompt_hash: str,
        response: CompletionResponse,
        cost_usd: float,
        retry_count: int,
    ) -> None:
        response_json = _json_dumps(
            {
                "text": response.text,
                "tool_calls": response.tool_calls,
                "finish_reason": response.finish_reason,
                "latency_ms": response.latency_ms,
                "served_model": response.served_model,
            }
        )
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                insert or replace into calls (
                    run_id, trace_id, sample_index, model, params_hash, prompt_hash, status,
                    response_json, usage_json, cost_usd, served_model, error, retry_count,
                    updated_at
                )
                values (?, ?, ?, ?, ?, ?, 'done', ?, ?, ?, ?, null, ?, current_timestamp)
                """,
                (
                    run_id,
                    trace_id,
                    sample_index,
                    model,
                    params_hash,
                    prompt_hash,
                    response_json,
                    _json_dumps(response.usage),
                    cost_usd,
                    response.served_model,
                    retry_count,
                ),
            )

    def write_error(
        self,
        *,
        run_id: str,
        trace_id: str,
        sample_index: int,
        model: str,
        params_hash: str,
        prompt_hash: str,
        error: str,
        retry_count: int,
    ) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                insert or replace into calls (
                    run_id, trace_id, sample_index, model, params_hash, prompt_hash, status,
                    response_json, usage_json, cost_usd, served_model, error, retry_count,
                    updated_at
                )
                values (?, ?, ?, ?, ?, ?, 'error', null, '{}', 0.0, null, ?, ?, current_timestamp)
                """,
                (
                    run_id,
                    trace_id,
                    sample_index,
                    model,
                    params_hash,
                    prompt_hash,
                    error,
                    retry_count,
                ),
            )

    def done_calls(self, run_id: str) -> list[StoredCall]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """
                select run_id, trace_id, sample_index, model, params_hash, prompt_hash, status,
                       response_json, usage_json, cost_usd, served_model, error, retry_count
                from calls
                where run_id = ? and status = 'done'
                order by trace_id, sample_index
                """,
                (run_id,),
            ).fetchall()
        return [_stored_call_from_row(row) for row in rows]

    def status_counts(self, run_id: str) -> dict[str, int]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "select status, count(*) from calls where run_id = ? group by status",
                (run_id,),
            ).fetchall()
        return {str(status): int(count) for status, count in rows}

    def total_cost(self, run_id: str) -> float:
        with sqlite3.connect(self.path) as conn:
            value = conn.execute(
                """
                select coalesce(sum(cost_usd), 0.0)
                from calls
                where run_id = ? and status = 'done'
                """,
                (run_id,),
            ).fetchone()[0]
        return float(value)

    def served_models(self, run_id: str) -> list[str]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """
                select distinct served_model
                from calls
                where run_id = ? and status = 'done' and served_model is not null
                order by served_model
                """,
                (run_id,),
            ).fetchall()
        return [str(row[0]) for row in rows]


def connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path)


def _stored_call_from_row(row: tuple[Any, ...]) -> StoredCall:
    response_json = row[7]
    usage_json = row[8]
    return StoredCall(
        run_id=str(row[0]),
        trace_id=str(row[1]),
        sample_index=int(row[2]),
        model=str(row[3]),
        params_hash=str(row[4]),
        prompt_hash=str(row[5]),
        status=str(row[6]),
        response=json.loads(response_json) if response_json else None,
        usage=json.loads(usage_json) if usage_json else {},
        cost_usd=float(row[9]),
        served_model=str(row[10]) if row[10] is not None else None,
        error=str(row[11]) if row[11] is not None else None,
        retry_count=int(row[12]),
    )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
