from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from acsi.replay.clients import CompletionResponse


@dataclass(frozen=True)
class StoredCall:
    phase: str
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
        with _connect(self.path) as conn:
            conn.execute(
                """
                create table if not exists calls (
                    phase text not null default 'replay',
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
                    primary key (phase, run_id, trace_id, sample_index)
                )
                """
            )
            _migrate_phase_column(conn)

    def get_done(
        self,
        run_id: str,
        trace_id: str,
        sample_index: int,
        *,
        phase: str = "replay",
    ) -> StoredCall | None:
        with _connect(self.path) as conn:
            row = conn.execute(
                """
                select phase, run_id, trace_id, sample_index, model, params_hash, prompt_hash,
                       status, response_json, usage_json, cost_usd, served_model, error,
                       retry_count
                from calls
                where phase = ? and run_id = ? and trace_id = ? and sample_index = ?
                  and status = 'done'
                """,
                (phase, run_id, trace_id, sample_index),
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
        phase: str = "replay",
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
        with _connect(self.path) as conn:
            conn.execute(
                """
                insert or replace into calls (
                    phase, run_id, trace_id, sample_index, model, params_hash, prompt_hash,
                    status, response_json, usage_json, cost_usd, served_model, error, retry_count,
                    updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, 'done', ?, ?, ?, ?, null, ?, current_timestamp)
                """,
                (
                    phase,
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
        phase: str = "replay",
    ) -> None:
        with _connect(self.path) as conn:
            conn.execute(
                """
                insert or replace into calls (
                    phase, run_id, trace_id, sample_index, model, params_hash, prompt_hash,
                    status, response_json, usage_json, cost_usd, served_model, error, retry_count,
                    updated_at
                )
                values (?, ?, ?, ?, ?, ?, ?, 'error', null, '{}', 0.0, null, ?, ?,
                        current_timestamp)
                """,
                (
                    phase,
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

    def done_calls(self, run_id: str, *, phase: str = "replay") -> list[StoredCall]:
        with _connect(self.path) as conn:
            rows = conn.execute(
                """
                select phase, run_id, trace_id, sample_index, model, params_hash, prompt_hash,
                       status, response_json, usage_json, cost_usd, served_model, error,
                       retry_count
                from calls
                where phase = ? and run_id = ? and status = 'done'
                order by trace_id, sample_index
                """,
                (phase, run_id),
            ).fetchall()
        return [_stored_call_from_row(row) for row in rows]

    def status_counts(self, run_id: str, *, phase: str = "replay") -> dict[str, int]:
        with _connect(self.path) as conn:
            rows = conn.execute(
                """
                select status, count(*)
                from calls
                where phase = ? and run_id = ?
                group by status
                """,
                (phase, run_id),
            ).fetchall()
        return {str(status): int(count) for status, count in rows}

    def total_cost(self, run_id: str, *, phase: str = "replay") -> float:
        with _connect(self.path) as conn:
            value = conn.execute(
                """
                select coalesce(sum(cost_usd), 0.0)
                from calls
                where phase = ? and run_id = ? and status = 'done'
                """,
                (phase, run_id),
            ).fetchone()[0]
        return float(value)

    def served_models(self, run_id: str, *, phase: str = "replay") -> list[str]:
        with _connect(self.path) as conn:
            rows = conn.execute(
                """
                select distinct served_model
                from calls
                where phase = ? and run_id = ? and status = 'done' and served_model is not null
                order by served_model
                """,
                (phase, run_id),
            ).fetchall()
        return [str(row[0]) for row in rows]


def connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path)


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _stored_call_from_row(row: tuple[Any, ...]) -> StoredCall:
    response_json = row[8]
    usage_json = row[9]
    return StoredCall(
        phase=str(row[0]),
        run_id=str(row[1]),
        trace_id=str(row[2]),
        sample_index=int(row[3]),
        model=str(row[4]),
        params_hash=str(row[5]),
        prompt_hash=str(row[6]),
        status=str(row[7]),
        response=json.loads(response_json) if response_json else None,
        usage=json.loads(usage_json) if usage_json else {},
        cost_usd=float(row[10]),
        served_model=str(row[11]) if row[11] is not None else None,
        error=str(row[12]) if row[12] is not None else None,
        retry_count=int(row[13]),
    )


def _migrate_phase_column(conn: sqlite3.Connection) -> None:
    columns = {str(row[1]) for row in conn.execute("pragma table_info(calls)").fetchall()}
    if "phase" in columns:
        return
    conn.execute("alter table calls rename to calls_legacy")
    conn.execute(
        """
        create table calls (
            phase text not null default 'replay',
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
            primary key (phase, run_id, trace_id, sample_index)
        )
        """
    )
    conn.execute(
        """
        insert into calls (
            phase, run_id, trace_id, sample_index, model, params_hash, prompt_hash, status,
            response_json, usage_json, cost_usd, served_model, error, retry_count, updated_at
        )
        select 'replay', run_id, trace_id, sample_index, model, params_hash, prompt_hash,
               status, response_json, usage_json, cost_usd, served_model, error, retry_count,
               updated_at
        from calls_legacy
        """
    )
    conn.execute("drop table calls_legacy")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
