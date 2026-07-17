from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from acsi.importers.common import ImportAccumulator, ImportResult

PAGE_SIZE = 1_000
# SPEC-NOTE: HTTP clients touching third-party APIs must handle redirects
# deliberately. Supabase/PostgREST can 3xx (e.g. a project URL that migrated, or
# http->https), so we follow up to MAX_REDIRECTS rather than rely on httpx's
# no-follow default, and surface an over-limit case as an actionable error.
MAX_REDIRECTS = 5


class SupabaseImportError(RuntimeError):
    pass


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    service_role_key: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SupabaseConfig:
        env = os.environ if environ is None else environ
        url = env.get("SUPABASE_URL")
        key = env.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise SupabaseImportError(
                "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY before importing Supabase traces."
            )
        return cls(url=url.rstrip("/"), service_role_key=key)


def import_supabase_records(
    config: SupabaseConfig,
    workload: str,
    since: str | None = None,
    client: httpx.Client | None = None,
) -> ImportResult:
    accumulator = ImportAccumulator()
    owns_client = client is None
    active_client = client or httpx.Client(
        timeout=30.0, follow_redirects=True, max_redirects=MAX_REDIRECTS
    )
    try:
        rows = _fetch_rows(active_client, config, workload=workload, since=since)
        accumulator.extend_payloads((_row_to_trace_payload(row) for row in rows), source="supabase")
    finally:
        if owns_client:
            active_client.close()
    return accumulator.result


def _fetch_rows(
    client: httpx.Client,
    config: SupabaseConfig,
    workload: str,
    since: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = _fetch_page(client, config, workload=workload, since=since, offset=offset)
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            return rows
        offset += PAGE_SIZE


def _fetch_page(
    client: httpx.Client,
    config: SupabaseConfig,
    workload: str,
    since: str | None,
    offset: int,
) -> list[dict[str, Any]]:
    params = {
        "select": "id,ts,workload,request,response,meta",
        "workload": f"eq.{workload}",
        "order": "ts.asc",
    }
    if since:
        params["ts"] = f"gte.{since}"

    try:
        response = client.get(
            f"{config.url}/rest/v1/acsi_traces",
            params=params,
            headers={
                "apikey": config.service_role_key,
                "Authorization": f"Bearer {config.service_role_key}",
                "Range-Unit": "items",
                "Range": f"{offset}-{offset + PAGE_SIZE - 1}",
            },
        )
    except httpx.TooManyRedirects as exc:
        raise SupabaseImportError(
            f"Supabase import exceeded the redirect limit (max {MAX_REDIRECTS}); "
            "verify SUPABASE_URL points at the current project endpoint."
        ) from exc
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise SupabaseImportError(
            f"Supabase import failed with HTTP {response.status_code}: {response.text}"
        ) from exc

    payload = response.json()
    if not isinstance(payload, list):
        raise SupabaseImportError("Supabase import expected a JSON array from PostgREST.")
    return payload


def _row_to_trace_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_id": row.get("id"),
        "ts": row.get("ts"),
        "source": "supabase",
        "workload": row.get("workload"),
        "request": row.get("request"),
        "response": row.get("response"),
        "meta": row.get("meta") or {},
    }
