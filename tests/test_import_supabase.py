from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, uuid5

import httpx
import pytest
from typer.testing import CliRunner

from acsi.cli import app
from acsi.importers.supabase import (
    SupabaseConfig,
    SupabaseImportError,
    import_supabase_records,
)
from scripts.generate_fixtures import generate_records

WORKLOAD = "support-ticket-summary"


def test_supabase_import_paginates_with_mock_transport() -> None:
    rows = [_supabase_row(index) for index in range(1_001)]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/rest/v1/acsi_traces"
        assert request.headers["apikey"] == "service-key"
        assert request.headers["authorization"] == "Bearer service-key"
        assert request.url.params["workload"] == f"eq.{WORKLOAD}"
        assert request.url.params["ts"] == "gte.2026-07-15T18:22:03Z"

        start, end = [int(value) for value in request.headers["range"].split("-")]
        return httpx.Response(200, json=rows[start : min(end + 1, len(rows))])

    transport = httpx.MockTransport(handler)
    config = SupabaseConfig(url="https://example.supabase.co", service_role_key="service-key")
    with httpx.Client(transport=transport) as client:
        result = import_supabase_records(
            config,
            workload=WORKLOAD,
            since="2026-07-15T18:22:03Z",
            client=client,
        )

    assert result.summary.lines_read == 1_001
    assert result.summary.valid == 1_001
    assert result.summary.malformed == 0
    assert result.summary.source_counts == {"supabase": 1_001}
    assert [request.headers["range"] for request in requests] == ["0-999", "1000-1999"]


def test_supabase_missing_env_is_actionable() -> None:
    with pytest.raises(SupabaseImportError, match="Set SUPABASE_URL"):
        SupabaseConfig.from_env({})


def test_supabase_cli_missing_env_returns_json_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    result = CliRunner().invoke(app, ["import", "supabase", "--workload", WORKLOAD, "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "error"
    assert "SUPABASE_URL" in payload["message"]


def _supabase_row(index: int) -> dict[str, object]:
    record = generate_records(count=1, seed=42)[0]
    ts = datetime(2026, 7, 15, 18, 22, 3, tzinfo=UTC) + timedelta(seconds=index)
    return {
        "id": str(uuid5(NAMESPACE_URL, f"acsi-supabase-test-{index}")),
        "ts": ts.isoformat().replace("+00:00", "Z"),
        "workload": WORKLOAD,
        "request": record["request"],
        "response": record["response"],
        "meta": record["meta"],
    }
