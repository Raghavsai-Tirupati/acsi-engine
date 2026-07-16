from __future__ import annotations

import json
from pathlib import Path

import pytest

from acsi.schemas import TraceRecord, export_json_schemas
from scripts.generate_fixtures import generate_records


def test_trace_record_accepts_fixture_record() -> None:
    record = TraceRecord.model_validate(generate_records(count=1)[0])

    assert record.request.messages[0].role == "user"
    assert record.response.served_model == "claude-haiku-4-5-20251001"


def test_trace_record_rejects_multi_turn() -> None:
    payload = generate_records(count=1)[0]
    payload["request"]["messages"].append({"role": "assistant", "content": "extra turn"})

    with pytest.raises(ValueError, match="exactly one user message"):
        TraceRecord.model_validate(payload)


def test_schema_export_writes_frozen_contracts(tmp_path: Path) -> None:
    written = export_json_schemas(tmp_path)

    assert {path.name for path in written} == {
        "trace-record.schema.json",
        "workload-manifest.schema.json",
        "run-manifest.schema.json",
        "certificate.schema.json",
    }
    trace_schema = json.loads((tmp_path / "trace-record.schema.json").read_text(encoding="utf-8"))
    assert trace_schema["title"] == "TraceRecord"

