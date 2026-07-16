from __future__ import annotations

import json

from scripts.generate_fixtures import generate_invalid_lines, generate_records


def test_fixture_generator_creates_expected_distribution() -> None:
    records = generate_records(count=300, seed=42)

    assert len(records) == 300
    templated = [record for record in records if record["meta"]["template_id"]]
    assert len(templated) == 270
    assert records[0]["request"]["messages"][0]["role"] == "user"


def test_fixture_records_are_json_serializable() -> None:
    records = generate_records(count=3, seed=1)

    for record in records:
        json.dumps(record)


def test_invalid_fixture_shape_is_exact() -> None:
    records = generate_records(count=300, seed=42)
    lines = generate_invalid_lines(records, seed=42)

    assert len(lines) == 15
    assert sum(1 for line in lines if line.startswith("{not") or line.startswith("[malformed")) == 2

    parsed = [json.loads(line) for line in lines if line.startswith("{\"")]
    multi_turn = [record for record in parsed if len(record["request"]["messages"]) > 1]
    empty_jsonl = [
        record
        for record in parsed
        if record["source"] != "backfill" and record["response"] == {}
    ]
    backfills = [record for record in parsed if record["source"] == "backfill"]

    assert len(multi_turn) == 5
    assert len(empty_jsonl) == 2
    assert len(backfills) == 3
    assert [record["trace_id"] for record in parsed[5:8]] == [
        record["trace_id"] for record in records[:3]
    ]
