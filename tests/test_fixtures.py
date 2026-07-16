from __future__ import annotations

import json

from scripts.generate_fixtures import generate_records


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

