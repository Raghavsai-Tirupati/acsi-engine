from __future__ import annotations

import json
from pathlib import Path

from acsi.importers.jsonl import import_jsonl_paths
from acsi.scrub import regex_scrub, scrub_traces, write_scrub_artifacts

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_traces.jsonl"


def test_regex_scrub_uses_typed_stable_placeholders() -> None:
    result = regex_scrub(
        "Email Dr. Ada Lovelace at ada@example.com or ada@example.com, "
        "call 312-555-1212, SSN 123-45-6789."
    )

    assert "ada@example.com" not in result.text
    assert result.text.count("[EMAIL_1]") == 2
    assert "[NAME_1]" in result.text
    assert "[PHONE_1]" in result.text
    assert "[SSN_1]" in result.text
    assert result.counts == {"email": 2, "name": 1, "phone": 1, "ssn": 1}


def test_scrub_traces_marks_meta_and_writes_report(tmp_path: Path) -> None:
    record = import_jsonl_paths([FIXTURE_PATH]).records[0]
    prompt = "Dr. Ada Lovelace can be reached at ada@example.com."
    record = record.model_copy(
        update={
            "request": record.request.model_copy(
                update={
                    "messages": [
                        record.request.messages[0].model_copy(update={"content": prompt})
                    ]
                }
            )
        }
    )

    result = scrub_traces([record])

    assert result.records[0].meta.pii_scrubbed is True
    assert "[EMAIL_1]" in result.records[0].request.messages[0].content
    assert result.report["counts"] == {"email": 1, "name": 1}
    digest = write_scrub_artifacts(
        result,
        traces_path=tmp_path / "scrubbed.jsonl",
        report_path=tmp_path / "scrub_report.json",
    )

    assert digest == result.sha256
    payload = json.loads((tmp_path / "scrub_report.json").read_text(encoding="utf-8"))
    assert payload["counts"] == {"email": 1, "name": 1}
    assert b"\r\n" not in (tmp_path / "scrubbed.jsonl").read_bytes()
