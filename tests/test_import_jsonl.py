from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from acsi.cli import app


def test_jsonl_import_acceptance_counts(tmp_path: Path) -> None:
    output = tmp_path / "normalized.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "import",
            "jsonl",
            "tests/fixtures/synthetic_traces.jsonl",
            "tests/fixtures/invalid.jsonl",
            "--out",
            str(output),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = _json_payload(result.output)
    assert payload["lines_read"] == 315
    assert payload["malformed"] == 2
    assert payload["valid"] == 303
    assert payload["template_ids"] == {"volunteer-json-summary-v1": 270}
    assert payload["templateless"] == 33
    assert payload["source_counts"] == {"backfill": 3, "jsonl": 300}
    assert payload["exclusions"]["multi_turn"] == 5
    assert payload["exclusions"]["invalid_response"] == 2
    assert payload["exclusions"]["duplicates"] == 3

    assert len(output.read_text(encoding="utf-8").splitlines()) == 303
    assert output.with_name(output.name + ".sha256").read_text(encoding="utf-8").strip()
    exclusion_lines = output.with_name(output.name + ".exclusions.jsonl").read_text(
        encoding="utf-8"
    )
    exclusions = [json.loads(line) for line in exclusion_lines.splitlines()]
    assert len(exclusions) == 10
    assert {exclusion["reason"] for exclusion in exclusions} == {
        "multi_turn",
        "invalid_response",
        "duplicates",
    }


def test_jsonl_import_default_output_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    fixture = Path(__file__).parent / "fixtures" / "synthetic_traces.jsonl"

    result = CliRunner().invoke(app, ["import", "jsonl", str(fixture), "--json"])

    assert result.exit_code == 0, result.output
    payload = _json_payload(result.output)
    assert Path(payload["output"]) == Path(".acsi") / "traces" / (
        "support-ticket-summary.jsonl"
    )
    assert Path(payload["output"]).exists()


def _json_payload(output: str) -> dict[str, object]:
    return json.loads(output[output.index("{") :])
