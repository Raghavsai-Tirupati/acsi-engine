from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from acsi.cli import app
from acsi.config import load_workload_manifest
from scripts.generate_benchmark_traces import TEMPLATE_ID, generate_traces

BENCHMARK_DIR = Path("benchmarks/oss-issues")


def test_oss_issue_workload_definition_loads() -> None:
    manifest = load_workload_manifest(BENCHMARK_DIR / "acsi.yaml")
    schema = json.loads((BENCHMARK_DIR / "summary.schema.json").read_text(encoding="utf-8"))

    assert manifest.workload == "oss-issue-summary"
    assert manifest.baseline.model == "claude-opus-4-1"
    assert manifest.candidate.model == "claude-sonnet-5"
    assert manifest.sampling.stratify_by == ["input_length_bucket"]
    assert manifest.judging.min_judges == 2
    assert manifest.judging.families_allowed == ["openai", "google", "local"]
    assert manifest.assertions[0].schema_ref == "summary.schema.json"
    assert manifest.assertions[0].model_extra["schema"] == schema
    assert manifest.assertions[-1].prompt_ref == "fabrication.txt"
    assert "SPEC-NOTE" in (BENCHMARK_DIR / "acsi.yaml").read_text(encoding="utf-8")


def test_fake_generation_is_importable_and_idempotent(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    output = tmp_path / "traces.jsonl"
    _write_corpus(corpus, count=3)

    result = generate_traces(
        corpus_path=corpus,
        manifest_path=BENCHMARK_DIR / "acsi.yaml",
        system_prompt_path=BENCHMARK_DIR / "system_prompt.txt",
        output_path=output,
        fake=True,
        yes=True,
        now="2026-07-16T00:00:00Z",
    )

    assert result.written == 3
    assert result.skipped_existing == 0
    assert result.resolved_served_models == ["claude-opus-4-1"]
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["meta"]["template_id"] for record in records] == [TEMPLATE_ID] * 3
    assert records[0]["request"]["params"] == {"max_tokens": 700, "temperature": 0.2}

    second = generate_traces(
        corpus_path=corpus,
        manifest_path=BENCHMARK_DIR / "acsi.yaml",
        system_prompt_path=BENCHMARK_DIR / "system_prompt.txt",
        output_path=output,
        fake=True,
        yes=True,
        now="2026-07-16T00:00:00Z",
    )
    assert second.written == 0
    assert second.skipped_existing == 3
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3

    normalized = tmp_path / "normalized.jsonl"
    imported = CliRunner().invoke(
        app,
        ["import", "jsonl", str(output), "--out", str(normalized), "--json"],
    )
    assert imported.exit_code == 0, imported.output
    payload = json.loads(imported.output)
    assert payload["lines_read"] == 3
    assert payload["malformed"] == 0
    assert payload["valid"] == 3
    assert payload["template_ids"] == {TEMPLATE_ID: 3}
    assert payload["templateless"] == 0
    assert payload["source_counts"] == {"capture": 3}


def test_live_generation_requires_anthropic_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Set ANTHROPIC_API_KEY"):
        generate_traces(
            corpus_path=tmp_path / "missing.jsonl",
            manifest_path=BENCHMARK_DIR / "acsi.yaml",
            system_prompt_path=BENCHMARK_DIR / "system_prompt.txt",
            output_path=tmp_path / "traces.jsonl",
            fake=False,
            yes=True,
        )


def test_generation_requires_yes_before_dispatch(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus, count=1)

    with pytest.raises(RuntimeError, match="Re-run with --yes"):
        generate_traces(
            corpus_path=corpus,
            manifest_path=BENCHMARK_DIR / "acsi.yaml",
            system_prompt_path=BENCHMARK_DIR / "system_prompt.txt",
            output_path=tmp_path / "traces.jsonl",
            fake=True,
            yes=False,
        )


def _write_corpus(path: Path, *, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(count):
            payload = {
                "body": f"Failure details for issue {index}. " * 30,
                "fetched_at": "2026-07-16T00:00:00Z",
                "html_url": f"https://github.com/example/project/issues/{index}",
                "issue_number": index,
                "source_repo": "example/project",
                "title": f"Example issue {index}",
                "truncated": False,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
