from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from acsi.cli import app

RUN_ID = "00000000-0000-0000-0000-000000000501"


def test_cluster_cli_writes_clusters_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest, traces = _write_inputs(tmp_path)
    _write_run_artifacts(tmp_path, traces)

    result = CliRunner().invoke(
        app,
        [
            "cluster",
            "--run",
            RUN_ID,
            "--manifest",
            str(manifest),
            "--traces",
            str(traces),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    run_dir = Path(payload["run_dir"])
    assert payload["regression_count"] == 1
    assert payload["cluster_count"] == 1
    assert (run_dir / "clusters.json").exists()
    assert (run_dir / "clusters.json.sha256").exists()


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "acsi.yaml"
    traces = tmp_path / "traces.jsonl"
    manifest_payload = {
        "assertions": [],
        "baseline": {"provider": "anthropic", "model": "claude-old"},
        "budget": {"max_usd": 1.0, "use_batch_api": False},
        "candidate": {"provider": "anthropic", "model": "claude-new"},
        "judging": {"families_allowed": ["openai"], "min_judges": 1},
        "privacy": {"egress": "hosted_api", "scrub": True},
        "sampling": {"k_baseline": 2, "n": 2, "seed": 42, "stratify_by": []},
        "thresholds": {"confidence": 0.95, "epsilon_pp": 2.0, "max_critical": 0},
        "workload": "demo",
    }
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest_payload, sort_keys=True))
        handle.write("\n")
    fixture_lines = (
        Path(__file__).parent / "fixtures" / "synthetic_traces.jsonl"
    ).read_text(encoding="utf-8").splitlines()[:2]
    with traces.open("w", encoding="utf-8", newline="\n") as handle:
        for line in fixture_lines:
            handle.write(f"{line}\n")
    return manifest, traces


def _write_run_artifacts(tmp_path: Path, traces: Path) -> None:
    run_dir = tmp_path / ".acsi" / "runs" / RUN_ID
    (run_dir / "baseline").mkdir(parents=True)
    (run_dir / "candidate").mkdir(parents=True)
    trace_ids = [
        json.loads(line)["trace_id"]
        for line in traces.read_text(encoding="utf-8").splitlines()
    ]
    _write_responses(
        run_dir / "baseline" / "responses.jsonl",
        [(trace_ids[0], "baseline ok"), (trace_ids[1], "baseline ok")],
    )
    _write_responses(
        run_dir / "candidate" / "responses.jsonl",
        [(trace_ids[0], "candidate broken json"), (trace_ids[1], "baseline ok")],
    )
    with (run_dir / "judgments.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        rows = [
            {"judge": "openai/fake", "outcome": "worse_minor", "pair_id": trace_ids[0]},
            {"judge": "openai/fake", "outcome": "equivalent", "pair_id": trace_ids[1]},
        ]
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _write_responses(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for trace_id, text in rows:
            payload = {
                "cost_usd": 0.0,
                "model": "model",
                "response": {
                    "finish_reason": "stop",
                    "latency_ms": 1,
                    "served_model": "model",
                    "text": text,
                    "tool_calls": None,
                },
                "retry_count": 0,
                "sample_index": 0,
                "served_model": "model",
                "status": "done",
                "trace_id": trace_id,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
