from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from acsi.cli import app

RUN_ID = "00000000-0000-0000-0000-0000000000a1"


def test_replay_cli_writes_run_artifacts_and_param_transform(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "acsi.yaml"
    traces = tmp_path / "traces.jsonl"
    _write_manifest(manifest)
    _write_one_trace(traces)

    result = CliRunner().invoke(
        app,
        [
            "replay",
            "--manifest",
            str(manifest),
            "--traces",
            str(traces),
            "--run-id",
            RUN_ID,
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    run_dir = Path(payload["run_dir"])
    run_json = run_dir / "run.json"
    responses = run_dir / "responses.jsonl"

    assert run_json.exists()
    assert responses.exists()
    assert responses.with_name("responses.jsonl.sha256").exists()
    assert run_json.with_name("run.json.sha256").exists()

    run_bytes = run_json.read_bytes()
    assert b"\r\n" not in run_bytes
    assert run_bytes.endswith(b"\n")

    run_payload = json.loads(run_json.read_text(encoding="utf-8"))
    assert run_payload["served_models"] == ["claude-sonnet-5"]
    assert run_payload["param_transformations"][0]["path"] == "params.temperature"
    assert run_payload["param_transformations"][0]["action"] == "strip"
    assert run_payload["param_transformations"][0]["count"] == 1
    assert "HTTP 400" in run_payload["param_transformations"][0]["reason"]


def test_replay_cli_degraded_flag_is_m3_placeholder() -> None:
    result = CliRunner().invoke(app, ["replay", "--degraded", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "error"
    assert "M3" in payload["message"]


def _write_manifest(path: Path) -> None:
    payload = {
        "workload": "volunteer-application-summary",
        "baseline": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        "candidate": {"provider": "anthropic", "model": "claude-sonnet-5"},
        "sampling": {"n": 1, "stratify_by": [], "seed": 42, "k_baseline": 2},
        "assertions": [],
        "judging": {"families_allowed": ["openai"], "min_judges": 1},
        "thresholds": {"epsilon_pp": 2.0, "max_critical": 0, "confidence": 0.95},
        "privacy": {"scrub": True, "egress": "hosted_api"},
        "budget": {"max_usd": 1.0, "use_batch_api": False},
    }
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def _write_one_trace(path: Path) -> None:
    source = Path(__file__).parent / "fixtures" / "synthetic_traces.jsonl"
    line = source.read_text(encoding="utf-8").splitlines()[0]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{line}\n")
