from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from acsi.cli import app

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_traces.jsonl"


def _write_manifest(path: Path) -> Path:
    payload = {
        "assertions": [{"id": "json-valid", "severity": "critical", "type": "json_valid"}],
        "baseline": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        "budget": {"max_usd": 1.0, "use_batch_api": False},
        "candidate": {"provider": "anthropic", "model": "claude-sonnet-5"},
        "judging": {
            "families_allowed": ["openai"],
            "judges": [{"model": "openai/fake-judge"}],
            "min_judges": 1,
        },
        "privacy": {"egress": "hosted_api", "scrub": True},
        "sampling": {"k_baseline": 2, "n": 300, "seed": 42, "stratify_by": ["template_id"]},
        "thresholds": {"confidence": 0.95, "epsilon_pp": 2.0, "max_critical": 0},
        "workload": "support-ticket-summary",
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _run(args: list[str]):
    return CliRunner().invoke(app, args)


def test_run_resumes_most_recent_incomplete_run(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "acsi.yaml")
    run_dir = tmp_path / ".acsi"
    common = [
        "run",
        "--manifest",
        str(manifest),
        "--traces",
        str(FIXTURE_PATH),
        "--run-dir",
        str(run_dir),
        "--fake-noise",
        "0.05",
        "--inject-broken-json-rate",
        "0.08",
        "--yes",
        "--json",
    ]

    # Control run (isolated dir) provides the deterministic reference artifacts.
    control_dir = tmp_path / ".control"
    control = _run(
        [
            "run",
            "--manifest",
            str(manifest),
            "--traces",
            str(FIXTURE_PATH),
            "--run-dir",
            str(control_dir),
            "--fake-noise",
            "0.05",
            "--inject-broken-json-rate",
            "0.08",
            "--yes",
            "--json",
            "--run-id",
            "control",
        ]
    )
    assert control.exit_code == 0, control.output
    control_run = control_dir / "runs" / "control"

    # Interrupt mid-judging with no explicit run id.
    interrupted = _run([*common, "--interrupt-after-judge-dispatches", "2"])
    assert interrupted.exit_code == 1, interrupted.output
    runs = [p for p in (run_dir / "runs").iterdir() if p.is_dir()]
    assert len(runs) == 1
    first_run_id = runs[0].name
    assert not (runs[0] / "cert.json").exists()

    # Re-invoke with no flags: it resumes the incomplete run rather than starting new.
    resumed = _run(common)
    assert resumed.exit_code == 0, resumed.output
    payload = json.loads(resumed.output)
    assert payload["run_id"] == first_run_id
    assert len([p for p in (run_dir / "runs").iterdir() if p.is_dir()]) == 1

    # Deterministic artifacts match the uninterrupted control (run-id independent).
    resumed_run = run_dir / "runs" / first_run_id
    for artifact in ("judgments.jsonl", "assertion_results.jsonl", "clusters.json"):
        assert (resumed_run / artifact).read_bytes() == (control_run / artifact).read_bytes()


def test_fresh_forces_a_new_run(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "acsi.yaml")
    run_dir = tmp_path / ".acsi"
    common = [
        "run",
        "--manifest",
        str(manifest),
        "--traces",
        str(FIXTURE_PATH),
        "--run-dir",
        str(run_dir),
        "--fake-noise",
        "0.05",
        "--inject-broken-json-rate",
        "0.08",
        "--yes",
        "--json",
    ]

    interrupted = _run([*common, "--interrupt-after-judge-dispatches", "2"])
    assert interrupted.exit_code == 1
    assert len([p for p in (run_dir / "runs").iterdir() if p.is_dir()]) == 1

    # --fresh ignores the resumable run and starts a second one.
    fresh = _run([*common, "--fresh"])
    assert fresh.exit_code == 0, fresh.output
    assert len([p for p in (run_dir / "runs").iterdir() if p.is_dir()]) == 2
