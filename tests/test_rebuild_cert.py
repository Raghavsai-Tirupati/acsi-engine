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


def test_rebuild_cert_reissues_without_spend_and_preserves_original(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "acsi.yaml")
    run_dir = tmp_path / ".acsi"
    run_id = "00000000-0000-0000-0000-0000000006fe"
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--manifest",
            str(manifest),
            "--traces",
            str(FIXTURE_PATH),
            "--run-dir",
            str(run_dir),
            "--run-id",
            run_id,
            "--fake-noise",
            "0.05",
            "--inject-broken-json-rate",
            "0.08",
            "--yes",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    original_cert = (run_dir / "runs" / run_id / "cert.json").read_bytes()

    out = tmp_path / "rebuilt"
    rebuild = CliRunner().invoke(
        app,
        [
            "rebuild-cert",
            "--run",
            run_id,
            "--manifest",
            str(manifest),
            "--run-dir",
            str(run_dir),
            "--out",
            str(out),
            "--json",
        ],
    )
    assert rebuild.exit_code == 0, rebuild.output
    payload = json.loads(rebuild.output)

    # Zero spend, new location, original untouched.
    assert payload["spend_usd"] == 0.0
    assert Path(payload["cert_path"]) == out / "cert.json"
    assert (run_dir / "runs" / run_id / "cert.json").read_bytes() == original_cert
    assert (out / "cert.json").exists()
    assert (out / "report.html").exists()

    # The rebuilt certificate is independently verifiable.
    verify = CliRunner().invoke(app, ["verify", str(out / "cert.json"), "--json"])
    assert verify.exit_code == 0, verify.output
    rebuilt = json.loads((out / "cert.json").read_text(encoding="utf-8"))["payload"]
    assert rebuilt["verdict"] == payload["verdict"]
    # Clusters were re-derived in the new directory.
    assert (out / "clusters.json").exists()


def test_rebuild_cert_rejects_mismatched_manifest(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "acsi.yaml")
    run_dir = tmp_path / ".acsi"
    run_id = "00000000-0000-0000-0000-0000000006fd"
    assert (
        CliRunner()
        .invoke(
            app,
            [
                "run",
                "--manifest",
                str(manifest),
                "--traces",
                str(FIXTURE_PATH),
                "--run-dir",
                str(run_dir),
                "--run-id",
                run_id,
                "--yes",
                "--json",
            ],
        )
        .exit_code
        == 0
    )

    other = tmp_path / "other.yaml"
    payload = json.loads(manifest.read_text())
    payload["thresholds"]["epsilon_pp"] = 5.0  # changes the config hash
    other.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    rebuild = CliRunner().invoke(
        app,
        [
            "rebuild-cert",
            "--run",
            run_id,
            "--manifest",
            str(other),
            "--run-dir",
            str(run_dir),
            "--out",
            str(tmp_path / "rebuilt"),
            "--json",
        ],
    )
    assert rebuild.exit_code == 1
    assert "config_hash" in rebuild.output
