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


FAKE_BANNER = "FAKE CLIENTS — NOT A CERTIFICATION"


def _fake_run(tmp_path: Path, run_id: str) -> tuple[Path, Path]:
    manifest = _write_manifest(tmp_path / "acsi.yaml")
    run_dir = tmp_path / ".acsi"
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
    return manifest, run_dir


def _rebuild(manifest: Path, run_dir: Path, run_id: str, out: Path):
    return CliRunner().invoke(
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


def _patch_json(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_client_mode_persisted_in_run_json(tmp_path: Path) -> None:
    _manifest, run_dir = _fake_run(tmp_path, "00000000-0000-0000-0000-0000000006fa")
    run_json = json.loads(
        (run_dir / "runs" / "00000000-0000-0000-0000-0000000006fa" / "run.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_json["client_mode"] == "fake"


def test_rebuild_fake_run_keeps_banner(tmp_path: Path) -> None:
    run_id = "00000000-0000-0000-0000-0000000006fb"
    manifest, run_dir = _fake_run(tmp_path, run_id)
    out = tmp_path / "rebuilt"
    result = _rebuild(manifest, run_dir, run_id, out)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["client_mode"] == "fake"
    assert json.loads((out / "cert.json").read_text())["payload"]["client_mode"] == "fake"
    assert FAKE_BANNER in (out / "report.html").read_text(encoding="utf-8")


def test_rebuild_live_run_carries_mode_and_drops_banner(tmp_path: Path) -> None:
    run_id = "00000000-0000-0000-0000-0000000006fc"
    manifest, run_dir = _fake_run(tmp_path, run_id)
    # Simulate a live run's persisted state.
    _patch_json(
        run_dir / "runs" / run_id / "run.json",
        lambda payload: payload.update(client_mode="live"),
    )
    out = tmp_path / "rebuilt"
    result = _rebuild(manifest, run_dir, run_id, out)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["client_mode"] == "live"
    assert json.loads((out / "cert.json").read_text())["payload"]["client_mode"] == "live"
    assert FAKE_BANNER not in (out / "report.html").read_text(encoding="utf-8")


def test_rebuild_reads_mode_from_cert_when_run_json_lacks_it(tmp_path: Path) -> None:
    # The 7f0978f5 case: run.json has no client_mode, original cert payload does.
    run_id = "00000000-0000-0000-0000-0000000006f9"
    manifest, run_dir = _fake_run(tmp_path, run_id)
    src = run_dir / "runs" / run_id
    _patch_json(src / "run.json", lambda payload: payload.pop("client_mode", None))
    _patch_json(src / "cert.json", lambda cert: cert["payload"].update(client_mode="live"))
    out = tmp_path / "rebuilt"
    result = _rebuild(manifest, run_dir, run_id, out)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["client_mode"] == "live"
    assert FAKE_BANNER not in (out / "report.html").read_text(encoding="utf-8")


def test_rebuild_refuses_when_mode_unrecoverable(tmp_path: Path) -> None:
    run_id = "00000000-0000-0000-0000-0000000006f8"
    manifest, run_dir = _fake_run(tmp_path, run_id)
    src = run_dir / "runs" / run_id
    _patch_json(src / "run.json", lambda payload: payload.pop("client_mode", None))
    _patch_json(src / "cert.json", lambda cert: cert["payload"].pop("client_mode", None))
    out = tmp_path / "rebuilt"
    result = _rebuild(manifest, run_dir, run_id, out)
    assert result.exit_code == 1
    assert "client_mode" in result.output
    assert not out.exists()


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
