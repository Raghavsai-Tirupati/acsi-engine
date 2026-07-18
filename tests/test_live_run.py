from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from acsi.cli import app

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_traces.jsonl"
LIVE_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
FAKE_CLIENT_BANNER = "FAKE CLIENTS — NOT A CERTIFICATION"


def _write_manifest(path: Path, *, n: int = 8) -> Path:
    payload = {
        "assertions": [{"id": "json-valid", "severity": "critical", "type": "json_valid"}],
        "baseline": {"provider": "anthropic", "model": "claude-opus-4-1"},
        "budget": {"max_usd": 60.0, "use_batch_api": False},
        "candidate": {"provider": "anthropic", "model": "claude-sonnet-5"},
        "judging": {
            "families_allowed": ["openai", "google"],
            "judges": [
                {"provider": "openai", "model": "gpt-5.4-mini"},
                {"provider": "google", "model": "gemini-3.5-flash"},
            ],
            "min_judges": 2,
        },
        "privacy": {"egress": "hosted_api", "scrub": True},
        "sampling": {"k_baseline": 2, "n": n, "seed": 42, "stratify_by": ["template_id"]},
        "thresholds": {"confidence": 0.95, "epsilon_pp": 2.0, "max_critical": 0},
        "workload": "support-ticket-summary",
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_run_preflight_table_prints_before_approval_error(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "acsi.yaml")

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--manifest",
            str(manifest),
            "--traces",
            str(FIXTURE_PATH),
            "--run-dir",
            str(tmp_path / ".acsi"),
        ],
    )

    assert result.exit_code == 1
    out = result.output
    assert "Estimated cost" in out
    assert "$0.000000 (fake clients)" in out
    assert "FAKE" in out
    assert "Pass --yes" in out
    # The price is shown before approval is demanded.
    assert out.index("Estimated cost") < out.index("Pass --yes")
    assert not (tmp_path / ".acsi" / "runs").exists()


def test_live_preflight_shows_live_mode_and_cost_without_spending(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "acsi.yaml")

    # No --yes: the LIVE preflight table (with a real cost estimate) prints, then
    # the approval error — no provider is ever called.
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--manifest",
            str(manifest),
            "--traces",
            str(FIXTURE_PATH),
            "--run-dir",
            str(tmp_path / ".acsi"),
            "--live",
        ],
    )

    assert result.exit_code == 1
    out = result.output
    assert "LIVE" in out
    assert "(LIVE)" in out
    assert "$0.000000 (fake clients)" not in out
    assert "Pass --yes" in out
    assert out.index("Estimated cost") < out.index("Pass --yes")
    assert not (tmp_path / ".acsi" / "runs").exists()


def test_live_estimate_is_nonzero_over_the_sample_plan(tmp_path: Path) -> None:
    from acsi.cli import _estimate_run_cost
    from acsi.config import load_workload_manifest
    from acsi.importers.jsonl import import_jsonl_paths

    manifest_path = _write_manifest(tmp_path / "acsi.yaml")
    manifest = load_workload_manifest(manifest_path)
    traces = import_jsonl_paths([FIXTURE_PATH]).records

    live_cost = _estimate_run_cost(traces, manifest, fake=False)
    fake_cost = _estimate_run_cost(traces, manifest, fake=True)

    assert live_cost > 0.0
    assert fake_cost == 0.0 or fake_cost < live_cost


def test_live_run_missing_keys_aborts_before_spend(tmp_path: Path, monkeypatch) -> None:
    import litellm  # noqa: F401  # force load_dotenv now so deletions below stick

    for key in LIVE_KEYS:
        monkeypatch.delenv(key, raising=False)
    manifest = _write_manifest(tmp_path / "acsi.yaml")

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--manifest",
            str(manifest),
            "--traces",
            str(FIXTURE_PATH),
            "--run-dir",
            str(tmp_path / ".acsi"),
            "--live",
            "--yes",
        ],
    )

    assert result.exit_code == 1
    out = result.output
    assert "Missing required provider environment variables" in out
    for key in LIVE_KEYS:
        assert key in out
    # Aborted before creating any run artifacts or calling a provider.
    assert not (tmp_path / ".acsi" / "runs").exists()


def test_fake_run_writes_watermarked_certificate(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "acsi.yaml", n=300)
    run_id = "00000000-0000-0000-0000-0000000006f0"

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--manifest",
            str(manifest),
            "--traces",
            str(FIXTURE_PATH),
            "--run-dir",
            str(tmp_path / ".acsi"),
            "--run-id",
            run_id,
            "--fake-noise",
            "0.05",
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    run_dir = Path(payload["run_dir"])
    cert_payload = json.loads((run_dir / "cert.json").read_text(encoding="utf-8"))["payload"]
    report_text = (run_dir / "report.html").read_text(encoding="utf-8")

    assert cert_payload["client_mode"] == "fake"
    assert FAKE_CLIENT_BANNER in report_text
