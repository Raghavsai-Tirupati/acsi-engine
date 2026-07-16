from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from acsi.cli import app

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_traces.jsonl"
RUN_A = "00000000-0000-0000-0000-0000000006a0"
RUN_B = "00000000-0000-0000-0000-0000000006b0"
BANNED_RE = re.compile(
    r"\b(?:guarantee|guaranteed|identical|zero risk|proven equivalent)\b",
    re.IGNORECASE,
)


def test_spec_a_clean_seeded_run_passes_and_verifies(tmp_path: Path) -> None:
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
            "--run-id",
            RUN_A,
            "--fake-noise",
            "0.05",
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    run_dir = Path(payload["run_dir"])
    cert = json.loads((run_dir / "cert.json").read_text(encoding="utf-8"))
    cert_payload = cert["payload"]

    assert cert_payload["verdict"] == "PASS"
    assert cert_payload["coverage"]["n"] == 300
    assert cert_payload["candidate_disagreement"] == {
        "confidence": 0.95,
        "lower": 0.0,
        "rate": 0.0,
        "upper": 0.0,
    }
    assert cert_payload["noise_floor"]["rate"] > 0.0
    assert cert_payload["noise_floor"]["upper"] == 0.016666666667
    assert cert_payload["criteria"][1]["actual_ci_upper"] <= cert_payload["criteria"][1][
        "threshold"
    ]
    assert "≤ 1.0%" in cert_payload["coverage"]["zero_event_bound_sentence"]
    verify = CliRunner().invoke(app, ["verify", str(run_dir / "cert.json"), "--json"])
    assert verify.exit_code == 0, verify.output
    assert not BANNED_RE.search((run_dir / "cert.json").read_text(encoding="utf-8"))
    assert not BANNED_RE.search((run_dir / "report.html").read_text(encoding="utf-8"))


def test_spec_b_injected_broken_json_blocks_and_clusters(tmp_path: Path) -> None:
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
            "--run-id",
            RUN_B,
            "--fake-noise",
            "0.05",
            "--inject-broken-json-rate",
            "0.08",
            "--yes",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    run_dir = Path(payload["run_dir"])
    cert_payload = json.loads((run_dir / "cert.json").read_text(encoding="utf-8"))["payload"]
    clusters = json.loads((run_dir / "clusters.json").read_text(encoding="utf-8"))["clusters"]
    patches = json.loads((run_dir / "patches" / "patch_report.json").read_text(encoding="utf-8"))

    assert cert_payload["verdict"] == "BLOCK"
    assert cert_payload["criteria"][0]["actual"] == 24
    assert cert_payload["assertions_by_severity"]["critical"]["failures"] == 24
    assert clusters
    assert clusters[0]["name"]
    assert clusters[0]["severity"] == "worse_critical"
    assert patches["patches"]


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
