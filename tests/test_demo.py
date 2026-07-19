from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from acsi.cert.render import render_report
from acsi.cli import app

LONG_RAW_FLOAT_RE = re.compile(r"\.\d{7,}")


def test_demo_runs_pass_and_block_and_verifies_both_certs(tmp_path: Path) -> None:
    run_dir = tmp_path / ".acsi"

    result = CliRunner().invoke(
        app,
        [
            "demo",
            "--run-dir",
            str(run_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert {item["run_id"]: item["verdict"] for item in payload["runs"]} == {
        "demo-pass": "PASS",
        "demo-block": "BLOCK",
    }
    by_run = {item["run_id"]: item for item in payload["runs"]}

    for item in payload["runs"]:
        run_path = Path(item["run_dir"])
        assert run_path == run_dir / "runs" / item["run_id"]
        assert Path(item["report_path"]) == run_path / "report.html"
        assert Path(item["report_path"]).exists()

        verify = CliRunner().invoke(app, ["verify", item["cert_path"], "--json"])
        assert verify.exit_code == 0, verify.output
        assert json.loads(verify.output)["status"] == "ok"

    pass_cert = _cert(by_run["demo-pass"])
    block_cert = _cert(by_run["demo-block"])
    pass_payload = pass_cert["payload"]
    block_payload = block_cert["payload"]
    pass_html = Path(by_run["demo-pass"]["report_path"]).read_text(encoding="utf-8")
    block_html = Path(by_run["demo-block"]["report_path"]).read_text(encoding="utf-8")
    block_text = " ".join(block_html.split())

    expected_pass_sentence = (
        "PASS at n=300, covering 100.0% of production template distribution, "
        "95% CI [0.0%, 0.0%]. This certifies the sampled workload against the stated "
        "assertions; it does not certify unsampled inputs."
    )
    expected_block_sentence = (
        "BLOCK at n=300, covering 100.0% of production template distribution, "
        "95% CI [5.0%, 11.0%]. This certifies the sampled workload against the stated "
        "assertions; it does not certify unsampled inputs."
    )
    assert pass_payload["coverage_sentence"] == expected_pass_sentence
    assert block_payload["coverage_sentence"] == expected_block_sentence
    assert expected_pass_sentence in pass_html
    assert expected_block_sentence in block_html

    assert block_payload["regressed_pairs"] == {
        "by_source": {"assertion": 0, "both": 24, "judge": 0},
        "count": 24,
        "rate": 0.08,
        "unresolved": 0,
        "unresolved_also_regressed": 0,
        "unresolved_only": 0,
        "unresolved_rate": 0.0,
    }
    assert (
        "24/300 sampled pairs regressed (8.0%) — 0 by assertion, "
        "0 by judge verdict, 24 by both."
    ) in block_text
    assert (
        block_payload["regressed_pairs"]["by_source"]["assertion"]
        + block_payload["regressed_pairs"]["by_source"]["both"]
        == 24
    )
    assert (
        block_payload["regressed_pairs"]["by_source"]["judge"]
        + block_payload["regressed_pairs"]["by_source"]["both"]
        > 0
    )
    assert "judge-adjudicated regression vs noise floor" in block_html

    pass_noise = pass_payload["noise_floor"]
    pass_criterion_b = pass_payload["criteria"][1]
    assert pass_payload["noise_floor_raw"]["threshold_source"] == "calibrated"
    assert pass_payload["noise_floor_raw"]["analytic_note"]["q"] == 0.05
    assert pass_noise["rate"] > 0
    assert pass_criterion_b["actual_ci_upper"] <= pass_criterion_b["threshold"]
    assert "Threshold τ" in pass_html
    assert "threshold_source" in pass_html
    assert "calibrated" in pass_html

    assert "None" not in pass_html
    assert "None" not in block_html
    assert "n/a — no pairs required judging" in pass_html
    assert "n/a — no calibration set provided" in block_html
    assert "100.0%" in block_html
    assert "1.00" in block_html

    assert re.search(r"Output length inflation</th><td>-?\d+\.\d{2}×", block_html)
    assert re.search(r"Latency delta</th><td>-?\d+ ms", block_html)
    assert re.search(r"USD delta</th><td>\$-?\d+\.\d{4}", block_html)
    assert not LONG_RAW_FLOAT_RE.search(pass_html)
    assert not LONG_RAW_FLOAT_RE.search(block_html)

    assert any(cluster["name"] == "Broken JSON output" for cluster in block_payload["clusters"])

    single_judge_cert = json.loads(json.dumps(block_cert))
    single_judge_cert["payload"]["judge_panel"].update(
        {
            "agreement_percent": None,
            "agreement_reason": "requires ≥2 judges with comparable verdicts",
            "krippendorff_alpha": None,
            "krippendorff_alpha_reason": "requires ≥2 judges with comparable verdicts",
            "models": ["openai/fake-judge-a"],
        }
    )
    single_judge_report = tmp_path / "single-judge-report.html"
    render_report(single_judge_cert, output_path=single_judge_report)
    assert "n/a — requires ≥2 judges with comparable verdicts" in single_judge_report.read_text(
        encoding="utf-8"
    )


def _cert(item: dict[str, str]) -> dict:
    return json.loads(Path(item["cert_path"]).read_text(encoding="utf-8"))
