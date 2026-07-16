from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from acsi.cli import app


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

    for item in payload["runs"]:
        run_path = Path(item["run_dir"])
        assert run_path == run_dir / "runs" / item["run_id"]
        assert Path(item["report_path"]) == run_path / "report.html"
        assert Path(item["report_path"]).exists()

        verify = CliRunner().invoke(app, ["verify", item["cert_path"], "--json"])
        assert verify.exit_code == 0, verify.output
        assert json.loads(verify.output)["status"] == "ok"
