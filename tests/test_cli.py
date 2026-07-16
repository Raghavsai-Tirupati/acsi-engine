from __future__ import annotations

from typer.testing import CliRunner

from acsi.cli import app


def test_help_lists_m0_command_surface() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in [
        "init",
        "import",
        "run",
        "baseline",
        "replay",
        "cluster",
        "judge",
        "cert",
        "review",
        "monitor",
        "verify",
        "publish",
        "schema",
    ]:
        assert command in result.output


def test_cert_requires_run_id_json_output() -> None:
    result = CliRunner().invoke(app, ["cert", "--json"])

    assert result.exit_code == 1
    assert '"status": "error"' in result.output
    assert "Pass --run" in result.output
