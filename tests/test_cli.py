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
        "judge",
        "cert",
        "review",
        "monitor",
        "verify",
        "publish",
        "schema",
    ]:
        assert command in result.output


def test_stub_commands_support_json_output() -> None:
    result = CliRunner().invoke(app, ["cert", "--json"])

    assert result.exit_code == 2
    assert '"status": "not_implemented"' in result.output
    assert '"milestone": "M6"' in result.output
