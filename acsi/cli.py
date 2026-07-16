from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from acsi import __version__
from acsi.importers.common import choose_output_path, inventory_table, write_import_artifacts
from acsi.importers.jsonl import import_jsonl_paths
from acsi.schemas import export_json_schemas

app = typer.Typer(
    help="ACSI replays production LLM traces and certifies model swaps against assertions.",
    no_args_is_help=True,
)
schema_app = typer.Typer(help="Export frozen ACSI JSON Schemas.")
app.add_typer(schema_app, name="schema")
console = Console()
err_console = Console(stderr=True)

JsonOutputOption = Annotated[
    bool,
    typer.Option("--json", help="Emit machine-readable output."),
]
RunDirOption = Annotated[
    Path,
    typer.Option("--run-dir", help="Run state directory."),
]
ManifestOption = Annotated[
    Path,
    typer.Option("--manifest", "-m", help="Path to acsi.yaml."),
]


def _emit_stub(command: str, milestone: str, json_output: bool) -> None:
    message = f"`acsi {command}` is scheduled for {milestone}; M0 provides the command surface."
    if json_output:
        console.print_json(
            data={"status": "not_implemented", "command": command, "milestone": milestone}
        )
    else:
        console.print(message)
    raise typer.Exit(2)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Print the ACSI engine version and exit."),
    ] = False,
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit()


@app.command()
def init(
    json_output: JsonOutputOption = False,
) -> None:
    _emit_stub("init", "M1", json_output)


@app.command("import")
def import_(
    source: Annotated[str, typer.Argument(help="Importer name: jsonl or supabase.")],
    input_paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Input JSONL files for file-based importers."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Normalized TraceRecord JSONL output path."),
    ] = None,
    workload: Annotated[
        str | None,
        typer.Option("--workload", help="Workload filter for importers that support it."),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option("--since", help="Optional ISO timestamp lower bound."),
    ] = None,
    json_output: JsonOutputOption = False,
) -> None:
    try:
        if source == "jsonl":
            if not input_paths:
                _fail("Pass at least one JSONL input path.", json_output)
            result = import_jsonl_paths(input_paths, warn=err_console.print)
        elif source == "supabase":
            _fail("Supabase import is not implemented in this module yet.", json_output)
        else:
            _fail("Unsupported importer. Use `jsonl` or `supabase`.", json_output)

        output_path = choose_output_path(result.records, out)
        digest = write_import_artifacts(result, output_path)
    except (OSError, ValueError) as exc:
        _fail(str(exc), json_output)

    _ = (workload, since)
    payload = result.summary.to_payload(output_path=output_path, sha256=digest)
    if json_output:
        console.print_json(data=payload)
    else:
        console.print(inventory_table(payload))


def _fail(message: str, json_output: bool) -> None:
    if json_output:
        console.print_json(data={"status": "error", "message": message})
    else:
        console.print(f"Error: {message}", style="red")
    raise typer.Exit(1)


@app.command()
def run(
    manifest: ManifestOption = Path("acsi.yaml"),
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Approve provider spend without prompting."),
    ] = False,
    json_output: JsonOutputOption = False,
) -> None:
    _ = (manifest, yes)
    _emit_stub("run", "M7", json_output)


@app.command()
def baseline(
    run_dir: RunDirOption = Path(".acsi"),
    degraded: Annotated[
        bool,
        typer.Option(
            "--degraded",
            help="Use stored responses when the baseline model is unavailable.",
        ),
    ] = False,
    json_output: JsonOutputOption = False,
) -> None:
    _ = (run_dir, degraded)
    _emit_stub("baseline", "M3", json_output)


@app.command()
def replay(
    run_dir: RunDirOption = Path(".acsi"),
    json_output: JsonOutputOption = False,
) -> None:
    _ = run_dir
    _emit_stub("replay", "M2", json_output)


@app.command()
def judge(
    run_dir: RunDirOption = Path(".acsi"),
    json_output: JsonOutputOption = False,
) -> None:
    _ = run_dir
    _emit_stub("judge", "M4", json_output)


@app.command()
def cert(
    run_dir: RunDirOption = Path(".acsi"),
    json_output: JsonOutputOption = False,
) -> None:
    _ = run_dir
    _emit_stub("cert", "M6", json_output)


@app.command()
def review(
    report: Annotated[Path, typer.Option("--report", help="Path to report.html.")] = Path(
        "report.html"
    ),
    json_output: JsonOutputOption = False,
) -> None:
    _ = report
    _emit_stub("review", "M7", json_output)


@app.command()
def monitor(
    manifest: ManifestOption = Path("acsi.yaml"),
    json_output: JsonOutputOption = False,
) -> None:
    _ = manifest
    _emit_stub("monitor", "M7", json_output)


@app.command()
def verify(
    cert_path: Annotated[Path, typer.Argument(help="Path to cert.json.")],
    json_output: JsonOutputOption = False,
) -> None:
    _ = cert_path
    _emit_stub("verify", "M6", json_output)


@app.command()
def publish(
    cert_path: Annotated[Path, typer.Argument(help="Path to cert.json.")],
    endpoint: Annotated[
        str | None,
        typer.Option("--endpoint", help="Explicit publish endpoint for verdict JSON."),
    ] = None,
    include_examples: Annotated[
        bool,
        typer.Option(
            "--include-examples",
            help="Include redacted examples in the published payload.",
        ),
    ] = False,
    json_output: JsonOutputOption = False,
) -> None:
    _ = (cert_path, endpoint, include_examples)
    _emit_stub("publish", "post-M6", json_output)


@schema_app.command("export")
def schema_export(
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory that receives JSON Schema files."),
    ] = Path("schemas"),
    json_output: JsonOutputOption = False,
) -> None:
    written = export_json_schemas(output_dir)
    payload = {"status": "ok", "schemas": [str(path) for path in written]}
    if json_output:
        console.print_json(json.dumps(payload))
    else:
        for path in written:
            console.print(f"Wrote {path}")
