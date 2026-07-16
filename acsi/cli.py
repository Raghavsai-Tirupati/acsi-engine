from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from acsi import __version__
from acsi.config import load_workload_manifest
from acsi.importers.common import choose_output_path, inventory_table, write_import_artifacts
from acsi.importers.jsonl import import_jsonl_paths
from acsi.importers.supabase import (
    SupabaseConfig,
    SupabaseImportError,
    import_supabase_records,
)
from acsi.replay.artifacts import RunClock, build_run_manifest, write_run_manifest
from acsi.replay.clients import FakeClient, LiveClient
from acsi.replay.runner import (
    ReplayAbortError,
    ReplayConfig,
    ReplayInterrupted,
    estimate_call_cost_usd,
    estimated_output_tokens,
    write_responses_jsonl,
)
from acsi.replay.runner import (
    replay as replay_traces,
)
from acsi.replay.store import ReplayStore
from acsi.schemas import ProviderModel, TraceRecord, export_json_schemas

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
            if input_paths:
                _fail("Supabase import does not accept input path arguments.", json_output)
            if not workload:
                _fail("Pass --workload to import Supabase traces.", json_output)
            result = import_supabase_records(
                SupabaseConfig.from_env(),
                workload=workload,
                since=since,
            )
        else:
            _fail("Unsupported importer. Use `jsonl` or `supabase`.", json_output)

        output_path = choose_output_path(result.records, out, workload=workload)
        digest = write_import_artifacts(result, output_path)
    except (OSError, ValueError, SupabaseImportError) as exc:
        _fail(str(exc), json_output)

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


def _target_model(default: ProviderModel, target: str | None) -> ProviderModel:
    if not target:
        return default
    if "/" in target:
        provider, model = target.split("/", 1)
        return ProviderModel(provider=provider, model=model)
    return ProviderModel(provider=default.provider, model=target)


def _estimate_replay_cost(
    traces: list[TraceRecord],
    model: ProviderModel,
    k_samples: int,
    *,
    fake: bool,
) -> float:
    total = 0.0
    for trace in traces:
        input_tokens = trace.response.usage.input_tokens if trace.response.usage else 0
        output_tokens = estimated_output_tokens(trace)
        total += estimate_call_cost_usd(
            model.provider,
            model.model,
            input_tokens,
            output_tokens,
            fake=fake,
        )
    return total * k_samples


def _confirm_replay(
    traces: list[TraceRecord],
    model: ProviderModel,
    estimate_usd: float,
) -> None:
    table = Table(title="ACSI Replay Preflight")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Traces", str(len(traces)))
    table.add_row("Target", f"{model.provider}/{model.model}")
    table.add_row("Estimated cost", f"${estimate_usd:.6f}")
    console.print(table)
    if not typer.confirm("Proceed with replay?"):
        raise typer.Exit(1)


def _resume_command(manifest: Path, traces: Path, run_id: str) -> str:
    return f"acsi replay --manifest {manifest} --traces {traces} --run-id {run_id} --yes"


def _replay_summary_table(payload: dict[str, object]) -> Table:
    table = Table(title="ACSI Replay Summary")
    table.add_column("Metric")
    table.add_column("Value")
    for key in (
        "status",
        "run_id",
        "run_dir",
        "completed",
        "errors",
        "cache_hits",
        "dispatched",
        "retry_count",
        "cost_usd",
        "halted_reason",
        "responses_sha256",
        "run_sha256",
    ):
        table.add_row(key, str(payload.get(key)))
    return table


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
    manifest: ManifestOption = Path("acsi.yaml"),
    traces: Annotated[
        Path | None,
        typer.Option("--traces", help="Normalized TraceRecord JSONL path."),
    ] = None,
    target: Annotated[
        str | None,
        typer.Option("--target", help="Override target as provider/model or model."),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Resume or create this run id."),
    ] = None,
    k_samples: Annotated[int, typer.Option("--k-samples", help="Replay samples per trace.")] = 1,
    seed: Annotated[int, typer.Option("--seed", help="Replay seed.")] = 42,
    concurrency: Annotated[int, typer.Option("--concurrency", help="Replay concurrency.")] = 4,
    max_cost: Annotated[
        float | None,
        typer.Option("--max-cost", help="Maximum replay spend in USD."),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Approve replay execution.")] = False,
    live: Annotated[
        bool,
        typer.Option("--live", help="Use LiveClient instead of FakeClient."),
    ] = False,
    fake_noise: Annotated[float, typer.Option("--fake-noise", help="FakeClient noise rate.")] = 0.0,
    degraded: Annotated[
        bool,
        typer.Option("--degraded", help="M3 baseline degraded mode placeholder."),
    ] = False,
    json_output: JsonOutputOption = False,
) -> None:
    if degraded:
        _fail("--degraded is scheduled for M3 baseline mode.", json_output)
    try:
        manifest_model = load_workload_manifest(manifest)
        target_model = _target_model(manifest_model.candidate, target)
        traces_path = traces or Path(".acsi") / "traces" / f"{manifest_model.workload}.jsonl"
        trace_records = import_jsonl_paths([traces_path]).records
        if not trace_records:
            _fail(f"No valid traces found in {traces_path}.", json_output)

        active_run_id = run_id or str(uuid4())
        run_dir = Path(".acsi") / "runs" / active_run_id
        store = ReplayStore(run_dir / "replay.sqlite")
        max_cost_usd = max_cost if max_cost is not None else manifest_model.budget.max_usd
        estimate = _estimate_replay_cost(trace_records, target_model, k_samples, fake=not live)
        if not yes:
            if json_output:
                _fail("Pass --yes to approve replay execution.", json_output)
            _confirm_replay(trace_records, target_model, estimate)

        client = LiveClient() if live else FakeClient(seed=seed, noise=fake_noise)
        clock = RunClock()
        result = asyncio.run(
            replay_traces(
                trace_records,
                target_model,
                k_samples,
                client=client,
                store=store,
                config=ReplayConfig(
                    run_id=active_run_id,
                    seed=seed,
                    concurrency=concurrency,
                    max_cost_usd=max_cost_usd,
                    resume_command=_resume_command(manifest, traces_path, active_run_id),
                ),
            )
        )
        responses_hash = write_responses_jsonl(store, active_run_id, run_dir / "responses.jsonl")
        run_manifest = build_run_manifest(
            run_id=active_run_id,
            manifest_path=manifest,
            traces_path=traces_path,
            seed=seed,
            provider=target_model.provider,
            endpoint="live" if live else "fake",
            store=store,
            result=result,
            wall_clock_seconds=clock.elapsed_seconds(),
        )
        run_hash = write_run_manifest(run_dir / "run.json", run_manifest)
    except ReplayAbortError as exc:
        _fail(str(exc), json_output)
    except ReplayInterrupted as exc:
        _fail(str(exc), json_output)
    except (OSError, ValueError) as exc:
        _fail(str(exc), json_output)

    payload = {
        "status": "ok" if not result.halted_reason else "halted",
        "run_id": active_run_id,
        "run_dir": str(run_dir),
        "completed": result.completed,
        "errors": result.errors,
        "cache_hits": result.cache_hits,
        "dispatched": result.dispatched,
        "retry_count": result.retry_count,
        "cost_usd": result.cost_usd,
        "halted_reason": result.halted_reason,
        "responses_sha256": responses_hash,
        "run_sha256": run_hash,
    }
    if json_output:
        console.print_json(data=payload)
    else:
        console.print(_replay_summary_table(payload))


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
