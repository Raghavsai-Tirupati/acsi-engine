from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from acsi import __version__
from acsi.baseline import run_baseline as run_baseline_stage
from acsi.cert.build import (
    BannedLanguageError,
    CertificateVerificationError,
    build_certificate,
    verify_certificate,
)
from acsi.cert.render import render_report
from acsi.config import load_workload_manifest
from acsi.diff.assertions import AssertionPair, evaluate_assertions
from acsi.diff.clustering import (
    AssertionFailure,
    CandidatePairRecord,
    FakeNamer,
    build_regression_set,
    cluster_regressions,
    name_clusters,
    write_clusters_json,
)
from acsi.diff.deterministic import DiffResponse
from acsi.importers.common import choose_output_path, inventory_table, write_import_artifacts
from acsi.importers.jsonl import import_jsonl_paths
from acsi.importers.supabase import (
    SupabaseConfig,
    SupabaseImportError,
    import_supabase_records,
)
from acsi.judge.calibration import (
    CalibrationSample,
    ingest_calibration_csv,
    write_calibration_sample,
)
from acsi.judge.clients import (
    FakeJudge,
    LiveJudge,
    select_judge_panel,
)
from acsi.judge.ensemble import aggregate_pair_outcomes
from acsi.judge.runner import (
    JudgeInterrupted,
    JudgeRunConfig,
    build_candidate_pairs,
    run_pairwise_judging,
    select_for_judging,
    write_judge_artifacts,
)
from acsi.patch import (
    FakePatcher,
    PatchReport,
    detect_templates,
    propose_patch,
    select_patch_target,
    write_patch_report,
)
from acsi.publish import PublishError, publish_certificate
from acsi.replay.artifacts import RunClock, build_run_manifest, write_run_manifest
from acsi.replay.clients import FakeClient, LiveClient, RegressionRule
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
from acsi.replay.store import ReplayStore, StoredCall
from acsi.sampling import sample_traces, write_sample_artifacts
from acsi.schemas import ProviderModel, Severity, TraceRecord, export_json_schemas
from acsi.scrub import scrub_traces, write_scrub_artifacts

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


def _baseline_resume_command(manifest: Path, traces: Path, run_id: str) -> str:
    return f"acsi baseline --manifest {manifest} --traces {traces} --run-id {run_id} --yes"


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


def _baseline_summary_table(payload: dict[str, object]) -> Table:
    table = Table(title="ACSI Baseline Summary")
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
        "cost_usd",
        "degraded",
        "threshold_source",
        "textual_mismatch_rate",
        "beyond_noise_rate",
        "beyond_noise_to_textual_mismatch_rate",
        "noise_floor_sha256",
        "responses_sha256",
        "run_sha256",
    ):
        table.add_row(key, str(payload.get(key)))
    return table


def _load_noise_tau(run_dir: Path) -> float:
    noise_path = run_dir / "baseline" / "noise_floor.json"
    payload = json.loads(noise_path.read_text(encoding="utf-8"))
    return float(payload["tau"])


def _baseline_response_paths(run_dir: Path) -> list[Path]:
    return [
        run_dir / "baseline" / "responses.jsonl",
        run_dir / "baseline_responses.jsonl",
        run_dir / "responses.jsonl",
    ]


def _candidate_response_paths(run_dir: Path) -> list[Path]:
    return [
        run_dir / "candidate" / "responses.jsonl",
        run_dir / "candidate_responses.jsonl",
        run_dir / "responses.candidate.jsonl",
        run_dir / "responses.jsonl",
    ]


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these response artifacts exist: {paths}")


def _load_response_calls(path: Path) -> list[StoredCall]:
    calls: list[StoredCall] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            calls.append(
                StoredCall(
                    phase="replay",
                    run_id="",
                    trace_id=str(payload["trace_id"]),
                    sample_index=int(payload["sample_index"]),
                    model=str(payload.get("model", "")),
                    params_hash="",
                    prompt_hash="",
                    status=str(payload.get("status", "done")),
                    response=payload.get("response"),
                    usage=payload.get("usage") or {},
                    cost_usd=float(payload.get("cost_usd", 0.0)),
                    served_model=payload.get("served_model"),
                    error=None,
                    retry_count=int(payload.get("retry_count", 0)),
                )
            )
    return calls


def _stored_diff_response(call: StoredCall) -> DiffResponse:
    response = call.response or {}
    return DiffResponse(
        text=response.get("text"),
        tool_calls=response.get("tool_calls"),
        finish_reason=response.get("finish_reason"),
        latency_ms=response.get("latency_ms"),
    )


def _assertion_pairs(
    traces: list[TraceRecord],
    baseline_calls: list[StoredCall],
    candidate_calls: list[StoredCall],
) -> list[AssertionPair]:
    baseline_by_trace = {call.trace_id: call for call in baseline_calls if call.sample_index == 0}
    candidate_by_trace = {call.trace_id: call for call in candidate_calls if call.sample_index == 0}
    pairs: list[AssertionPair] = []
    for trace in sorted(traces, key=lambda item: str(item.trace_id)):
        trace_id = str(trace.trace_id)
        if trace_id not in baseline_by_trace or trace_id not in candidate_by_trace:
            continue
        pairs.append(
            AssertionPair(
                trace_id=trace_id,
                baseline=_stored_diff_response(baseline_by_trace[trace_id]),
                candidate=_stored_diff_response(candidate_by_trace[trace_id]),
            )
        )
    return pairs


def _write_assertion_results(
    path: Path,
    traces: list[TraceRecord],
    baseline_calls: list[StoredCall],
    candidate_calls: list[StoredCall],
    manifest_model,
) -> tuple[str, int]:
    pairs = _assertion_pairs(traces, baseline_calls, candidate_calls)
    evaluations = evaluate_assertions(
        manifest_model.assertions,
        pairs,
        max_failures=len(pairs),
    )
    rows: list[dict[str, object]] = []
    for evaluation in evaluations:
        for trace_id in evaluation.failing_trace_ids:
            rows.append(
                {
                    "assertion_id": evaluation.assertion_id,
                    "baseline_passed": True,
                    "candidate_passed": False,
                    "pair_id": trace_id,
                    "severity": evaluation.severity.value,
                    "trace_id": trace_id,
                }
            )
    return _write_jsonl(path, rows), len(rows)


def _write_json(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{content}\n")
    digest = hashlib.sha256(f"{content}\n".encode()).hexdigest()
    with Path(f"{path}.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}\n")
    return digest


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows]
    content = "".join(f"{line}\n" for line in lines)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    digest = hashlib.sha256(content.encode()).hexdigest()
    with Path(f"{path}.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}\n")
    return digest


def _load_run_state(run_path: Path, run_id: str) -> tuple[str, dict[str, dict[str, object]]]:
    if run_path.exists():
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        started_at = str(payload.get("run_started_at") or _utc_now())
        stages = {
            str(stage): dict(value)
            for stage, value in (payload.get("stages") or {}).items()
        }
        return started_at, stages
    return _utc_now(), {}


def _mark_stage(
    run_path: Path,
    *,
    run_id: str,
    run_started_at: str,
    stages: dict[str, dict[str, object]],
    stage: str,
    status: str,
    **details: object,
) -> None:
    stages[stage] = {"status": status, **details}
    payload: dict[str, object] = {}
    if run_path.exists():
        payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload["run_id"] = run_id
    payload["run_started_at"] = run_started_at
    payload["stages"] = stages
    _write_json(run_path, payload)


def _stage_finished(stages: dict[str, dict[str, object]], stage: str) -> bool:
    return stages.get(stage, {}).get("status") in {"completed", "skipped"}


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _candidate_regression_rules(
    traces: list[TraceRecord],
    *,
    broken_json_rate: float,
    broken_json_token: str | None,
) -> list[RegressionRule]:
    broken_ids: set[str] = set()
    if broken_json_rate > 0:
        count = max(1, round(len(traces) * broken_json_rate))
        broken_ids.update(str(trace.trace_id) for trace in traces[:count])
    if broken_json_token:
        broken_ids.update(
            str(trace.trace_id)
            for trace in traces
            if broken_json_token in trace.request.messages[0].content
        )
    if not broken_ids:
        return []

    def predicate(prompt: str) -> bool:
        return any(
            trace.request.messages[0].content in prompt
            for trace in traces
            if str(trace.trace_id) in broken_ids
        )

    return [
        RegressionRule(
            predicate=predicate,
            transform=lambda _prompt, _text: "{broken",
        )
    ]


def _votes_from_judgments(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    votes: dict[str, dict[str, object]] = {}
    for row in rows:
        votes.setdefault(str(row["pair_id"]), {})[str(row["judge"])] = row["outcome"]
    return votes


def _load_judgment_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_assertion_failures(path: Path) -> dict[str, list[AssertionFailure]]:
    if not path.exists():
        return {}
    failures: dict[str, list[AssertionFailure]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            pair_id = str(payload.get("pair_id") or payload.get("trace_id"))
            failures.setdefault(pair_id, []).append(
                AssertionFailure(
                    assertion_id=str(payload["assertion_id"]),
                    severity=Severity(str(payload["severity"])),
                    baseline_passed=bool(payload.get("baseline_passed", True)),
                    candidate_passed=bool(payload.get("candidate_passed", False)),
                )
            )
    return failures


def _text_by_trace(calls: list[StoredCall]) -> dict[str, str]:
    values: dict[str, str] = {}
    for call in calls:
        response = call.response or {}
        values[call.trace_id] = str(response.get("text") or "")
    return values


def _candidate_records_for_clustering(
    traces: list[TraceRecord],
    baseline_calls: list[StoredCall],
    candidate_calls: list[StoredCall],
    judgment_rows: list[dict[str, object]],
    assertion_failures: dict[str, list[AssertionFailure]],
) -> list[CandidatePairRecord]:
    votes = _votes_from_judgments(judgment_rows)
    outcomes = aggregate_pair_outcomes(votes)
    baseline_text = _text_by_trace(baseline_calls)
    candidate_text = _text_by_trace(candidate_calls)
    records: list[CandidatePairRecord] = []
    for trace in sorted(traces, key=lambda item: str(item.trace_id)):
        trace_id = str(trace.trace_id)
        if trace_id not in baseline_text or trace_id not in candidate_text:
            continue
        records.append(
            CandidatePairRecord(
                pair_id=trace_id,
                prompt=trace.request.messages[0].content,
                baseline_response=baseline_text[trace_id],
                candidate_response=candidate_text[trace_id],
                ensemble_outcome=outcomes.get(trace_id, "equivalent"),
                judge_reasons=[
                    str(row.get("reason"))
                    for row in judgment_rows
                    if row.get("pair_id") == trace_id and row.get("reason")
                ],
                assertion_failures=assertion_failures.get(trace_id, []),
                template_id=trace.meta.template_id,
                system=trace.request.system,
            )
        )
    return records


@app.command()
def run(
    manifest: ManifestOption = Path("acsi.yaml"),
    traces: Annotated[
        Path | None,
        typer.Option("--traces", help="Normalized TraceRecord JSONL path."),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Resume or create this run id."),
    ] = None,
    run_dir: RunDirOption = Path(".acsi"),
    fake_noise: Annotated[
        float,
        typer.Option("--fake-noise", help="FakeClient noise rate for baseline and candidate."),
    ] = 0.0,
    inject_broken_json_rate: Annotated[
        float,
        typer.Option("--inject-broken-json-rate", help="Break this fraction of sampled prompts."),
    ] = 0.0,
    inject_broken_json_token: Annotated[
        str | None,
        typer.Option("--inject-broken-json-token", help="Break sampled prompts containing token."),
    ] = None,
    interrupt_after_judge_dispatches: Annotated[
        int | None,
        typer.Option(
            "--interrupt-after-judge-dispatches",
            help="Testing hook: interrupt during judge dispatch.",
            hidden=True,
        ),
    ] = None,
    degraded: Annotated[
        bool,
        typer.Option("--degraded", help="Run baseline in degraded mode."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Approve provider spend without prompting."),
    ] = False,
    json_output: JsonOutputOption = False,
) -> None:
    if not yes:
        _fail("Pass --yes to approve the full run preflight.", json_output)
    try:
        manifest_model = load_workload_manifest(manifest)
        source_traces_path = traces or run_dir / "traces" / f"{manifest_model.workload}.jsonl"
        source_records = import_jsonl_paths([source_traces_path]).records
        if not source_records:
            _fail(f"No valid traces found in {source_traces_path}.", json_output)

        active_run_id = run_id or str(uuid4())
        active_run_dir = run_dir / "runs" / active_run_id
        active_run_dir.mkdir(parents=True, exist_ok=True)
        run_path = active_run_dir / "run.json"
        run_started_at, stages = _load_run_state(run_path, active_run_id)

        table = Table(title="ACSI Run Preflight")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Traces", str(len(source_records)))
        table.add_row("Sample target", str(manifest_model.sampling.n))
        table.add_row(
            "Baseline",
            f"{manifest_model.baseline.provider}/{manifest_model.baseline.model}",
        )
        table.add_row(
            "Candidate",
            f"{manifest_model.candidate.provider}/{manifest_model.candidate.model}",
        )
        table.add_row("Estimated cost", "$0.000000 (fake clients)")
        if not json_output:
            console.print(table)

        if not _stage_finished(stages, "import-check"):
            _mark_stage(
                run_path,
                run_id=active_run_id,
                run_started_at=run_started_at,
                stages=stages,
                stage="import-check",
                status="completed",
                traces=len(source_records),
            )

        scrubbed_path = active_run_dir / "scrubbed_traces.jsonl"
        if manifest_model.privacy.scrub:
            if _stage_finished(stages, "scrub") and scrubbed_path.exists():
                scrubbed_records = import_jsonl_paths([scrubbed_path]).records
            else:
                scrub_result = scrub_traces(source_records)
                write_scrub_artifacts(
                    scrub_result,
                    traces_path=scrubbed_path,
                    report_path=active_run_dir / "scrub_report.json",
                )
                scrubbed_records = scrub_result.records
                _mark_stage(
                    run_path,
                    run_id=active_run_id,
                    run_started_at=run_started_at,
                    stages=stages,
                    stage="scrub",
                    status="completed",
                    counts=scrub_result.report.get("counts", {}),
                )
        else:
            scrubbed_records = source_records
            if not _stage_finished(stages, "scrub"):
                _mark_stage(
                    run_path,
                    run_id=active_run_id,
                    run_started_at=run_started_at,
                    stages=stages,
                    stage="scrub",
                    status="skipped",
                    reason="privacy.scrub=false",
                )

        sampled_path = active_run_dir / "sampled_traces.jsonl"
        if _stage_finished(stages, "sample") and sampled_path.exists():
            sampled_records = import_jsonl_paths([sampled_path]).records
        else:
            sampling_result = sample_traces(scrubbed_records, manifest_model.sampling)
            write_sample_artifacts(
                sampling_result.records,
                output_path=sampled_path,
                report_path=active_run_dir / "sampling_report.json",
                report=sampling_result.report,
            )
            sampled_records = sampling_result.records
            _mark_stage(
                run_path,
                run_id=active_run_id,
                run_started_at=run_started_at,
                stages=stages,
                stage="sample",
                status="completed",
                mode=sampling_result.sampling_mode,
                n=len(sampled_records),
            )

        store = ReplayStore(active_run_dir / "replay.sqlite")
        if not _stage_finished(stages, "baseline"):
            baseline_result = asyncio.run(
                run_baseline_stage(
                    sampled_records,
                    manifest_model.baseline,
                    manifest_model.sampling.k_baseline,
                    client=FakeClient(seed=manifest_model.sampling.seed, noise=fake_noise),
                    store=store,
                    config=ReplayConfig(
                        run_id=active_run_id,
                        phase="baseline",
                        seed=manifest_model.sampling.seed,
                        concurrency=4,
                        max_cost_usd=manifest_model.budget.max_usd,
                    ),
                    run_dir=active_run_dir,
                    manifest_path=manifest,
                    traces_path=sampled_path,
                    endpoint="degraded" if degraded else "fake",
                    degraded=degraded,
                )
            )
            write_responses_jsonl(
                store,
                active_run_id,
                active_run_dir / "baseline" / "responses.jsonl",
                phase="baseline",
            )
            _mark_stage(
                run_path,
                run_id=active_run_id,
                run_started_at=run_started_at,
                stages=stages,
                stage="baseline",
                status="completed",
                degraded=degraded,
                dispatched=baseline_result.replay_result.dispatched,
            )

        if not _stage_finished(stages, "replay"):
            candidate_client = FakeClient(
                seed=manifest_model.sampling.seed,
                noise=fake_noise,
                regressions=_candidate_regression_rules(
                    sampled_records,
                    broken_json_rate=inject_broken_json_rate,
                    broken_json_token=inject_broken_json_token,
                ),
            )
            clock = RunClock()
            replay_result = asyncio.run(
                replay_traces(
                    sampled_records,
                    manifest_model.candidate,
                    1,
                    client=candidate_client,
                    store=store,
                    config=ReplayConfig(
                        run_id=active_run_id,
                        phase="candidate",
                        seed=manifest_model.sampling.seed,
                        concurrency=4,
                        max_cost_usd=manifest_model.budget.max_usd,
                    ),
                )
            )
            write_responses_jsonl(
                store,
                active_run_id,
                active_run_dir / "candidate" / "responses.jsonl",
                phase="candidate",
            )
            run_manifest = build_run_manifest(
                run_id=active_run_id,
                manifest_path=manifest,
                traces_path=sampled_path,
                seed=manifest_model.sampling.seed,
                provider=manifest_model.candidate.provider,
                endpoint="fake",
                store=store,
                result=replay_result,
                wall_clock_seconds=clock.elapsed_seconds(),
                degraded=degraded,
                phase="candidate",
                run_started_at=run_started_at,
                stages=stages,
            )
            write_run_manifest(run_path, run_manifest)
            _mark_stage(
                run_path,
                run_id=active_run_id,
                run_started_at=run_started_at,
                stages=stages,
                stage="replay",
                status="completed",
                dispatched=replay_result.dispatched,
            )

        baseline_calls = _load_response_calls(active_run_dir / "baseline" / "responses.jsonl")
        candidate_calls = _load_response_calls(active_run_dir / "candidate" / "responses.jsonl")

        if not _stage_finished(stages, "diff"):
            assertion_hash, assertion_failures = _write_assertion_results(
                active_run_dir / "assertion_results.jsonl",
                sampled_records,
                baseline_calls,
                candidate_calls,
                manifest_model,
            )
            _mark_stage(
                run_path,
                run_id=active_run_id,
                run_started_at=run_started_at,
                stages=stages,
                stage="diff",
                status="completed",
                assertion_failures=assertion_failures,
                sha256=assertion_hash,
            )

        if not _stage_finished(stages, "judge"):
            tau = _load_noise_tau(active_run_dir)
            pairs = build_candidate_pairs(
                sampled_records,
                baseline_calls,
                candidate_calls,
                tau=tau,
            )
            selected = select_for_judging(pairs, tau)
            panel = select_judge_panel(manifest_model)
            clients = {
                judge_spec.model: FakeJudge(model=judge_spec.model)
                for judge_spec in panel
            }
            judge_result = run_pairwise_judging(
                selected,
                clients,
                store=store,
                config=JudgeRunConfig(
                    run_id=active_run_id,
                    seed=manifest_model.sampling.seed,
                    interrupt_after_dispatches=interrupt_after_judge_dispatches,
                ),
            )
            judgments_hash, stats_hash = write_judge_artifacts(
                active_run_dir,
                judge_result,
            )
            _mark_stage(
                run_path,
                run_id=active_run_id,
                run_started_at=run_started_at,
                stages=stages,
                stage="judge",
                status="completed",
                selected_pairs=len(selected),
                judgments_sha256=judgments_hash,
                stats_sha256=stats_hash,
            )

        if not _stage_finished(stages, "cluster"):
            judgments = _load_judgment_rows(active_run_dir / "judgments.jsonl")
            assertion_failures = _load_assertion_failures(
                active_run_dir / "assertion_results.jsonl"
            )
            records = _candidate_records_for_clustering(
                sampled_records,
                baseline_calls,
                candidate_calls,
                judgments,
                assertion_failures,
            )
            regressions = build_regression_set(records)
            buckets = cluster_regressions(
                regressions,
                n_sampled_pairs=len(sampled_records),
                min_cluster_size=manifest_model.clustering.min_cluster_size,
            )
            named, stats = name_clusters(
                buckets,
                namer=FakeNamer(),
                store=store,
                run_id=active_run_id,
            )
            clusters_hash = write_clusters_json(
                active_run_dir / "clusters.json",
                named,
                stats=stats,
            )
            detection = detect_templates(sampled_records)
            target = select_patch_target(sampled_records, detection)
            patch_reports: list[PatchReport] = []
            patches_dir = active_run_dir / "patches"
            for bucket in named:
                proposal, _patch_stats = propose_patch(
                    cluster=bucket,
                    regressions=regressions,
                    target=target,
                    patcher=FakePatcher(),
                    store=store,
                    run_id=active_run_id,
                )
                if proposal is None:
                    continue
                diff_path = patches_dir / f"patch_{bucket.cluster_id}.diff"
                diff_path.parent.mkdir(parents=True, exist_ok=True)
                with diff_path.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write(proposal.diff_text)
                patch_reports.append(
                    PatchReport(
                        cluster_id=bucket.cluster_id,
                        diff_path=str(diff_path),
                        fixed_fraction=0.0,
                        control_regressions=0,
                        accepted=False,
                        reason="not_validated_orchestrator",
                    )
                )
            patch_hash = write_patch_report(patches_dir / "patch_report.json", patch_reports)
            _mark_stage(
                run_path,
                run_id=active_run_id,
                run_started_at=run_started_at,
                stages=stages,
                stage="cluster",
                status="completed",
                clusters=len(named),
                clusters_sha256=clusters_hash,
                patches_sha256=patch_hash,
            )

        cert_result = None
        if not _stage_finished(stages, "cert"):
            cert_result = build_certificate(
                manifest=manifest_model,
                traces=sampled_records,
                run_dir=active_run_dir,
                manifest_path=manifest,
                degraded=degraded,
            )
            report_hash = render_report(
                cert_result.cert,
                output_path=active_run_dir / "report.html",
            )
            _mark_stage(
                run_path,
                run_id=active_run_id,
                run_started_at=run_started_at,
                stages=stages,
                stage="verdict",
                status="completed",
                verdict=cert_result.payload["verdict"],
            )
            _mark_stage(
                run_path,
                run_id=active_run_id,
                run_started_at=run_started_at,
                stages=stages,
                stage="cert",
                status="completed",
                cert_sha256=cert_result.cert_sha256,
                report_sha256=report_hash,
            )
        else:
            cert_payload = json.loads(
                (active_run_dir / "cert.json").read_text(encoding="utf-8")
            )
            cert_result = type(
                "CertResult",
                (),
                {
                    "cert_sha256": (active_run_dir / "cert.json.sha256")
                    .read_text(encoding="utf-8")
                    .strip(),
                    "payload": cert_payload["payload"],
                },
            )()
    except (
        BannedLanguageError,
        CertificateVerificationError,
        JudgeInterrupted,
        OSError,
        ReplayAbortError,
        ReplayInterrupted,
        ValueError,
    ) as exc:
        _fail(str(exc), json_output)

    payload = {
        "status": "ok",
        "run_id": active_run_id,
        "run_dir": str(active_run_dir),
        "verdict": cert_result.payload["verdict"],
        "cert_path": str(active_run_dir / "cert.json"),
        "cert_sha256": cert_result.cert_sha256,
        "report_path": str(active_run_dir / "report.html"),
    }
    if json_output:
        console.print_json(data=payload)
    else:
        console.print_json(data=payload)


@app.command()
def baseline(
    manifest: ManifestOption = Path("acsi.yaml"),
    traces: Annotated[
        Path | None,
        typer.Option("--traces", help="Normalized TraceRecord JSONL path."),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Resume or create this run id."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Baseline replay seed.")] = 42,
    concurrency: Annotated[int, typer.Option("--concurrency", help="Replay concurrency.")] = 4,
    max_cost: Annotated[
        float | None,
        typer.Option("--max-cost", help="Maximum baseline replay spend in USD."),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Approve baseline execution.")] = False,
    live: Annotated[
        bool,
        typer.Option("--live", help="Use LiveClient instead of FakeClient."),
    ] = False,
    fake_noise: Annotated[float, typer.Option("--fake-noise", help="FakeClient noise rate.")] = 0.0,
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
    try:
        manifest_model = load_workload_manifest(manifest)
        baseline_model = manifest_model.baseline
        traces_path = traces or run_dir / "traces" / f"{manifest_model.workload}.jsonl"
        trace_records = import_jsonl_paths([traces_path]).records
        if not trace_records:
            _fail(f"No valid traces found in {traces_path}.", json_output)

        active_run_id = run_id or str(uuid4())
        active_run_dir = run_dir / "runs" / active_run_id
        max_cost_usd = max_cost if max_cost is not None else manifest_model.budget.max_usd
        if not degraded:
            estimate = _estimate_replay_cost(
                trace_records,
                baseline_model,
                manifest_model.sampling.k_baseline,
                fake=not live,
            )
            if not yes:
                if json_output:
                    _fail("Pass --yes to approve baseline execution.", json_output)
                _confirm_replay(trace_records, baseline_model, estimate)

        client = LiveClient() if live else FakeClient(seed=seed, noise=fake_noise)
        result = asyncio.run(
            run_baseline_stage(
                trace_records,
                baseline_model,
                manifest_model.sampling.k_baseline,
                client=client,
                store=ReplayStore(active_run_dir / "replay.sqlite"),
                config=ReplayConfig(
                    run_id=active_run_id,
                    phase="baseline",
                    seed=seed,
                    concurrency=concurrency,
                    max_cost_usd=max_cost_usd,
                    resume_command=_baseline_resume_command(
                        manifest,
                        traces_path,
                        active_run_id,
                    ),
                ),
                run_dir=active_run_dir,
                manifest_path=manifest,
                traces_path=traces_path,
                endpoint="degraded" if degraded else ("live" if live else "fake"),
                degraded=degraded,
            )
        )
    except ReplayAbortError as exc:
        _fail(str(exc), json_output)
    except ReplayInterrupted as exc:
        _fail(str(exc), json_output)
    except (OSError, ValueError) as exc:
        _fail(str(exc), json_output)

    replay_result = result.replay_result
    payload = {
        "status": "ok" if not replay_result.halted_reason else "halted",
        "run_id": active_run_id,
        "run_dir": str(active_run_dir),
        "completed": replay_result.completed,
        "errors": replay_result.errors,
        "cache_hits": replay_result.cache_hits,
        "dispatched": replay_result.dispatched,
        "retry_count": replay_result.retry_count,
        "cost_usd": replay_result.cost_usd,
        "halted_reason": replay_result.halted_reason,
        "degraded": degraded,
        "threshold_source": result.noise_floor.get("threshold_source"),
        "textual_mismatch_rate": result.noise_floor.get("textual_mismatch_rate"),
        "beyond_noise_rate": result.noise_floor.get("beyond_noise_rate"),
        "beyond_noise_to_textual_mismatch_rate": result.noise_floor.get(
            "beyond_noise_to_textual_mismatch_rate"
        ),
        "noise_floor_path": str(active_run_dir / "baseline" / "noise_floor.json"),
        "noise_floor_sha256": result.noise_floor_sha256,
        "responses_sha256": result.responses_sha256,
        "run_sha256": result.run_sha256,
    }
    if json_output:
        console.print_json(data=payload)
    else:
        console.print(_baseline_summary_table(payload))


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
def cluster(
    run_id: Annotated[
        str | None,
        typer.Option("--run", help="Run id under --run-dir/runs to cluster."),
    ] = None,
    manifest: ManifestOption = Path("acsi.yaml"),
    traces: Annotated[
        Path | None,
        typer.Option("--traces", help="Normalized TraceRecord JSONL path."),
    ] = None,
    run_dir: RunDirOption = Path(".acsi"),
    propose_patches: Annotated[
        bool,
        typer.Option("--propose-patches", help="Propose patches for all clusters."),
    ] = False,
    json_output: JsonOutputOption = False,
) -> None:
    if run_id is None:
        _fail("Pass --run with the run id to cluster.", json_output)
    try:
        manifest_model = load_workload_manifest(manifest)
        traces_path = traces or run_dir / "traces" / f"{manifest_model.workload}.jsonl"
        trace_records = import_jsonl_paths([traces_path]).records
        active_run_dir = run_dir / "runs" / run_id
        baseline_calls = _load_response_calls(
            _first_existing(_baseline_response_paths(active_run_dir))
        )
        candidate_calls = _load_response_calls(
            _first_existing(_candidate_response_paths(active_run_dir))
        )
        judgments = _load_judgment_rows(active_run_dir / "judgments.jsonl")
        assertion_failures = _load_assertion_failures(
            active_run_dir / "assertion_results.jsonl"
        )
        records = _candidate_records_for_clustering(
            trace_records,
            baseline_calls,
            candidate_calls,
            judgments,
            assertion_failures,
        )
        regressions = build_regression_set(records)
        buckets = cluster_regressions(
            regressions,
            n_sampled_pairs=len(trace_records),
            min_cluster_size=manifest_model.clustering.min_cluster_size,
        )
        named, stats = name_clusters(
            buckets,
            namer=FakeNamer(),
            store=ReplayStore(active_run_dir / "replay.sqlite"),
            run_id=run_id,
        )
        clusters_hash = write_clusters_json(
            active_run_dir / "clusters.json",
            named,
            stats=stats,
        )
        patch_hash = None
        if propose_patches:
            detection = detect_templates(trace_records)
            target = select_patch_target(trace_records, detection)
            reports: list[PatchReport] = []
            patches_dir = active_run_dir / "patches"
            for bucket in named:
                proposal, _patch_stats = propose_patch(
                    cluster=bucket,
                    regressions=regressions,
                    target=target,
                    patcher=FakePatcher(),
                    store=ReplayStore(active_run_dir / "replay.sqlite"),
                    run_id=run_id,
                )
                if proposal is None:
                    continue
                diff_path = patches_dir / f"patch_{bucket.cluster_id}.diff"
                diff_path.parent.mkdir(parents=True, exist_ok=True)
                with diff_path.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write(proposal.diff_text)
                reports.append(
                    PatchReport(
                        cluster_id=bucket.cluster_id,
                        diff_path=str(diff_path),
                        fixed_fraction=0.0,
                        control_regressions=0,
                        accepted=False,
                        reason="not_validated_cli",
                    )
                )
            patch_hash = write_patch_report(patches_dir / "patch_report.json", reports)
    except (OSError, ValueError) as exc:
        _fail(str(exc), json_output)

    payload = {
        "status": "ok",
        "run_id": run_id,
        "run_dir": str(active_run_dir),
        "regression_count": len(regressions),
        "cluster_count": len(named),
        "clusters_sha256": clusters_hash,
        "patch_report_sha256": patch_hash,
    }
    if json_output:
        console.print_json(data=payload)
    else:
        console.print_json(data=payload)


@app.command()
def judge(
    run_id: Annotated[
        str | None,
        typer.Option("--run", help="Run id under --run-dir/runs to judge."),
    ] = None,
    manifest: ManifestOption = Path("acsi.yaml"),
    traces: Annotated[
        Path | None,
        typer.Option("--traces", help="Normalized TraceRecord JSONL path."),
    ] = None,
    run_dir: RunDirOption = Path(".acsi"),
    export_calibration_sample: Annotated[
        int | None,
        typer.Option(
            "--export-calibration-sample",
            help="Write N selected judged pairs for human labels and exit.",
        ),
    ] = None,
    calibration_csv: Annotated[
        Path | None,
        typer.Option("--calibration-csv", help="Human labels CSV to fold into judge_stats.json."),
    ] = None,
    fake: Annotated[
        bool,
        typer.Option("--fake/--live", help="Use FakeJudge clients instead of LiveJudge."),
    ] = True,
    json_output: JsonOutputOption = False,
) -> None:
    if run_id is None:
        _fail("Pass --run with the run id to judge.", json_output)
    try:
        manifest_model = load_workload_manifest(manifest)
        traces_path = traces or run_dir / "traces" / f"{manifest_model.workload}.jsonl"
        trace_records = import_jsonl_paths([traces_path]).records
        active_run_dir = run_dir / "runs" / run_id
        tau = _load_noise_tau(active_run_dir)
        baseline_calls = _load_response_calls(
            _first_existing(_baseline_response_paths(active_run_dir))
        )
        candidate_calls = _load_response_calls(
            _first_existing(_candidate_response_paths(active_run_dir))
        )
        pairs = build_candidate_pairs(
            trace_records,
            baseline_calls,
            candidate_calls,
            tau=tau,
        )
        selected = select_for_judging(pairs, tau)
        if export_calibration_sample is not None:
            output = active_run_dir / "calibration_sample.csv"
            write_calibration_sample(
                output,
                [
                    CalibrationSample(
                        pair.pair_id,
                        pair.prompt,
                        pair.baseline.text or "",
                        pair.candidate.text or "",
                    )
                    for pair in selected
                ],
                export_calibration_sample,
            )
            payload = {"status": "ok", "path": str(output), "rows": export_calibration_sample}
            if json_output:
                console.print_json(data=payload)
            else:
                console.print(f"Wrote {output}")
            return

        panel = select_judge_panel(manifest_model)
        clients = {
            judge_spec.model: FakeJudge(model=judge_spec.model)
            if fake
            else LiveJudge(judge_spec.model)
            for judge_spec in panel
        }
        result = run_pairwise_judging(
            selected,
            clients,
            store=ReplayStore(active_run_dir / "replay.sqlite"),
            config=JudgeRunConfig(run_id=run_id, seed=manifest_model.sampling.seed),
        )
        calibration = None
        if calibration_csv is not None:
            calibration = ingest_calibration_csv(
                calibration_csv,
                _votes_from_judgments(result.judgments),
            )
        judgments_hash, stats_hash = write_judge_artifacts(
            active_run_dir,
            result,
            calibration=calibration,
        )
    except (OSError, ValueError) as exc:
        _fail(str(exc), json_output)

    payload = {
        "status": "ok",
        "run_id": run_id,
        "run_dir": str(active_run_dir),
        "selected_pairs": len(selected),
        "judgments_sha256": judgments_hash,
        "judge_stats_sha256": stats_hash,
    }
    if json_output:
        console.print_json(data=payload)
    else:
        console.print_json(data=payload)


@app.command()
def cert(
    run_id: Annotated[
        str | None,
        typer.Option("--run", help="Run id under --run-dir/runs to certify."),
    ] = None,
    manifest: ManifestOption = Path("acsi.yaml"),
    traces: Annotated[
        Path | None,
        typer.Option("--traces", help="Sampled TraceRecord JSONL path."),
    ] = None,
    run_dir: RunDirOption = Path(".acsi"),
    json_output: JsonOutputOption = False,
) -> None:
    if run_id is None:
        _fail("Pass --run with the run id to certify.", json_output)
    try:
        manifest_model = load_workload_manifest(manifest)
        active_run_dir = run_dir / "runs" / run_id
        traces_path = traces or active_run_dir / "sampled_traces.jsonl"
        trace_records = import_jsonl_paths([traces_path]).records
        result = build_certificate(
            manifest=manifest_model,
            traces=trace_records,
            run_dir=active_run_dir,
            manifest_path=manifest,
        )
        report_hash = render_report(result.cert, output_path=active_run_dir / "report.html")
    except (BannedLanguageError, OSError, ValueError) as exc:
        _fail(str(exc), json_output)

    payload = {
        "status": "ok",
        "run_id": run_id,
        "run_dir": str(active_run_dir),
        "verdict": result.payload["verdict"],
        "cert_sha256": result.cert_sha256,
        "report_sha256": report_hash,
        "key_generated": result.key_generated,
    }
    if json_output:
        console.print_json(data=payload)
    else:
        if result.key_generated:
            console.print("Generated .acsi/keys/ed25519.key for certificate signing.")
        console.print_json(data=payload)


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
    try:
        cert_payload = verify_certificate(cert_path)
    except (CertificateVerificationError, OSError, ValueError) as exc:
        _fail(str(exc), json_output)
    payload = {
        "status": "ok",
        "path": str(cert_path),
        "verdict": cert_payload["payload"].get("verdict"),
    }
    if json_output:
        console.print_json(data=payload)
    else:
        console.print("Certificate signature verified.")


@app.command()
def publish(
    run_id: Annotated[
        str | None,
        typer.Option("--run", help="Run id under --run-dir/runs to publish."),
    ] = None,
    cert_path: Annotated[
        Path | None,
        typer.Option("--cert", help="Explicit cert.json path."),
    ] = None,
    run_dir: RunDirOption = Path(".acsi"),
    url: Annotated[
        str | None,
        typer.Option("--url", "--endpoint", help="Explicit publish endpoint for verdict JSON."),
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
    if cert_path is None:
        if run_id is None:
            _fail("Pass --run or --cert to publish a certificate.", json_output)
        cert_path = run_dir / "runs" / run_id / "cert.json"
    try:
        result = publish_certificate(
            cert_path,
            url=url,
            include_examples=include_examples,
        )
    except (PublishError, OSError, ValueError) as exc:
        _fail(str(exc), json_output)
    payload = {
        "status": "ok",
        "status_code": result.status_code,
        "published_keys": sorted(result.payload.keys()),
    }
    if json_output:
        console.print_json(data=payload)
    else:
        console.print_json(data=payload)


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
