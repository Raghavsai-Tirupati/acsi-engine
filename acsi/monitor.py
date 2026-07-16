from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from acsi.diff.assertions import AssertionPair, evaluate_assertions
from acsi.diff.deterministic import DiffResponse
from acsi.diff.semantic import FakeEmbedder, classify_pair
from acsi.replay.artifacts import sha256_file
from acsi.replay.clients import CompletionClient, FakeClient
from acsi.replay.runner import (
    ReplayAbortError,
    ReplayConfig,
    ReplayInterrupted,
    build_completion_request,
    replay,
    write_responses_jsonl,
)
from acsi.replay.store import ReplayStore, StoredCall
from acsi.schemas import ProviderModel, Severity, TraceRecord, WorkloadManifest


@dataclass(frozen=True)
class MonitorInitResult:
    workload: str
    suite_size: int
    suite_hash: str
    golden_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class MonitorRunResult:
    exit_code: int
    summary: dict[str, Any]
    summary_path: Path | None
    summary_sha256: str | None
    cache_hits: int = 0
    dispatched: int = 0
    operational_message: str | None = None


def init_monitor(
    *,
    manifest: WorkloadManifest,
    run_id: str,
    run_dir: Path = Path(".acsi"),
    cert_run_dir: Path | None = None,
) -> MonitorInitResult:
    source_run_dir = cert_run_dir or run_dir / "runs" / run_id
    traces = _read_traces(source_run_dir / "sampled_traces.jsonl")
    baseline_calls = _read_jsonl(source_run_dir / "baseline" / "responses.jsonl")
    baseline_by_trace = {
        str(row["trace_id"]): row
        for row in baseline_calls
        if int(row.get("sample_index", 0)) == 0
    }
    assertion_trace_ids = _assertion_trace_ids(source_run_dir / "assertion_results.jsonl")
    selected = _select_suite(traces, assertion_trace_ids, manifest.monitor.suite_size)
    golden_rows = []
    for trace in selected:
        trace_id = str(trace.trace_id)
        baseline = baseline_by_trace.get(trace_id)
        if baseline is None:
            raise ValueError(f"Missing baseline output for trace {trace_id}.")
        request, _prompt_hash, _params_hash, _transforms = build_completion_request(
            trace,
            manifest.baseline,
            0,
        )
        golden_rows.append(
            {
                "baseline_response": baseline["response"],
                "post_mapping_params": request.params,
                "trace": trace.model_dump(mode="json"),
                "trace_id": trace_id,
            }
        )

    monitor_dir = _monitor_dir(run_dir, manifest.workload)
    golden_path = monitor_dir / "golden.jsonl"
    suite_hash = _write_jsonl(golden_path, golden_rows)
    cert_payload = _read_json(source_run_dir / "cert.json")["payload"]
    noise_floor = cert_payload.get("noise_floor_raw") or {}
    golden_manifest = {
        "created_at": _now(),
        "init_run_id": run_id,
        "pinned_model": manifest.baseline.model_dump(mode="json"),
        "post_mapping_params_hash": _params_hash_for_rows(golden_rows),
        "stored_noise_floor_ci": noise_floor.get("beyond_noise_ci"),
        "suite_content_hash": suite_hash,
        "suite_size": len(golden_rows),
        "tau": float(noise_floor.get("tau", 0.9)),
        "threshold_source": noise_floor.get("threshold_source", "unknown"),
        "workload": manifest.workload,
    }
    manifest_path = monitor_dir / "golden_manifest.json"
    _write_json(manifest_path, golden_manifest)
    return MonitorInitResult(
        workload=manifest.workload,
        suite_size=len(golden_rows),
        suite_hash=suite_hash,
        golden_path=golden_path,
        manifest_path=manifest_path,
    )


def run_monitor(
    *,
    manifest: WorkloadManifest,
    run_dir: Path = Path(".acsi"),
    client: CompletionClient | None = None,
    run_id: str | None = None,
    interrupt_after_dispatches: int | None = None,
) -> MonitorRunResult:
    monitor_dir = _monitor_dir(run_dir, manifest.workload)
    golden_manifest = _read_json(monitor_dir / "golden_manifest.json")
    golden_rows = _read_jsonl(monitor_dir / "golden.jsonl")
    if not golden_rows:
        raise ValueError("Monitor suite is empty; run `acsi monitor init` first.")

    active_run_id = run_id or f"monitor-{_timestamp_slug()}"
    active_run_dir = monitor_dir / "runs" / active_run_id
    traces = [
        TraceRecord.model_validate(row["trace"])
        for row in golden_rows
    ]
    pinned_model = ProviderModel.model_validate(golden_manifest["pinned_model"])
    active_client = client or FakeClient(seed=manifest.sampling.seed)
    store = ReplayStore(active_run_dir / "monitor.sqlite")
    try:
        result = asyncio.run(
            replay(
                traces,
                pinned_model,
                1,
                client=active_client,
                store=store,
                config=ReplayConfig(
                    run_id=active_run_id,
                    phase="monitor",
                    seed=manifest.sampling.seed,
                    concurrency=1,
                    max_cost_usd=manifest.budget.max_usd,
                    interrupt_after_dispatches=interrupt_after_dispatches,
                ),
            )
        )
    except ReplayInterrupted:
        raise
    except ReplayAbortError as exc:
        return MonitorRunResult(
            exit_code=1,
            summary={},
            summary_path=None,
            summary_sha256=None,
            operational_message=f"monitor operational failure: {exc}",
        )

    write_responses_jsonl(
        store,
        active_run_id,
        active_run_dir / "responses.jsonl",
        phase="monitor",
    )
    current_calls = {
        call.trace_id: call
        for call in store.done_calls(active_run_id, phase="monitor")
        if call.sample_index == 0
    }
    summary = _monitor_summary(
        manifest=manifest,
        golden_manifest=golden_manifest,
        golden_rows=golden_rows,
        current_calls=current_calls,
    )
    summary_path = active_run_dir / "summary.json"
    summary_hash = _write_json(summary_path, summary)
    exit_code = 2 if summary["drift"] else 0
    return MonitorRunResult(
        exit_code=exit_code,
        summary=summary,
        summary_path=summary_path,
        summary_sha256=summary_hash,
        cache_hits=result.cache_hits,
        dispatched=result.dispatched,
    )


def _monitor_summary(
    *,
    manifest: WorkloadManifest,
    golden_manifest: dict[str, Any],
    golden_rows: list[dict[str, Any]],
    current_calls: dict[str, StoredCall],
) -> dict[str, Any]:
    tau = float(golden_manifest.get("tau", 0.9))
    stored_ci = golden_manifest.get("stored_noise_floor_ci") or {}
    stored_bound = float(stored_ci.get("upper", 0.0))
    epsilon = manifest.monitor.epsilon_pp / 100
    classifications = []
    assertion_pairs: list[AssertionPair] = []
    served_models = sorted(
        {
            str(call.served_model)
            for call in current_calls.values()
            if call.served_model
        }
    )
    pinned_model = ProviderModel.model_validate(golden_manifest["pinned_model"])
    pinned_model_name = pinned_model.model
    for row in golden_rows:
        trace_id = str(row["trace_id"])
        current = current_calls.get(trace_id)
        if current is None or current.response is None:
            continue
        baseline = _diff_response_from_mapping(row["baseline_response"])
        candidate = _diff_response_from_mapping(current.response)
        classification = classify_pair(
            baseline,
            candidate,
            embedder=FakeEmbedder(),
            threshold=tau,
        )
        classifications.append(classification.beyond_noise)
        assertion_pairs.append(
            AssertionPair(trace_id=trace_id, baseline=baseline, candidate=candidate)
        )
    beyond_noise_rate = (
        sum(1 for item in classifications if item) / len(classifications)
        if classifications
        else 0.0
    )
    baseline_pairs = [
        AssertionPair(
            trace_id=str(row["trace_id"]),
            baseline=_diff_response_from_mapping(row["baseline_response"]),
            candidate=_diff_response_from_mapping(row["baseline_response"]),
        )
        for row in golden_rows
    ]
    init_critical_failures = _critical_assertion_failures(manifest, baseline_pairs)
    current_critical_failures = _critical_assertion_failures(manifest, assertion_pairs)
    assertion_regressions = sorted(current_critical_failures - init_critical_failures)
    served_model = served_models[0] if len(served_models) == 1 else ",".join(served_models)

    reasons: list[str] = []
    if beyond_noise_rate > stored_bound + epsilon:
        reasons.append("beyond_noise_rate")
    if assertion_regressions:
        reasons.append("critical_assertion_regression")
    if served_model and served_model != pinned_model_name:
        reasons.append("served_model_mismatch")
    return {
        "assertion_regressions": assertion_regressions,
        "beyond_noise_rate": round(beyond_noise_rate, 12),
        "drift": bool(reasons),
        "pinned_model": pinned_model_name,
        "reasons": reasons,
        "served_model": served_model,
        "stored_bound": round(stored_bound + epsilon, 12),
    }


def _critical_assertion_failures(
    manifest: WorkloadManifest,
    pairs: list[AssertionPair],
) -> set[str]:
    failures: set[str] = set()
    for result in evaluate_assertions(manifest.assertions, pairs):
        if result.severity == Severity.CRITICAL and result.status == "failed":
            failures.add(result.assertion_id)
    return failures


def _select_suite(
    traces: list[TraceRecord],
    assertion_trace_ids: set[str],
    suite_size: int,
) -> list[TraceRecord]:
    by_id = {str(trace.trace_id): trace for trace in traces}
    selected_ids: list[str] = [
        trace_id
        for trace_id in sorted(assertion_trace_ids)
        if trace_id in by_id
    ]
    strata_seen: set[str] = set()
    for trace in sorted(traces, key=lambda item: (item.meta.template_id or "", str(item.trace_id))):
        if len(selected_ids) >= suite_size:
            break
        stratum = trace.meta.template_id or "templateless"
        trace_id = str(trace.trace_id)
        if stratum in strata_seen or trace_id in selected_ids:
            continue
        strata_seen.add(stratum)
        selected_ids.append(trace_id)
    if len(selected_ids) < min(suite_size, len(traces)):
        for trace in sorted(traces, key=lambda item: str(item.trace_id)):
            if len(selected_ids) >= suite_size:
                break
            trace_id = str(trace.trace_id)
            if trace_id not in selected_ids:
                selected_ids.append(trace_id)
    return [by_id[trace_id] for trace_id in selected_ids]


def _assertion_trace_ids(path: Path) -> set[str]:
    rows = _read_jsonl(path)
    return {
        str(row.get("pair_id") or row.get("trace_id"))
        for row in rows
        if row.get("assertion_id") and (row.get("pair_id") or row.get("trace_id"))
    }


def _monitor_dir(run_dir: Path, workload: str) -> Path:
    return run_dir / "monitor" / workload


def _read_traces(path: Path) -> list[TraceRecord]:
    return [TraceRecord.model_validate(row) for row in _read_jsonl(path)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        f"{json.dumps(row, ensure_ascii=False, separators=(',', ':'), sort_keys=True)}\n"
        for row in rows
    )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    with Path(f"{path}.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}\n")
    return digest


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{content}\n")
    digest = sha256_file(path)
    with Path(f"{path}.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}\n")
    return digest


def _params_hash_for_rows(rows: list[dict[str, Any]]) -> str:
    params = [
        {"params": row["post_mapping_params"], "trace_id": row["trace_id"]}
        for row in rows
    ]
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _diff_response_from_mapping(payload: dict[str, Any]) -> DiffResponse:
    return DiffResponse(
        text=payload.get("text"),
        tool_calls=payload.get("tool_calls"),
        finish_reason=payload.get("finish_reason"),
        latency_ms=payload.get("latency_ms"),
    )


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


__all__ = [
    "MonitorInitResult",
    "MonitorRunResult",
    "init_monitor",
    "run_monitor",
]
