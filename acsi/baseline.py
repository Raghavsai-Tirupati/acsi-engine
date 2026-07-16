from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from acsi.diff.deterministic import DiffResponse
from acsi.diff.semantic import EmbeddingClient, FakeEmbedder, classify_pair
from acsi.replay.artifacts import RunClock, build_run_manifest, write_run_manifest
from acsi.replay.clients import CompletionClient, CompletionResponse
from acsi.replay.runner import (
    ReplayConfig,
    ReplayResult,
    build_completion_request,
    replay,
    write_responses_jsonl,
)
from acsi.replay.store import ReplayStore, StoredCall
from acsi.schemas import ProviderModel, TraceRecord
from acsi.stats import ConfidenceInterval, percentile_bootstrap_ci

DEFAULT_TAU = 0.90
INSUFFICIENT_VARIATION_PAIRS = 20
BOOTSTRAP_REPLICATES = 2_000


@dataclass(frozen=True)
class BaselineResult:
    run_id: str
    run_dir: Path
    replay_result: ReplayResult
    responses_sha256: str
    run_sha256: str
    noise_floor_sha256: str
    noise_floor: dict[str, Any]


async def run_baseline(
    traces: list[TraceRecord],
    model: ProviderModel,
    k_baseline: int,
    *,
    client: CompletionClient,
    store: ReplayStore,
    config: ReplayConfig,
    run_dir: Path,
    manifest_path: Path,
    traces_path: Path,
    endpoint: str,
    embedder: EmbeddingClient | None = None,
    degraded: bool = False,
) -> BaselineResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    clock = RunClock()
    if degraded:
        result = _write_degraded_baseline_calls(traces, model, store, config)
    else:
        result = await replay(
            traces,
            model,
            k_baseline,
            client=client,
            store=store,
            config=config,
        )

    responses_hash = write_responses_jsonl(store, config.run_id, run_dir / "responses.jsonl")
    run_manifest = build_run_manifest(
        run_id=config.run_id,
        manifest_path=manifest_path,
        traces_path=traces_path,
        seed=config.seed,
        provider=model.provider,
        endpoint=endpoint,
        store=store,
        result=result,
        wall_clock_seconds=clock.elapsed_seconds(),
        degraded=degraded,
    )
    run_hash = write_run_manifest(run_dir / "run.json", run_manifest)

    if degraded:
        noise_floor = degraded_noise_floor(
            run_id=config.run_id,
            prompt_count=len(traces),
            k_baseline=k_baseline,
        )
    else:
        noise_floor = calculate_noise_floor(
            store.done_calls(config.run_id),
            run_id=config.run_id,
            k_baseline=k_baseline,
            seed=config.seed,
            embedder=embedder or FakeEmbedder(),
            client_noise_rate=_client_noise_rate(client),
        )
    noise_hash = write_noise_floor_json(run_dir / "baseline" / "noise_floor.json", noise_floor)

    return BaselineResult(
        run_id=config.run_id,
        run_dir=run_dir,
        replay_result=result,
        responses_sha256=responses_hash,
        run_sha256=run_hash,
        noise_floor_sha256=noise_hash,
        noise_floor=noise_floor,
    )


def calculate_noise_floor(
    calls: list[StoredCall],
    *,
    run_id: str,
    k_baseline: int,
    seed: int,
    embedder: EmbeddingClient,
    client_noise_rate: float | None = None,
) -> dict[str, Any]:
    pairs = _self_pairs(calls)
    mismatch_indicators: list[float] = []
    unequal_pairs: list[tuple[DiffResponse, DiffResponse]] = []
    unequal_similarities: list[float] = []

    for left, right in pairs:
        classification = classify_pair(
            _diff_response(left),
            _diff_response(right),
            embedder=embedder,
            threshold=0.0,
        )
        mismatched = not classification.deterministic_equal
        mismatch_indicators.append(float(mismatched))
        if mismatched:
            unequal_pairs.append((_diff_response(left), _diff_response(right)))
            unequal_similarities.append(classification.similarity)

    if not mismatch_indicators:
        mismatch_indicators = [0.0]

    if len(unequal_similarities) < INSUFFICIENT_VARIATION_PAIRS:
        tau = DEFAULT_TAU
        threshold_source = "default_insufficient_variation"
    else:
        tau = float(np.percentile(np.asarray(unequal_similarities, dtype=np.float64), 5))
        threshold_source = "calibrated"

    beyond_indicators = [0.0 for _ in pairs]
    if unequal_pairs:
        beyond_by_pair = [
            float(
                classify_pair(
                    left,
                    right,
                    embedder=embedder,
                    threshold=tau,
                ).beyond_noise
            )
            for left, right in unequal_pairs
        ]
        beyond_iter = iter(beyond_by_pair)
        beyond_indicators = [
            next(beyond_iter) if mismatch else 0.0
            for mismatch in mismatch_indicators
        ]
    if not beyond_indicators:
        beyond_indicators = [0.0]

    textual_ci = percentile_bootstrap_ci(
        mismatch_indicators,
        b=BOOTSTRAP_REPLICATES,
        seed=seed,
    )
    beyond_ci = percentile_bootstrap_ci(
        beyond_indicators,
        b=BOOTSTRAP_REPLICATES,
        seed=seed,
    )
    prompt_count = len({call.trace_id for call in calls})
    pair_count = len(pairs)
    unequal_count = int(sum(mismatch_indicators))
    beyond_count = int(sum(beyond_indicators))
    beyond_to_textual = (
        beyond_ci.mean / textual_ci.mean if textual_ci.mean > 0 else 0.0
    )
    artifact: dict[str, Any] = {
        "beyond_noise_ci": _ci_payload(beyond_ci),
        "beyond_noise_rate": _stable_float(beyond_ci.mean),
        "beyond_noise_to_textual_mismatch_rate": _stable_float(beyond_to_textual),
        "classifications": {
            "beyond_noise": beyond_count,
            "deterministically_equal": pair_count - unequal_count,
            "deterministically_unequal": unequal_count,
        },
        "degraded": False,
        "deterministically_unequal_self_pairs": unequal_count,
        "k_baseline": k_baseline,
        "pair_count": pair_count,
        "prompt_count": prompt_count,
        "run_id": run_id,
        "tau": _stable_float(tau),
        "textual_mismatch_ci": _ci_payload(textual_ci),
        "textual_mismatch_rate": _stable_float(textual_ci.mean),
        "threshold_source": threshold_source,
    }
    if client_noise_rate is not None:
        artifact["analytic_note"] = {
            "expected_mismatch_rate": _stable_float(
                2 * client_noise_rate * (1 - client_noise_rate)
            ),
            "q": _stable_float(client_noise_rate),
        }
    return artifact


def degraded_noise_floor(
    *,
    run_id: str,
    prompt_count: int,
    k_baseline: int,
) -> dict[str, Any]:
    return {
        "degraded": True,
        "k_baseline": k_baseline,
        "noise_floor": "unavailable",
        "pair_count": 0,
        "prompt_count": prompt_count,
        "run_id": run_id,
        "tau": DEFAULT_TAU,
        "threshold_source": "default_degraded",
    }


def write_noise_floor_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{content}\n")
    digest = hashlib.sha256(f"{content}\n".encode()).hexdigest()
    return digest


def _write_degraded_baseline_calls(
    traces: list[TraceRecord],
    model: ProviderModel,
    store: ReplayStore,
    config: ReplayConfig,
) -> ReplayResult:
    store.initialize()
    result = ReplayResult(run_id=config.run_id)
    for trace in traces:
        request, prompt_hash, params_hash, transforms = build_completion_request(
            trace,
            model,
            0,
        )
        response = CompletionResponse(
            text=trace.response.text,
            tool_calls=trace.response.tool_calls,
            finish_reason=trace.response.finish_reason or "unknown",
            usage=trace.response.usage.model_dump(mode="json") if trace.response.usage else {},
            latency_ms=trace.response.latency_ms or 0,
            served_model=trace.response.served_model or trace.request.model,
        )
        store.write_done(
            run_id=config.run_id,
            trace_id=str(trace.trace_id),
            sample_index=0,
            model=request.model,
            params_hash=params_hash,
            prompt_hash=prompt_hash,
            response=response,
            cost_usd=0.0,
            retry_count=0,
        )
        result.completed += 1
        result.param_transforms.extend(transforms)
        result.served_models.add(response.served_model)
    return result


def _self_pairs(calls: list[StoredCall]) -> list[tuple[StoredCall, StoredCall]]:
    by_trace_id: defaultdict[str, list[StoredCall]] = defaultdict(list)
    for call in calls:
        by_trace_id[call.trace_id].append(call)
    pairs: list[tuple[StoredCall, StoredCall]] = []
    for trace_id in sorted(by_trace_id):
        ordered = sorted(by_trace_id[trace_id], key=lambda call: call.sample_index)
        pairs.extend(combinations(ordered, 2))
    return pairs


def _diff_response(call: StoredCall) -> DiffResponse:
    response = call.response or {}
    return DiffResponse(
        text=response.get("text"),
        tool_calls=response.get("tool_calls"),
        finish_reason=response.get("finish_reason"),
        latency_ms=response.get("latency_ms"),
    )


def _ci_payload(interval: ConfidenceInterval) -> dict[str, float]:
    return {
        "confidence": _stable_float(interval.confidence),
        "lower": _stable_float(interval.lower),
        "rate": _stable_float(interval.mean),
        "upper": _stable_float(interval.upper),
    }


def _stable_float(value: float) -> float:
    return round(float(value), 12)


def _client_noise_rate(client: CompletionClient) -> float | None:
    noise = getattr(client, "noise", None)
    return float(noise) if isinstance(noise, int | float) else None
