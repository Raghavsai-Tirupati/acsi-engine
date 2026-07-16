from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

from acsi import __version__
from acsi.replay.params import AppliedParamTransform, summarize_param_transforms
from acsi.replay.runner import ReplayResult
from acsi.replay.store import ReplayStore
from acsi.schemas import (
    ContentHash,
    CostLedgerEntry,
    ParamTransformation,
    RunManifest,
)


class RunClock:
    def __init__(self) -> None:
        self.started_at = perf_counter()

    def elapsed_seconds(self) -> float:
        return perf_counter() - self.started_at


def build_run_manifest(
    *,
    run_id: str,
    manifest_path: Path,
    traces_path: Path,
    seed: int,
    provider: str,
    endpoint: str,
    store: ReplayStore,
    result: ReplayResult,
    wall_clock_seconds: float,
    degraded: bool = False,
) -> RunManifest:
    calls = store.done_calls(run_id)
    tokens_in = sum(call.usage.get("input_tokens", 0) for call in calls)
    tokens_out = sum(call.usage.get("output_tokens", 0) for call in calls)
    cost_usd = sum(call.cost_usd for call in calls)
    return RunManifest(
        run_id=run_id,
        manifest_hash=ContentHash(value=sha256_file(manifest_path)),
        sampled_trace_hash=ContentHash(value=sha256_file(traces_path)),
        engine_version=__version__,
        seeds={"replay": seed},
        endpoints={provider: endpoint},
        served_models=store.served_models(run_id),
        degraded=degraded,
        param_transformations=_param_transformations(result.param_transforms),
        wall_clock_seconds=wall_clock_seconds,
        cost_ledger=[
            CostLedgerEntry(
                stage="replay",
                provider=provider,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                usd=cost_usd,
            )
        ],
    )


def write_run_manifest(path: Path, manifest: RunManifest) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{content}\n")
    digest = hashlib.sha256(f"{content}\n".encode()).hexdigest()
    with Path(f"{path}.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}\n")
    return digest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _param_transformations(transforms: list[AppliedParamTransform]) -> list[ParamTransformation]:
    return [
        ParamTransformation.model_validate(summary)
        for summary in summarize_param_transforms(transforms)
    ]
