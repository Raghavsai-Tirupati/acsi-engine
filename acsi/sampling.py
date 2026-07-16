from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acsi.schemas import SamplingConfig, TraceRecord

SHINGLE_SIZE = 5
DEDUP_JACCARD_THRESHOLD = 0.9


@dataclass(frozen=True)
class DedupCollapse:
    trace_id: str
    representative_trace_id: str
    jaccard: float


@dataclass(frozen=True)
class StratumReport:
    key: str
    available: int
    sampled: int


@dataclass(frozen=True)
class SamplingResult:
    records: list[TraceRecord]
    sampling_mode: str
    sha256: str
    report: dict[str, Any]
    dedup_collapses: list[DedupCollapse] = field(default_factory=list)


def sample_traces(
    records: list[TraceRecord],
    config: SamplingConfig,
) -> SamplingResult:
    representatives, collapses = deduplicate_traces(records)
    if config.n >= len(representatives):
        sampled = representatives
        mode = "exhaustive"
        strata = _stratum_reports(representatives, representatives, config.stratify_by)
    else:
        sampled, strata = _stratified_sample(representatives, config)
        mode = "stratified"
    content = _records_jsonl(sampled)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    report = {
        "dedup": {
            "collapsed_count": len(collapses),
            "jaccard_threshold": DEDUP_JACCARD_THRESHOLD,
            "shingle_size": SHINGLE_SIZE,
            "collapses": [
                {
                    "trace_id": collapse.trace_id,
                    "representative_trace_id": collapse.representative_trace_id,
                    "jaccard": round(collapse.jaccard, 12),
                }
                for collapse in collapses
            ],
        },
        "n_available_after_dedup": len(representatives),
        "n_requested": config.n,
        "n_sampled": len(sampled),
        "sampling_mode": mode,
        "seed": config.seed,
        "sha256": digest,
        "strata": [
            {
                "available": stratum.available,
                "key": stratum.key,
                "sampled": stratum.sampled,
            }
            for stratum in strata
        ],
        "stratify_by": config.stratify_by,
    }
    return SamplingResult(
        records=sampled,
        sampling_mode=mode,
        sha256=digest,
        report=report,
        dedup_collapses=collapses,
    )


def deduplicate_traces(records: list[TraceRecord]) -> tuple[list[TraceRecord], list[DedupCollapse]]:
    representatives: list[TraceRecord] = []
    representative_shingles: list[set[str]] = []
    collapses: list[DedupCollapse] = []
    for record in records:
        shingles = character_shingles(_dedup_text(record))
        matched_index = None
        matched_score = 0.0
        for index, existing in enumerate(representative_shingles):
            score = jaccard_similarity(shingles, existing)
            if score >= DEDUP_JACCARD_THRESHOLD:
                matched_index = index
                matched_score = score
                break
        if matched_index is None:
            representatives.append(record)
            representative_shingles.append(shingles)
        else:
            collapses.append(
                DedupCollapse(
                    trace_id=str(record.trace_id),
                    representative_trace_id=str(representatives[matched_index].trace_id),
                    jaccard=matched_score,
                )
            )
    return representatives, collapses


def character_shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    normalized = " ".join(text.split())
    if len(normalized) <= size:
        return {normalized}
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def write_sample_artifacts(
    records: list[TraceRecord],
    *,
    output_path: Path,
    report_path: Path,
    report: dict[str, Any],
) -> str:
    content = _records_jsonl(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    with Path(f"{output_path}.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}\n")

    report_payload = dict(report)
    report_payload["sha256"] = digest
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_content = json.dumps(report_payload, sort_keys=True, separators=(",", ":"))
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{report_content}\n")
    report_digest = hashlib.sha256(f"{report_content}\n".encode()).hexdigest()
    with Path(f"{report_path}.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{report_digest}\n")
    return digest


def _stratified_sample(
    records: list[TraceRecord],
    config: SamplingConfig,
) -> tuple[list[TraceRecord], list[StratumReport]]:
    rng = random.Random(config.seed)
    strata: defaultdict[str, list[TraceRecord]] = defaultdict(list)
    for record in records:
        strata[_stratum_key(record, config.stratify_by)].append(record)

    ordered_keys = sorted(strata)
    quotas = _allocate_quotas(
        {key: len(strata[key]) for key in ordered_keys},
        config.n,
    )
    sampled: list[TraceRecord] = []
    reports: list[StratumReport] = []
    for key in ordered_keys:
        members = sorted(strata[key], key=lambda item: str(item.trace_id))
        quota = quotas[key]
        if quota >= len(members):
            selected = members
        else:
            selected = sorted(rng.sample(members, quota), key=lambda item: str(item.trace_id))
        sampled.extend(selected)
        reports.append(StratumReport(key=key, available=len(members), sampled=len(selected)))
    return sorted(sampled, key=lambda item: str(item.trace_id)), reports


def _allocate_quotas(stratum_sizes: dict[str, int], n: int) -> dict[str, int]:
    non_empty = {key: size for key, size in stratum_sizes.items() if size > 0}
    if n < len(non_empty):
        ordered = sorted(non_empty, key=lambda key: (-non_empty[key], key))
        return {key: int(key in ordered[:n]) for key in stratum_sizes}

    total = sum(non_empty.values())
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for key, size in sorted(non_empty.items()):
        exact = n * size / total
        base = min(size, max(1, math.floor(exact)))
        quotas[key] = base
        remainders.append((exact - math.floor(exact), key))

    while sum(quotas.values()) > n:
        candidates = [
            (quotas[key], key)
            for key in quotas
            if quotas[key] > 1
        ]
        if not candidates:
            break
        _, key = sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True)[0]
        quotas[key] -= 1

    while sum(quotas.values()) < n:
        for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
            if quotas[key] < non_empty[key]:
                quotas[key] += 1
                break
        else:
            break

    for key in stratum_sizes:
        quotas.setdefault(key, 0)
    return quotas


def _stratum_reports(
    available_records: list[TraceRecord],
    sampled_records: list[TraceRecord],
    stratify_by: list[str],
) -> list[StratumReport]:
    available: defaultdict[str, int] = defaultdict(int)
    sampled: defaultdict[str, int] = defaultdict(int)
    for record in available_records:
        available[_stratum_key(record, stratify_by)] += 1
    for record in sampled_records:
        sampled[_stratum_key(record, stratify_by)] += 1
    return [
        StratumReport(key=key, available=available[key], sampled=sampled.get(key, 0))
        for key in sorted(available)
    ]


def _stratum_key(record: TraceRecord, stratify_by: list[str]) -> str:
    if not stratify_by:
        return "all"
    values = [f"{key}={_stratum_value(record, key)}" for key in stratify_by]
    return "|".join(values)


def _stratum_value(record: TraceRecord, key: str) -> str:
    if key == "template_id":
        return record.meta.template_id or "<none>"
    if key == "input_length_bucket":
        length = len(_prompt_text(record))
        if length < 500:
            return "short"
        if length < 1500:
            return "medium"
        return "long"
    extras = record.meta.model_extra or {}
    if key in extras:
        return str(extras[key])
    if hasattr(record, key):
        return str(getattr(record, key))
    return "<missing>"


def _prompt_text(record: TraceRecord) -> str:
    return "\n".join(message.content for message in record.request.messages)


def _dedup_text(record: TraceRecord) -> str:
    return json.dumps(
        record.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )


def _records_jsonl(records: list[TraceRecord]) -> str:
    lines = [
        json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        for record in sorted(records, key=lambda item: str(item.trace_id))
    ]
    return "".join(f"{line}\n" for line in lines)
