from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from sklearn.cluster import HDBSCAN

from acsi.diff.semantic import EmbeddingClient, FakeEmbedder
from acsi.judge.rubric import CandidateOutcome
from acsi.replay.clients import CompletionClient, CompletionRequest, CompletionResponse
from acsi.replay.runner import estimate_call_cost_usd
from acsi.replay.store import ReplayStore
from acsi.schemas import Severity

REGRESSION_OUTCOMES: set[CandidateOutcome] = {
    "worse_minor",
    "worse_critical",
    "unresolved",
}
ASSERTION_SEVERITY_RANK: dict[Severity, int] = {
    Severity.MINOR: 1,
    Severity.MAJOR: 2,
    Severity.CRITICAL: 3,
}
OUTCOME_SEVERITY_RANK: dict[CandidateOutcome, int] = {
    "worse_minor": 1,
    "unresolved": 2,
    "worse_critical": 3,
}
SEVERITY_BY_RANK = {1: "worse_minor", 2: "major", 3: "worse_critical"}
_OUTCOME_LABELS: dict[str, str] = {
    "unresolved": "Unresolved — panel could not decide",
    "worse_minor": "Judge: candidate worse (minor)",
    "worse_critical": "Judge: candidate worse (critical)",
}
CLUSTER_NAME_PHASE = "cluster_name"
CLUSTER_NAME_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "description"],
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 80},
        "description": {"type": "string", "minLength": 1},
    },
}


@dataclass(frozen=True)
class AssertionFailure:
    assertion_id: str
    severity: Severity
    baseline_passed: bool = True
    candidate_passed: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class CandidatePairRecord:
    pair_id: str
    prompt: str
    baseline_response: str
    candidate_response: str
    ensemble_outcome: CandidateOutcome
    judge_reasons: list[str] = field(default_factory=list)
    assertion_failures: list[AssertionFailure] = field(default_factory=list)
    template_id: str | None = None
    system: str | None = None


@dataclass(frozen=True)
class RegressionPair:
    pair_id: str
    prompt: str
    baseline_response: str
    candidate_response: str
    ensemble_outcome: CandidateOutcome
    judge_reasons: list[str]
    flipped_assertion_ids: list[str]
    assertion_failures: list[AssertionFailure]
    detection_source: Literal["assertion", "judge", "mixed"]
    signature: str
    severity_rank: int
    reason_labels: list[str] = field(default_factory=list)
    template_id: str | None = None
    system: str | None = None


@dataclass(frozen=True)
class ClusterBucket:
    cluster_id: str
    label: int | str
    name: str
    description: str
    pair_ids: list[str]
    signatures: list[str]
    severity: str
    share_of_sampled: float
    unclustered: bool = False
    skip_reason: str | None = None
    parse_failure: bool = False
    reason_name: str | None = None


@dataclass
class ClusterNameResult:
    name: str
    description: str
    parse_failure: bool
    cache_hit: bool
    dispatched: bool
    cost_usd: float


class ClusterInterrupted(RuntimeError):
    pass


class ClusterNameParseError(ValueError):
    pass


class FakeNamer:
    def __init__(
        self,
        *,
        names: dict[str, tuple[str, str]] | None = None,
        malformed_attempts: set[tuple[str, int]] | None = None,
        seed: int = 42,
    ) -> None:
        self.names = names or {}
        self.malformed_attempts = malformed_attempts or set()
        self.seed = seed
        self.call_count = 0

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.call_count += 1
        cluster_id = str(request.params.get("cluster_id", "cluster"))
        attempt = int(request.params.get("attempt", request.sample_index))
        if (cluster_id, attempt) in self.malformed_attempts:
            text = "{malformed"
        else:
            name, description = self.names.get(
                cluster_id,
                (f"{cluster_id} issue", "Cluster of similar regressions."),
            )
            text = json.dumps(
                {"description": description, "name": name},
                sort_keys=True,
                separators=(",", ":"),
            )
        return CompletionResponse(
            text=text,
            tool_calls=None,
            finish_reason="stop",
            usage={
                "input_tokens": max(1, len(request.prompt_text) // 4),
                "output_tokens": max(1, len(text) // 4),
            },
            latency_ms=10 + self.seed % 10,
            served_model=request.model,
        )


@dataclass
class DispatchBudget:
    interrupt_after_dispatches: int | None = None
    dispatched: int = 0

    def record_dispatch(self) -> None:
        self.dispatched += 1
        if (
            self.interrupt_after_dispatches is not None
            and self.dispatched >= self.interrupt_after_dispatches
        ):
            raise ClusterInterrupted("Cluster naming interrupted; rerun to resume.")


def build_regression_set(records: list[CandidatePairRecord]) -> list[RegressionPair]:
    regressions: list[RegressionPair] = []
    for record in sorted(records, key=lambda item: item.pair_id):
        judge_detected = record.ensemble_outcome in REGRESSION_OUTCOMES
        assertion_failures = [
            failure
            for failure in record.assertion_failures
            if (
                failure.baseline_passed
                and not failure.candidate_passed
                and failure.severity in {Severity.CRITICAL, Severity.MAJOR}
            )
        ]
        if not judge_detected and not assertion_failures:
            continue
        detection_source: Literal["assertion", "judge", "mixed"]
        if judge_detected and assertion_failures:
            detection_source = "mixed"
        elif judge_detected:
            detection_source = "judge"
        else:
            detection_source = "assertion"
        flipped_ids = sorted(failure.assertion_id for failure in assertion_failures)
        assertion_reasons = sorted(
            failure.reason for failure in assertion_failures if failure.reason
        )
        reason_labels = [reason_label(reason) for reason in assertion_reasons]
        severity_rank = _regression_severity_rank(record.ensemble_outcome, assertion_failures)
        regressions.append(
            RegressionPair(
                pair_id=record.pair_id,
                prompt=record.prompt,
                baseline_response=record.baseline_response,
                candidate_response=record.candidate_response,
                ensemble_outcome=record.ensemble_outcome,
                judge_reasons=record.judge_reasons,
                flipped_assertion_ids=flipped_ids,
                assertion_failures=assertion_failures,
                detection_source=detection_source,
                signature=compose_signature(
                    flipped_ids,
                    record.ensemble_outcome,
                    record.judge_reasons,
                    record.candidate_response,
                    assertion_reasons=assertion_reasons,
                ),
                severity_rank=severity_rank,
                reason_labels=reason_labels,
                template_id=record.template_id,
                system=record.system,
            )
        )
    return regressions


_QUOTE_CHARS = "'\"`"
_QUOTED_RE = re.compile(r"(['\"`]).*?\1", re.DOTALL)
_DIGITS_RE = re.compile(r"\d+")


def normalize_reason_template(reason: str) -> str:
    """Collapse a validator reason to its mechanism for clustering.

    Drops per-pair specifics so every "summary too long" failure shares one key
    instead of fragmenting on the offending text (run 7f0978f5 left 150
    same-mechanism failures unclustered for this reason). The whole span from the
    first to the last quote is replaced — robust to quotes and digits inside the
    offending value (e.g. an apostrophe in the summary) that a balanced-quote
    match would trip on.
    """
    first = min((reason.find(char) for char in _QUOTE_CHARS if char in reason), default=-1)
    if first != -1:
        last = max(reason.rfind(char) for char in _QUOTE_CHARS)
        reason = f"{reason[:first]}<x>{reason[last + 1 :]}"
    return " ".join(_DIGITS_RE.sub("N", reason).split())


def reason_label(reason: str) -> str:
    """Human cluster name from a reason: shorten only long quoted values, keep
    short ones (e.g. the ``` fence) and constants (e.g. a 400 limit)."""

    def _shorten(match: re.Match[str]) -> str:
        value = match.group(0)
        return value if len(value) <= 26 else f"{value[0]}…{value[0]}"

    return " ".join(_QUOTED_RE.sub(_shorten, reason).split())


def compose_signature(
    flipped_assertion_ids: list[str],
    ensemble_outcome: CandidateOutcome,
    judge_reasons: list[str],
    candidate_response: str,
    assertion_reasons: Sequence[str] = (),
) -> str:
    # SPEC-NOTE: assertion-flagged regressions cluster on assertion id + the
    # normalized reason TEMPLATE (mechanism), not the raw reason or candidate
    # output. Same-mechanism failures then share an identical signature and group,
    # instead of every pair's unique reason text/output fragmenting them into the
    # unclustered bucket. Judge-only regressions keep the outcome/reason/output
    # signature so distinct judge failure modes still separate.
    if flipped_assertion_ids:
        templates = sorted({normalize_reason_template(r) for r in assertion_reasons if r})
        # Cluster purely on the assertion mechanism (id + reason template). The
        # judge's outcome is deliberately excluded so one mechanism does not split
        # three ways by whatever the panel happened to say about each pair.
        pieces = [
            " ".join(flipped_assertion_ids),
            " ".join(templates),
        ]
        return " ".join(piece for piece in pieces if piece)
    pieces = [
        ensemble_outcome,
        " ".join(judge_reasons),
        candidate_response[:500],
    ]
    return " ".join(piece for piece in pieces if piece)


def cluster_regressions(
    regressions: list[RegressionPair],
    *,
    n_sampled_pairs: int,
    embedder: EmbeddingClient | None = None,
    min_cluster_size: int | None = None,
    name_by_outcome: bool = False,
) -> list[ClusterBucket]:
    if not regressions:
        return []
    active_min = min_cluster_size or max(3, math.ceil(0.02 * len(regressions)))
    if len(regressions) < 2 * active_min:
        return [
            _bucket_from_members(
                "all_regressions",
                "all_regressions",
                regressions,
                n_sampled_pairs=n_sampled_pairs,
                name="all_regressions",
                description="All regressions grouped because there are too few samples.",
                skip_reason=(
                    f"regression_count {len(regressions)} < 2 * min_cluster_size {active_min}"
                ),
                name_by_outcome=name_by_outcome,
            )
        ]

    active_embedder = embedder or FakeEmbedder()
    embeddings = active_embedder.embed([regression.signature for regression in regressions])
    labels = HDBSCAN(
        min_cluster_size=active_min,
        metric="euclidean",
        allow_single_cluster=True,
        copy=True,
    ).fit_predict(np.asarray(embeddings, dtype=np.float64))

    buckets: list[ClusterBucket] = []
    for label in sorted(set(int(label) for label in labels)):
        members = [
            regression
            for regression, regression_label in zip(regressions, labels, strict=True)
            if int(regression_label) == label
        ]
        cluster_id = "unclustered" if label == -1 else f"cluster-{label}"
        buckets.append(
            _bucket_from_members(
                cluster_id,
                label,
                members,
                n_sampled_pairs=n_sampled_pairs,
                name=cluster_id,
                description="Unclustered regressions." if label == -1 else "Unnamed cluster.",
                unclustered=label == -1,
                name_by_outcome=name_by_outcome,
            )
        )
    return buckets


def name_clusters(
    buckets: list[ClusterBucket],
    *,
    namer: CompletionClient,
    store: ReplayStore,
    run_id: str,
    interrupt_after_dispatches: int | None = None,
) -> tuple[list[ClusterBucket], dict[str, int | float]]:
    store.initialize()
    budget = DispatchBudget(interrupt_after_dispatches)
    named: list[ClusterBucket] = []
    parse_failures = 0
    cache_hits = 0
    cost_usd = 0.0
    for bucket in buckets:
        if bucket.skip_reason or bucket.reason_name:
            # Assertion clusters are named deterministically from their dominant
            # reason label; only judge-only / unnamed clusters consult the namer.
            named.append(bucket)
            continue
        result = _name_cluster(
            bucket,
            namer=namer,
            store=store,
            run_id=run_id,
            budget=budget,
        )
        parse_failures += int(result.parse_failure)
        cache_hits += int(result.cache_hit)
        cost_usd += result.cost_usd
        named.append(
            ClusterBucket(
                cluster_id=bucket.cluster_id,
                label=bucket.label,
                name=result.name,
                description=result.description,
                pair_ids=bucket.pair_ids,
                signatures=bucket.signatures,
                severity=bucket.severity,
                share_of_sampled=bucket.share_of_sampled,
                unclustered=bucket.unclustered,
                skip_reason=bucket.skip_reason,
                parse_failure=result.parse_failure,
            )
        )
    return named, {
        "cache_hits": cache_hits,
        "cost_usd": round(cost_usd, 12),
        "dispatched": budget.dispatched,
        "parse_failures": parse_failures,
    }


def write_clusters_json(
    path: Path,
    buckets: list[ClusterBucket],
    *,
    stats: dict[str, int | float] | None = None,
) -> str:
    stable_stats = {
        key: value
        for key, value in (stats or {}).items()
        if key not in {"cache_hits", "dispatched"}
    }
    payload = {
        "clusters": [
            {
                "cluster_id": bucket.cluster_id,
                "description": bucket.description,
                "label": bucket.label,
                "member_count": len(bucket.pair_ids),
                "name": bucket.name,
                "pair_ids": bucket.pair_ids,
                "parse_failure": bucket.parse_failure,
                "severity": bucket.severity,
                "share_of_sampled": bucket.share_of_sampled,
                "skip_reason": bucket.skip_reason,
                "unclustered": bucket.unclustered,
            }
            for bucket in sorted(buckets, key=lambda item: item.cluster_id)
        ],
        "stats": stable_stats,
    }
    return _write_json(path, payload)


def _name_cluster(
    bucket: ClusterBucket,
    *,
    namer: CompletionClient,
    store: ReplayStore,
    run_id: str,
    budget: DispatchBudget,
) -> ClusterNameResult:
    prompt = "\n\n".join(
        [
            "Name this group of model-regression signatures.",
            "Return only JSON with keys name and description.",
            *bucket.signatures[:5],
        ]
    )
    cache_hit = False
    cost_usd = 0.0
    for attempt in range(2):
        cached = store.get_done(
            run_id,
            f"name:{bucket.cluster_id}",
            attempt,
            phase=CLUSTER_NAME_PHASE,
        )
        if cached:
            cache_hit = True
            text = cached.response.get("text") if cached.response else None
            cost_usd += cached.cost_usd
        else:
            request = CompletionRequest(
                provider="fake",
                model="cluster-namer",
                system=None,
                messages=[{"role": "user", "content": prompt}],
                params={"attempt": attempt, "cluster_id": bucket.cluster_id},
                sample_index=attempt,
            )
            response = namer.complete(request)
            actual_cost = _call_cost(request, response, namer)
            store.write_done(
                run_id=run_id,
                trace_id=f"name:{bucket.cluster_id}",
                sample_index=attempt,
                model=request.model,
                params_hash=_hash_json(request.params),
                prompt_hash=_hash_text(prompt),
                response=response,
                cost_usd=actual_cost,
                retry_count=0,
                phase=CLUSTER_NAME_PHASE,
            )
            budget.record_dispatch()
            text = response.text
            cost_usd += actual_cost
        try:
            parsed = _parse_cluster_name(text)
            return ClusterNameResult(
                name=parsed["name"],
                description=parsed["description"],
                parse_failure=False,
                cache_hit=cache_hit,
                dispatched=not cache_hit,
                cost_usd=cost_usd,
            )
        except ClusterNameParseError:
            continue
    return ClusterNameResult(
        name=bucket.cluster_id,
        description="Cluster naming failed.",
        parse_failure=True,
        cache_hit=cache_hit,
        dispatched=not cache_hit,
        cost_usd=cost_usd,
    )


def _parse_cluster_name(text: str | None) -> dict[str, str]:
    if text is None:
        raise ClusterNameParseError("empty cluster name response")
    try:
        payload = json.loads(text)
        validate_json_schema(payload, CLUSTER_NAME_SCHEMA)
    except (json.JSONDecodeError, JsonSchemaValidationError) as exc:
        raise ClusterNameParseError(str(exc)) from exc
    words = str(payload["name"]).split()
    if len(words) > 6:
        payload["name"] = " ".join(words[:6])
    return {"description": str(payload["description"]), "name": str(payload["name"])}


def _bucket_from_members(
    cluster_id: str,
    label: int | str,
    members: list[RegressionPair],
    *,
    n_sampled_pairs: int,
    name: str,
    description: str,
    unclustered: bool = False,
    skip_reason: str | None = None,
    name_by_outcome: bool = False,
) -> ClusterBucket:
    # SPEC-NOTE: an assertion cluster carries an assertion-style severity and is
    # described as a shared assertion failure. A judge-only cluster (no member has
    # an assertion failure) is a panel OUTCOME, not an assertion failure: it is
    # named and severity-labeled by that outcome (e.g. "unresolved") and never
    # described as an assertion failure — unresolved is not a harm severity.
    assertion_name = _dominant_reason_label(members)
    if assertion_name is not None:
        reason_name = assertion_name
        severity = SEVERITY_BY_RANK[max(member.severity_rank for member in members)]
        resolved_description = (
            f"{len(members)} pair(s) share this assertion failure: {assertion_name}"
        )
    elif name_by_outcome:
        dominant_outcome = min(
            Counter(member.ensemble_outcome for member in members).items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
        reason_name = _OUTCOME_LABELS.get(dominant_outcome)
        severity = dominant_outcome if reason_name else SEVERITY_BY_RANK[
            max(member.severity_rank for member in members)
        ]
        resolved_description = (
            f"{len(members)} pair(s) with judge panel outcome: {reason_name}"
            if reason_name
            else description
        )
    else:
        reason_name = None
        severity = SEVERITY_BY_RANK[max(member.severity_rank for member in members)]
        resolved_description = description
    return ClusterBucket(
        cluster_id=cluster_id,
        label=label,
        name=reason_name or name,
        description=resolved_description,
        pair_ids=sorted(member.pair_id for member in members),
        signatures=[member.signature for member in sorted(members, key=lambda item: item.pair_id)],
        severity=severity,
        share_of_sampled=round(len(members) / n_sampled_pairs, 12),
        unclustered=unclustered,
        skip_reason=skip_reason,
        reason_name=reason_name,
    )


def _dominant_reason_label(members: list[RegressionPair]) -> str | None:
    """Most common assertion reason label across a cluster's members, if any."""
    labels = Counter(label for member in members for label in member.reason_labels)
    if not labels:
        return None
    # Deterministic tie-break: highest count, then lexicographically first.
    return min(labels.items(), key=lambda item: (-item[1], item[0]))[0]


def _regression_severity_rank(
    outcome: CandidateOutcome,
    failures: list[AssertionFailure],
) -> int:
    ranks = [OUTCOME_SEVERITY_RANK.get(outcome, 0)]
    ranks.extend(ASSERTION_SEVERITY_RANK[failure.severity] for failure in failures)
    return max(ranks)


def _call_cost(
    request: CompletionRequest,
    response: CompletionResponse,
    client: CompletionClient,
) -> float:
    return estimate_call_cost_usd(
        request.provider,
        request.model,
        response.usage.get("input_tokens", 0),
        response.usage.get("output_tokens", 0),
        fake=isinstance(client, FakeNamer),
    )


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{content}\n")
    digest = hashlib.sha256(f"{content}\n".encode()).hexdigest()
    with Path(f"{path}.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}\n")
    return digest
