from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema

from acsi.diff.clustering import ClusterBucket, RegressionPair
from acsi.diff.deterministic import DiffResponse, json_valid
from acsi.diff.semantic import FakeEmbedder, classify_pair
from acsi.replay.clients import CompletionClient, CompletionRequest, CompletionResponse, FakeClient
from acsi.replay.runner import ReplayConfig, replay
from acsi.replay.store import ReplayStore
from acsi.schemas import ProviderModel, TraceRecord

PATCH_PHASE = "patch_proposal"
PATCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["target", "replacement", "rationale"],
    "properties": {
        "target": {"enum": ["system", "template_prefix"]},
        "replacement": {"type": "string"},
        "rationale": {"type": "string", "minLength": 1},
    },
}


@dataclass(frozen=True)
class TemplateInfo:
    template_id: str | None
    prefix: str
    suffix: str
    stable: bool
    ineligible: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class TemplateDetectionResult:
    templates: dict[str, TemplateInfo]
    null_template_count: int
    fallback: TemplateInfo | None = None
    skip_reason: str | None = None


@dataclass(frozen=True)
class PatchTarget:
    kind: Literal["system", "template_prefix"]
    text: str
    skip_reason: str | None = None


@dataclass(frozen=True)
class PatchProposal:
    cluster_id: str
    target: Literal["system", "template_prefix"]
    replacement: str
    rationale: str
    diff_text: str
    parse_failure: bool = False


@dataclass(frozen=True)
class PatchReport:
    cluster_id: str
    diff_path: str
    fixed_fraction: float
    control_regressions: int
    accepted: bool
    reason: str


class PatchInterrupted(RuntimeError):
    pass


class PatchParseError(ValueError):
    pass


class FakePatcher:
    def __init__(
        self,
        *,
        replacements: dict[str, tuple[str, str, str]] | None = None,
        malformed_attempts: set[tuple[str, int]] | None = None,
        seed: int = 42,
    ) -> None:
        self.replacements = replacements or {}
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
            target, replacement, rationale = self.replacements.get(
                cluster_id,
                ("system", "STRICT_JSON_MODE", "Add stricter output guidance."),
            )
            text = json.dumps(
                {
                    "rationale": rationale,
                    "replacement": replacement,
                    "target": target,
                },
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
            latency_ms=12 + self.seed % 10,
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
            raise PatchInterrupted("Patch proposal interrupted; rerun to resume.")


def detect_templates(traces: list[TraceRecord]) -> TemplateDetectionResult:
    groups: dict[str | None, list[TraceRecord]] = {}
    for trace in traces:
        groups.setdefault(trace.meta.template_id, []).append(trace)

    templates: dict[str, TemplateInfo] = {}
    null_count = len(groups.get(None, []))
    for template_id, members in sorted(
        ((key, value) for key, value in groups.items() if key is not None),
        key=lambda item: str(item[0]),
    ):
        info = _template_info(str(template_id), members)
        templates[str(template_id)] = info

    if templates:
        return TemplateDetectionResult(
            templates=templates,
            null_template_count=null_count,
            skip_reason=None,
        )

    fallback = _template_info("__all__", traces)
    if fallback.stable:
        return TemplateDetectionResult(
            templates={},
            null_template_count=null_count,
            fallback=fallback,
        )
    return TemplateDetectionResult(
        templates={},
        null_template_count=null_count,
        fallback=fallback,
        skip_reason="no_stable_template",
    )


def select_patch_target(
    traces: list[TraceRecord],
    detection: TemplateDetectionResult,
) -> PatchTarget:
    if detection.skip_reason and not detection.templates:
        return PatchTarget(
            kind="template_prefix",
            text="",
            skip_reason=detection.skip_reason,
        )

    systems = [trace.request.system for trace in traces if trace.request.system]
    if systems:
        common_system, count = Counter(systems).most_common(1)[0]
        if count / len(traces) >= 0.95:
            return PatchTarget(kind="system", text=common_system)

    stable_templates = [info for info in detection.templates.values() if info.stable]
    if stable_templates:
        selected = max(stable_templates, key=lambda info: len(info.prefix))
        return PatchTarget(kind="template_prefix", text=selected.prefix)
    if detection.fallback and detection.fallback.stable:
        return PatchTarget(kind="template_prefix", text=detection.fallback.prefix)
    return PatchTarget(
        kind="template_prefix",
        text="",
        skip_reason=detection.skip_reason or "no_stable_template",
    )


def propose_patch(
    *,
    cluster: ClusterBucket,
    regressions: list[RegressionPair],
    target: PatchTarget,
    patcher: CompletionClient,
    store: ReplayStore,
    run_id: str,
    interrupt_after_dispatches: int | None = None,
) -> tuple[PatchProposal | None, dict[str, int]]:
    if target.skip_reason:
        return None, {"cache_hits": 0, "dispatched": 0, "parse_failures": 0}
    store.initialize()
    budget = DispatchBudget(interrupt_after_dispatches)
    cache_hit = False
    selected = [regression for regression in regressions if regression.pair_id in cluster.pair_ids]
    prompt = _patch_prompt(cluster, selected[:3], target)
    for attempt in range(2):
        cached = store.get_done(run_id, f"patch:{cluster.cluster_id}", attempt, phase=PATCH_PHASE)
        if cached:
            cache_hit = True
            text = cached.response.get("text") if cached.response else None
        else:
            request = CompletionRequest(
                provider="fake",
                model="patch-proposer",
                system=None,
                messages=[{"role": "user", "content": prompt}],
                params={"attempt": attempt, "cluster_id": cluster.cluster_id},
                sample_index=attempt,
            )
            response = patcher.complete(request)
            store.write_done(
                run_id=run_id,
                trace_id=f"patch:{cluster.cluster_id}",
                sample_index=attempt,
                model=request.model,
                params_hash=_hash_json(request.params),
                prompt_hash=_hash_text(prompt),
                response=response,
                cost_usd=0.0,
                retry_count=0,
                phase=PATCH_PHASE,
            )
            budget.record_dispatch()
            text = response.text
        try:
            payload = _parse_patch(text)
            diff_text = _unified_diff(target.text, payload["replacement"])
            return PatchProposal(
                cluster_id=cluster.cluster_id,
                target=payload["target"],
                replacement=payload["replacement"],
                rationale=payload["rationale"],
                diff_text=diff_text,
            ), {
                "cache_hits": int(cache_hit),
                "dispatched": budget.dispatched,
                "parse_failures": 0,
            }
        except PatchParseError:
            continue
    return PatchProposal(
        cluster_id=cluster.cluster_id,
        target=target.kind,
        replacement=target.text,
        rationale="Patch proposal failed to parse.",
        diff_text="",
        parse_failure=True,
    ), {
        "cache_hits": int(cache_hit),
        "dispatched": budget.dispatched,
        "parse_failures": 1,
    }


def validate_patch(
    *,
    proposal: PatchProposal,
    cluster: ClusterBucket,
    regressions: list[RegressionPair],
    equivalent_pairs: list[RegressionPair],
    traces_by_pair_id: dict[str, TraceRecord],
    model: ProviderModel,
    client: FakeClient,
    run_dir: Path,
    min_fix_rate: float = 0.8,
    seed: int = 42,
) -> PatchReport:
    cluster_regressions = [item for item in regressions if item.pair_id in cluster.pair_ids]
    control = _sample_control(equivalent_pairs, len(cluster_regressions), seed)
    fixed = _replay_and_count_fixed(
        cluster_regressions,
        traces_by_pair_id,
        proposal,
        model,
        client,
        run_dir / f"{cluster.cluster_id}-cluster.sqlite",
        run_id=f"{cluster.cluster_id}-cluster",
        expect_fixed=True,
    )
    control_regressions = _replay_and_count_fixed(
        control,
        traces_by_pair_id,
        proposal,
        model,
        client,
        run_dir / f"{cluster.cluster_id}-control.sqlite",
        run_id=f"{cluster.cluster_id}-control",
        expect_fixed=False,
    )
    fixed_fraction = fixed / len(cluster_regressions) if cluster_regressions else 0.0
    accepted = fixed_fraction >= min_fix_rate and control_regressions == 0
    reason = "accepted"
    if fixed_fraction < min_fix_rate:
        reason = "insufficient_fix_rate"
    elif control_regressions:
        reason = "control_regression"
    diff_path = run_dir / f"patch_{cluster.cluster_id}.diff"
    with diff_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(proposal.diff_text)
    return PatchReport(
        cluster_id=cluster.cluster_id,
        diff_path=str(diff_path),
        fixed_fraction=round(fixed_fraction, 12),
        control_regressions=control_regressions,
        accepted=accepted,
        reason=reason,
    )


def write_patch_report(path: Path, reports: list[PatchReport]) -> str:
    payload = {
        "patches": [
            {
                "accepted": report.accepted,
                "cluster_id": report.cluster_id,
                "control_regressions": report.control_regressions,
                "diff_path": report.diff_path,
                "fixed_fraction": report.fixed_fraction,
                "reason": report.reason,
            }
            for report in sorted(reports, key=lambda item: item.cluster_id)
        ]
    }
    return _write_json(path, payload)


def apply_patch_to_trace(trace: TraceRecord, proposal: PatchProposal) -> TraceRecord:
    if proposal.target == "system":
        request = trace.request.model_copy(update={"system": proposal.replacement})
        return trace.model_copy(update={"request": request})
    message = trace.request.messages[0]
    content = message.content
    lines = content.splitlines(keepends=True)
    replaced = proposal.replacement + "".join(lines[1:]) if lines else proposal.replacement
    new_message = message.model_copy(update={"content": replaced})
    request = trace.request.model_copy(update={"messages": [new_message]})
    return trace.model_copy(update={"request": request})


def _template_info(template_id: str | None, traces: list[TraceRecord]) -> TemplateInfo:
    prompts = [trace.request.messages[0].content for trace in traces]
    prefix = _longest_common_prefix(prompts)
    suffix = _longest_common_suffix(prompts)
    median_length = sorted(len(prompt) for prompt in prompts)[len(prompts) // 2] if prompts else 0
    stable = bool(prompts) and (
        len(prefix) + len(suffix) >= 100
        or len(prefix) + len(suffix) >= 0.2 * median_length
    )
    return TemplateInfo(template_id=template_id, prefix=prefix, suffix=suffix, stable=stable)


def _longest_common_prefix(values: list[str]) -> str:
    if not values:
        return ""
    prefix = values[0]
    for value in values[1:]:
        index = 0
        limit = min(len(prefix), len(value))
        while index < limit and prefix[index] == value[index]:
            index += 1
        prefix = prefix[:index]
    return prefix


def _longest_common_suffix(values: list[str]) -> str:
    reversed_suffix = _longest_common_prefix([value[::-1] for value in values])
    return reversed_suffix[::-1]


def _patch_prompt(
    cluster: ClusterBucket,
    regressions: list[RegressionPair],
    target: PatchTarget,
) -> str:
    exemplars = [
        "\n".join(
            [
                f"Prompt: {regression.prompt[:300]}",
                f"Baseline: {regression.baseline_response[:300]}",
                f"Candidate: {regression.candidate_response[:300]}",
            ]
        )
        for regression in regressions
    ]
    return "\n\n".join(
        [
            f"Cluster: {cluster.name}",
            f"Description: {cluster.description}",
            f"Patch target: {target.kind}",
            f"Current target text:\n{target.text}",
            *exemplars,
            "Return JSON with target, replacement, and rationale.",
        ]
    )


def _parse_patch(text: str | None) -> dict[str, str]:
    if text is None:
        raise PatchParseError("empty patch response")
    try:
        payload = json.loads(text)
        validate_json_schema(payload, PATCH_SCHEMA)
    except (json.JSONDecodeError, JsonSchemaValidationError) as exc:
        raise PatchParseError(str(exc)) from exc
    return {
        "rationale": str(payload["rationale"]),
        "replacement": str(payload["replacement"]),
        "target": payload["target"],
    }


def _unified_diff(old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="old",
            tofile="new",
            lineterm="",
        )
    )


def _sample_control(items: list[RegressionPair], size: int, seed: int) -> list[RegressionPair]:
    if size <= 0:
        return []
    rng = random.Random(seed)
    ordered = sorted(items, key=lambda item: item.pair_id)
    if len(ordered) <= size:
        return ordered
    return sorted(rng.sample(ordered, size), key=lambda item: item.pair_id)


def _replay_and_count_fixed(
    items: list[RegressionPair],
    traces_by_pair_id: dict[str, TraceRecord],
    proposal: PatchProposal,
    model: ProviderModel,
    client: FakeClient,
    store_path: Path,
    *,
    run_id: str,
    expect_fixed: bool,
) -> int:
    if not items:
        return 0
    traces = [apply_patch_to_trace(traces_by_pair_id[item.pair_id], proposal) for item in items]
    store = ReplayStore(store_path)
    asyncio.run(
        replay(
            traces,
            model,
            1,
            client=client,
            store=store,
            config=ReplayConfig(run_id=run_id, seed=42, concurrency=1),
        )
    )
    calls = store.done_calls(run_id)
    by_trace = {call.trace_id: call for call in calls}
    count = 0
    for item in items:
        response = by_trace[item.pair_id].response or {}
        candidate = DiffResponse(text=response.get("text"))
        is_regression = _still_regresses(item, candidate)
        count += int(not is_regression if expect_fixed else is_regression)
    return count


def _still_regresses(item: RegressionPair, candidate: DiffResponse) -> bool:
    flipped = set(item.flipped_assertion_ids)
    if "json_valid" in flipped or "json_schema" in flipped:
        return not json_valid(candidate)
    baseline = DiffResponse(text=item.baseline_response)
    classification = classify_pair(
        baseline,
        candidate,
        embedder=FakeEmbedder(),
        threshold=0.90,
    )
    return not classification.deterministic_equal and classification.similarity < 0.90


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
