from __future__ import annotations

import json
from pathlib import Path

import pytest

from acsi.diff.clustering import AssertionFailure, ClusterBucket, RegressionPair
from acsi.patch import (
    FakePatcher,
    PatchInterrupted,
    PatchProposal,
    detect_templates,
    propose_patch,
    select_patch_target,
    validate_patch,
    write_patch_report,
)
from acsi.replay.clients import CompletionRequest, CompletionResponse, FakeClient
from acsi.replay.store import ReplayStore
from acsi.schemas import ProviderModel, Severity, TraceRecord
from scripts.generate_fixtures import generate_records


def test_template_detection_recovers_fixture_template_and_null_group() -> None:
    traces = [TraceRecord.model_validate(record) for record in generate_records(count=300)]

    detection = detect_templates(traces)
    info = detection.templates["volunteer-json-summary-v1"]

    assert info.stable
    assert info.prefix.startswith("Summarize this volunteer application as JSON")
    assert info.suffix == "."
    assert detection.null_template_count == 30


def test_no_template_id_below_threshold_skips_patching() -> None:
    records = generate_records(count=4)
    prompts = [
        "alpha",
        "bravo volunteer note",
        "charlie schedule summary",
        "delta intake paragraph",
    ]
    for index, record in enumerate(records):
        record["meta"]["template_id"] = None
        record["request"]["messages"][0]["content"] = prompts[index]
    traces = [TraceRecord.model_validate(record) for record in records]

    detection = detect_templates(traces)
    target = select_patch_target(traces, detection)

    assert detection.skip_reason == "no_stable_template"
    assert target.skip_reason == "no_stable_template"


def test_patch_proposer_retries_malformed_then_valid(tmp_path: Path) -> None:
    cluster = _cluster()
    target = select_patch_target(_traces(1, trigger_count=1), detect_templates(_traces(1, 1)))
    proposal, stats = propose_patch(
        cluster=cluster,
        regressions=[_regression("p0")],
        target=target,
        patcher=FakePatcher(
            replacements={
                "cluster-json": (
                    "system",
                    "Return JSON. STRICT_JSON_MODE",
                    "Require strict JSON.",
                )
            },
            malformed_attempts={("cluster-json", 0)},
        ),
        store=ReplayStore(tmp_path / "patch.sqlite"),
        run_id="run-1",
    )

    assert proposal is not None
    assert proposal.replacement == "Return JSON. STRICT_JSON_MODE"
    assert not proposal.parse_failure
    assert stats["parse_failures"] == 0


def test_patch_validation_accepts_fixed_cluster_and_clean_control(tmp_path: Path) -> None:
    traces = _traces(6, trigger_count=3)
    pair_ids = [str(trace.trace_id) for trace in traces]
    traces_by_pair_id = {str(trace.trace_id): trace for trace in traces}
    cluster_regressions = [_regression(pair_id) for pair_id in pair_ids[:3]]
    control_pairs = [_control_pair(pair_id) for pair_id in pair_ids[3:]]
    proposal = PatchProposal(
        cluster_id="cluster-json",
        target="system",
        replacement="Return JSON. STRICT_JSON_MODE",
        rationale="Require strict JSON.",
        diff_text="--- old\n+++ new\n",
    )

    report = validate_patch(
        proposal=proposal,
        cluster=_cluster(pair_ids[:3]),
        regressions=cluster_regressions,
        equivalent_pairs=control_pairs,
        traces_by_pair_id=traces_by_pair_id,
        model=ProviderModel(provider="fake", model="candidate"),
        client=MarkerJsonClient(),
        run_dir=tmp_path,
    )

    assert report.accepted
    assert report.fixed_fraction == 1.0
    assert report.control_regressions == 0
    assert Path(report.diff_path).exists()


def test_patch_validation_rejects_control_regression(tmp_path: Path) -> None:
    traces = _traces(6, trigger_count=3)
    pair_ids = [str(trace.trace_id) for trace in traces]
    traces_by_pair_id = {str(trace.trace_id): trace for trace in traces}
    proposal = PatchProposal(
        cluster_id="cluster-json",
        target="system",
        replacement="Return JSON. STRICT_JSON_MODE BAD_CONTROL",
        rationale="Bad patch.",
        diff_text="--- old\n+++ new\n",
    )

    report = validate_patch(
        proposal=proposal,
        cluster=_cluster(pair_ids[:3]),
        regressions=[_regression(pair_id) for pair_id in pair_ids[:3]],
        equivalent_pairs=[_control_pair(pair_id) for pair_id in pair_ids[3:]],
        traces_by_pair_id=traces_by_pair_id,
        model=ProviderModel(provider="fake", model="candidate"),
        client=MarkerJsonClient(),
        run_dir=tmp_path,
    )
    digest = write_patch_report(tmp_path / "patches" / "patch_report.json", [report])

    assert not report.accepted
    assert report.reason == "control_regression"
    assert digest
    assert (tmp_path / "patches" / "patch_report.json.sha256").exists()


def test_patch_proposal_checkpoint_resume_uses_cached_call(tmp_path: Path) -> None:
    cluster = _cluster()
    traces = _traces(1, trigger_count=1)
    target = select_patch_target(traces, detect_templates(traces))
    replacements = {
        "cluster-json": ("system", "Return JSON. STRICT_JSON_MODE", "Require strict JSON.")
    }
    control, _ = propose_patch(
        cluster=cluster,
        regressions=[_regression("p0")],
        target=target,
        patcher=FakePatcher(replacements=replacements),
        store=ReplayStore(tmp_path / "control.sqlite"),
        run_id="run-1",
    )
    store = ReplayStore(tmp_path / "resume.sqlite")
    with pytest.raises(PatchInterrupted):
        propose_patch(
            cluster=cluster,
            regressions=[_regression("p0")],
            target=target,
            patcher=FakePatcher(replacements=replacements),
            store=store,
            run_id="run-1",
            interrupt_after_dispatches=1,
        )
    resumed, stats = propose_patch(
        cluster=cluster,
        regressions=[_regression("p0")],
        target=target,
        patcher=FakePatcher(replacements=replacements),
        store=store,
        run_id="run-1",
    )

    assert resumed == control
    assert stats["cache_hits"] == 1
    assert stats["dispatched"] == 0


class MarkerJsonClient(FakeClient):
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        prompt = request.prompt_text
        if (
            "BAD_CONTROL" in prompt
            and "TRIGGER_JSON" not in prompt
            or "TRIGGER_JSON" in prompt
            and "STRICT_JSON_MODE" not in prompt
        ):
            text = "{broken"
        else:
            text = json.dumps({"ok": True}, sort_keys=True)
        return CompletionResponse(
            text=text,
            tool_calls=None,
            finish_reason="stop",
            usage={"input_tokens": 1, "output_tokens": 1},
            latency_ms=1,
            served_model=request.model,
        )


def _traces(count: int, trigger_count: int) -> list[TraceRecord]:
    records = generate_records(count=count)
    traces: list[TraceRecord] = []
    for index, record in enumerate(records):
        record["trace_id"] = f"00000000-0000-0000-0000-{index:012d}"
        record["request"]["system"] = "Return JSON."
        suffix = " TRIGGER_JSON" if index < trigger_count else ""
        record["request"]["messages"][0]["content"] += suffix
        traces.append(TraceRecord.model_validate(record))
    return traces


def _cluster(pair_ids: list[str] | None = None) -> ClusterBucket:
    return ClusterBucket(
        cluster_id="cluster-json",
        label=0,
        name="Broken JSON",
        description="Candidate emits broken JSON.",
        pair_ids=pair_ids or ["p0", "p1", "p2"],
        signatures=["json_valid worse_critical broken"],
        severity="worse_critical",
        share_of_sampled=0.5,
    )


def _regression(pair_id: str) -> RegressionPair:
    return RegressionPair(
        pair_id=pair_id,
        prompt=f"Prompt {pair_id} TRIGGER_JSON",
        baseline_response=json.dumps({"ok": True}, sort_keys=True),
        candidate_response="{broken",
        ensemble_outcome="worse_critical",
        judge_reasons=["Broken JSON."],
        flipped_assertion_ids=["json_valid"],
        assertion_failures=[AssertionFailure("json_valid", Severity.CRITICAL)],
        detection_source="mixed",
        signature="json_valid worse_critical Broken JSON. {broken",
        severity_rank=3,
    )


def _control_pair(pair_id: str) -> RegressionPair:
    return RegressionPair(
        pair_id=pair_id,
        prompt=f"Prompt {pair_id}",
        baseline_response=json.dumps({"ok": True}, sort_keys=True),
        candidate_response=json.dumps({"ok": True}, sort_keys=True),
        ensemble_outcome="equivalent",
        judge_reasons=[],
        flipped_assertion_ids=[],
        assertion_failures=[],
        detection_source="judge",
        signature="equivalent",
        severity_rank=0,
    )
