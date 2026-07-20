from __future__ import annotations

from pathlib import Path

import pytest

from acsi.diff.clustering import (
    AssertionFailure,
    CandidatePairRecord,
    ClusterInterrupted,
    FakeNamer,
    build_regression_set,
    cluster_regressions,
    compose_signature,
    name_clusters,
    write_clusters_json,
)
from acsi.diff.semantic import FakeEmbedder
from acsi.replay.store import ReplayStore
from acsi.schemas import Severity


def test_regression_set_sources_and_signature_order() -> None:
    records = [
        _record("judge", outcome="worse_minor", reason="Judge saw broken json."),
        _record(
            "assertion",
            outcome="equivalent",
            failures=[AssertionFailure("json_schema", Severity.CRITICAL)],
        ),
        _record(
            "mixed",
            outcome="unresolved",
            reason="Panel tied.",
            failures=[AssertionFailure("contains", Severity.MAJOR)],
        ),
    ]

    regressions = build_regression_set(records)

    assert [regression.detection_source for regression in regressions] == [
        "assertion",
        "judge",
        "mixed",
    ]
    assert regressions[0].signature == "json_schema"
    # Assertion-flagged regressions cluster purely on assertion id + normalized
    # reason template; outcome, judge reasons, and candidate text are excluded so
    # same-mechanism failures share one signature.
    assert (
        compose_signature(
            ["a1", "a2"],
            "worse_critical",
            ["reason one.", "reason two."],
            "x" * 600,
            assertion_reasons=["summary: 612 is longer than 400"],
        )
        == "a1 a2 summary: N is longer than N"
    )
    # Judge-only regressions keep outcome + judge reasons + candidate text.
    assert (
        compose_signature([], "worse_critical", ["reason one."], "x" * 600)
        == f"worse_critical reason one. {'x' * 500}"
    )


def test_assertion_reason_template_groups_same_mechanism_failures() -> None:
    # Two maxLength failures whose raw reasons differ only in the offending value
    # must share a signature so they cluster together, not fragment.
    records = [
        _record(
            f"schema-{i}",
            outcome="worse_critical",
            candidate=f"unique body {i}",
            failures=[
                AssertionFailure(
                    "summary-schema",
                    Severity.CRITICAL,
                    reason=f"summary: {600 + i} is longer than 400",
                )
            ],
        )
        for i in range(2)
    ]

    regressions = build_regression_set(records)

    assert regressions[0].signature == regressions[1].signature
    assert "summary: N is longer than N" in regressions[0].signature
    assert regressions[0].reason_labels == ["summary: 600 is longer than 400"]


def test_unresolved_cluster_is_not_labeled_an_assertion_failure() -> None:
    # A judge-only cluster of unresolved pairs must read as a panel outcome, never
    # as a shared assertion failure, and must not carry an assertion-style severity.
    regressions = build_regression_set(
        [_record(f"u{i}", outcome="unresolved", candidate="body") for i in range(6)]
    )
    buckets = cluster_regressions(
        regressions,
        n_sampled_pairs=100,
        min_cluster_size=3,
        name_by_outcome=True,
    )

    assert buckets
    for bucket in buckets:
        assert bucket.name == "Unresolved — panel could not decide"
        assert bucket.severity == "unresolved"
        assert "assertion failure" not in bucket.description
        assert "judge panel outcome" in bucket.description


def test_judge_worse_cluster_keeps_harm_severity() -> None:
    regressions = build_regression_set(
        [_record(f"w{i}", outcome="worse_critical", candidate="body") for i in range(6)]
    )
    buckets = cluster_regressions(
        regressions,
        n_sampled_pairs=100,
        min_cluster_size=3,
        name_by_outcome=True,
    )

    assert buckets
    assert all(bucket.severity == "worse_critical" for bucket in buckets)


def test_small_n_guard_emits_all_regressions_bucket() -> None:
    regressions = build_regression_set([_record(f"p{i}", outcome="worse_minor") for i in range(4)])

    buckets = cluster_regressions(regressions, n_sampled_pairs=100, min_cluster_size=3)

    assert len(buckets) == 1
    assert buckets[0].cluster_id == "all_regressions"
    assert buckets[0].skip_reason
    assert buckets[0].share_of_sampled == 0.04


def test_hdbscan_separates_two_failure_modes_and_unclustered() -> None:
    records = [
        *[
            _record(f"json-{i}", outcome="worse_critical", candidate="jsonmode alpha alpha")
            for i in range(5)
        ],
        *[
            _record(f"latency-{i}", outcome="worse_minor", candidate="latencymode beta beta")
            for i in range(5)
        ],
        _record("outlier-1", outcome="unresolved", candidate="rare gamma"),
        _record("outlier-2", outcome="unresolved", candidate="strange delta"),
    ]
    regressions = build_regression_set(records)

    buckets = cluster_regressions(
        regressions,
        n_sampled_pairs=20,
        embedder=FakeEmbedder(),
        min_cluster_size=3,
    )
    sizes = sorted(len(bucket.pair_ids) for bucket in buckets if not bucket.unclustered)
    unclustered = [bucket for bucket in buckets if bucket.unclustered]

    assert sizes == [5, 5]
    assert len(unclustered) == 1
    assert len(unclustered[0].pair_ids) == 2


def test_cluster_namer_retries_and_falls_back_on_double_failure(tmp_path: Path) -> None:
    regressions = build_regression_set([_record(f"p{i}", outcome="worse_minor") for i in range(6)])
    buckets = cluster_regressions(regressions, n_sampled_pairs=10, min_cluster_size=3)
    cluster_id = buckets[0].cluster_id
    namer = FakeNamer(
        names={cluster_id: ("Broken JSON", "Responses are malformed JSON.")},
        malformed_attempts={(cluster_id, 0)},
    )
    named, stats = name_clusters(
        buckets,
        namer=namer,
        store=ReplayStore(tmp_path / "retry.sqlite"),
        run_id="run-1",
    )

    assert named[0].name == "Broken JSON"
    assert stats["parse_failures"] == 0

    fallback, fallback_stats = name_clusters(
        buckets,
        namer=FakeNamer(
            malformed_attempts={(cluster_id, 0), (cluster_id, 1)}
        ),
        store=ReplayStore(tmp_path / "fallback.sqlite"),
        run_id="run-1",
    )

    assert fallback[0].name == cluster_id
    assert fallback[0].parse_failure
    assert fallback_stats["parse_failures"] == 1


def test_cluster_naming_checkpoint_resume_is_byte_identical(tmp_path: Path) -> None:
    regressions = build_regression_set([_record(f"p{i}", outcome="worse_minor") for i in range(8)])
    buckets = cluster_regressions(regressions, n_sampled_pairs=10, min_cluster_size=3)
    cluster_id = buckets[0].cluster_id
    names = {cluster_id: ("Broken JSON", "Responses are malformed JSON.")}
    namer = FakeNamer(names=names)
    control, control_stats = name_clusters(
        buckets,
        namer=namer,
        store=ReplayStore(tmp_path / "control.sqlite"),
        run_id="run-1",
    )
    write_clusters_json(tmp_path / "control" / "clusters.json", control, stats=control_stats)

    resume_store = ReplayStore(tmp_path / "resume.sqlite")
    with pytest.raises(ClusterInterrupted):
        name_clusters(
            buckets,
            namer=FakeNamer(names=names),
            store=resume_store,
            run_id="run-1",
            interrupt_after_dispatches=1,
        )
    resumed, resumed_stats = name_clusters(
        buckets,
        namer=FakeNamer(names=names),
        store=resume_store,
        run_id="run-1",
    )
    write_clusters_json(tmp_path / "resume" / "clusters.json", resumed, stats=resumed_stats)

    assert resumed_stats["cache_hits"] == 1
    assert resumed_stats["dispatched"] == 0
    assert (tmp_path / "resume" / "clusters.json").read_bytes() == (
        tmp_path / "control" / "clusters.json"
    ).read_bytes()


def test_clusters_json_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    regressions = build_regression_set([_record(f"p{i}", outcome="worse_minor") for i in range(8)])
    buckets = cluster_regressions(regressions, n_sampled_pairs=10, min_cluster_size=3)

    for dirname in ("a", "b"):
        named, stats = name_clusters(
            buckets,
            namer=FakeNamer(seed=7),
            store=ReplayStore(tmp_path / f"{dirname}.sqlite"),
            run_id="run-1",
        )
        write_clusters_json(tmp_path / dirname / "clusters.json", named, stats=stats)

    assert (tmp_path / "a" / "clusters.json").read_bytes() == (
        tmp_path / "b" / "clusters.json"
    ).read_bytes()


def _record(
    pair_id: str,
    *,
    outcome="equivalent",
    reason: str | None = None,
    failures: list[AssertionFailure] | None = None,
    candidate: str = "candidate response",
) -> CandidatePairRecord:
    return CandidatePairRecord(
        pair_id=pair_id,
        prompt=f"Prompt {pair_id}",
        baseline_response="baseline response",
        candidate_response=candidate,
        ensemble_outcome=outcome,
        judge_reasons=[reason] if reason else [],
        assertion_failures=failures or [],
    )
