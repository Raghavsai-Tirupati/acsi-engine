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
    assert regressions[0].signature.startswith("json_schema equivalent")
    assert compose_signature(
        ["a1", "a2"],
        "worse_critical",
        ["reason one.", "reason two."],
        "x" * 600,
    ) == f"a1 a2 worse_critical reason one. reason two. {'x' * 500}"


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
