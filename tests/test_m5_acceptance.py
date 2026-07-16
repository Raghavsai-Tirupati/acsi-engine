from __future__ import annotations

from pathlib import Path

from acsi.diff.clustering import (
    AssertionFailure,
    CandidatePairRecord,
    FakeNamer,
    build_regression_set,
    cluster_regressions,
    name_clusters,
)
from acsi.replay.store import ReplayStore
from acsi.schemas import Severity


def test_end_to_end_injected_broken_json_regression_clusters_cleanly(tmp_path: Path) -> None:
    injected_ids = {f"pair-{index:03d}" for index in range(8)}
    records = [
        CandidatePairRecord(
            pair_id=f"pair-{index:03d}",
            prompt=(
                f"Prompt {index} "
                f"{'TRIGGER_JSON' if f'pair-{index:03d}' in injected_ids else ''}"
            ),
            baseline_response='{"ok":true}',
            candidate_response="{broken"
            if f"pair-{index:03d}" in injected_ids
            else '{"ok":true}',
            ensemble_outcome="worse_critical"
            if f"pair-{index:03d}" in injected_ids
            else "equivalent",
            judge_reasons=["Candidate emitted broken JSON."]
            if f"pair-{index:03d}" in injected_ids
            else [],
            assertion_failures=[
                AssertionFailure("json_valid", Severity.CRITICAL),
            ]
            if f"pair-{index:03d}" in injected_ids
            else [],
        )
        for index in range(100)
    ]

    regressions = build_regression_set(records)
    buckets = cluster_regressions(regressions, n_sampled_pairs=100, min_cluster_size=3)
    cluster_id = next(bucket.cluster_id for bucket in buckets if not bucket.unclustered)
    named, _stats = name_clusters(
        buckets,
        namer=FakeNamer(
            names={cluster_id: ("Broken JSON", "Responses are malformed JSON.")}
        ),
        store=ReplayStore(tmp_path / "cluster.sqlite"),
        run_id="run-1",
    )
    cluster = next(bucket for bucket in named if bucket.name == "Broken JSON")
    purity = len(set(cluster.pair_ids) & injected_ids) / len(cluster.pair_ids)

    assert purity >= 0.9
    assert cluster.severity == "worse_critical"
    assert cluster.share_of_sampled == 0.08
