from __future__ import annotations

import json
from pathlib import Path

from acsi.importers.jsonl import import_jsonl_paths
from acsi.sampling import sample_traces, write_sample_artifacts
from acsi.schemas import SamplingConfig

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_traces.jsonl"


def test_sampling_dedups_near_duplicate_first_seen_and_samples_strata(
    tmp_path: Path,
) -> None:
    records = import_jsonl_paths([FIXTURE_PATH]).records[:12]
    duplicate = records[0].model_copy(
        update={"trace_id": records[11].trace_id},
        deep=True,
    )
    candidates = [*records[:11], duplicate]

    result = sample_traces(
        candidates,
        SamplingConfig(
            n=5,
            seed=7,
            stratify_by=["template_id", "input_length_bucket"],
            k_baseline=2,
        ),
    )

    assert result.sampling_mode == "stratified"
    assert result.report["dedup"]["collapsed_count"] == 1
    assert result.dedup_collapses[0].representative_trace_id == str(records[0].trace_id)
    assert len(result.records) == 5
    assert sum(stratum["sampled"] for stratum in result.report["strata"]) == 5

    digest = write_sample_artifacts(
        result.records,
        output_path=tmp_path / "sampled.jsonl",
        report_path=tmp_path / "sampling_report.json",
        report=result.report,
    )
    assert digest == result.sha256
    assert (tmp_path / "sampled.jsonl").read_bytes().endswith(b"\n")
    assert b"\r\n" not in (tmp_path / "sampling_report.json").read_bytes()


def test_sampling_exhaustive_when_requested_n_covers_available() -> None:
    records = import_jsonl_paths([FIXTURE_PATH]).records[:3]

    result = sample_traces(
        records,
        SamplingConfig(n=3, seed=42, stratify_by=["template_id"], k_baseline=2),
    )

    assert result.sampling_mode == "exhaustive"
    assert len(result.records) == 3
    assert json.dumps(result.report, sort_keys=True)
