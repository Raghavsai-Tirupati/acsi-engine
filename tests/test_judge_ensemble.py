from __future__ import annotations

import csv
from pathlib import Path

from acsi.judge.calibration import (
    CalibrationSample,
    ingest_calibration_csv,
    write_calibration_sample,
)
from acsi.judge.ensemble import (
    aggregate_pair_outcomes,
    krippendorff_alpha_nominal,
    majority_outcome,
    raw_agreement_percent,
)


def test_majority_outcome_ties_and_all_abstain_are_unresolved() -> None:
    assert majority_outcome(["equivalent", "equivalent", "worse_minor"]) == "equivalent"
    assert majority_outcome(["equivalent", "worse_minor"]) == "unresolved"
    assert majority_outcome(["unresolved", "unresolved"]) == "unresolved"


def test_alpha_hits_hand_computed_value() -> None:
    matrix = [
        ["equivalent", "equivalent", "worse_minor"],
        ["equivalent", "equivalent", "equivalent"],
        ["worse_minor", None, "worse_minor"],
    ]

    # Observed disagreement = 2 disagreements / 7 judge-pairs.
    # Label counts are equivalent=5 and worse_minor=3, so expected disagreement
    # = 1 - (5/8)^2 - (3/8)^2 = 30/64. Alpha = 1 - (2/7)/(30/64).
    assert krippendorff_alpha_nominal(matrix) == 0.39047619047619053
    assert raw_agreement_percent(matrix) == 100 * 5 / 7


def test_high_agreement_seed_alpha_exceeds_low_agreement_seed() -> None:
    high = [
        ["equivalent", "equivalent", "equivalent"],
        ["worse_minor", "worse_minor", "worse_minor"],
        ["candidate_better", "candidate_better", "candidate_better"],
    ]
    low = [
        ["equivalent", "worse_minor", "candidate_better"],
        ["worse_minor", "candidate_better", "equivalent"],
        ["candidate_better", "equivalent", "worse_minor"],
    ]

    assert krippendorff_alpha_nominal(high) > krippendorff_alpha_nominal(low)


def test_calibration_ingest_produces_exact_expected_accuracies(tmp_path: Path) -> None:
    calibration = tmp_path / "labels.csv"
    _write_labels(
        calibration,
        [
            ("p1", "equivalent"),
            ("p2", "worse_minor"),
            ("p3", "candidate_better"),
        ],
    )
    votes = {
        "p1": {"j1": "equivalent", "j2": "equivalent", "j3": "worse_minor"},
        "p2": {"j1": "worse_minor", "j2": "equivalent", "j3": "worse_minor"},
        "p3": {"j1": "worse_minor", "j2": "candidate_better", "j3": "candidate_better"},
    }

    result = ingest_calibration_csv(calibration, votes)

    assert result["ensemble_accuracy"] == 1.0
    assert result["judge_accuracy"] == {"j1": 2 / 3, "j2": 2 / 3, "j3": 2 / 3}


def test_export_calibration_sample_writes_n_blank_labels(tmp_path: Path) -> None:
    output = tmp_path / "sample.csv"
    write_calibration_sample(
        output,
        [
            CalibrationSample("p1", "prompt 1", "a1", "b1"),
            CalibrationSample("p2", "prompt 2", "a2", "b2"),
            CalibrationSample("p3", "prompt 3", "a3", "b3"),
        ],
        2,
    )

    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["pair_id"] for row in rows] == ["p1", "p2"]
    assert [row["human_label"] for row in rows] == ["", ""]


def test_aggregate_pair_outcomes_is_deterministic() -> None:
    assert aggregate_pair_outcomes(
        {
            "p2": {"j2": "worse_minor", "j1": None},
            "p1": {"j1": "equivalent", "j2": "equivalent"},
        }
    ) == {"p1": "equivalent", "p2": "worse_minor"}


def _write_labels(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pair_id", "human_label"],
            lineterminator="\n",
        )
        writer.writeheader()
        for pair_id, label in rows:
            writer.writerow({"human_label": label, "pair_id": pair_id})
