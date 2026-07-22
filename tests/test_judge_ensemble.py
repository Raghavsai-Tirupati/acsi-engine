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
from acsi.overrides import aggregate_judgment_rows, apply_overrides_to_judgments


def test_cross_boundary_and_all_abstain_are_unresolved() -> None:
    # A non-worse/worse split is a genuine direction-of-harm conflict -> unresolved
    # (previously plurality-resolved to "equivalent").
    assert majority_outcome(["equivalent", "equivalent", "worse_minor"]) == "unresolved"
    assert majority_outcome(["equivalent", "worse_minor"]) == "unresolved"
    assert majority_outcome(["unresolved", "unresolved"]) == "unresolved"


def test_direction_of_harm_reconciles_same_class_splits() -> None:
    # Non-worse split: judges agree the candidate did not regress, only on how good
    # -> conservative representative "equivalent".
    assert majority_outcome(["candidate_better", "equivalent"]) == "equivalent"
    # Unanimous candidate_better stays candidate_better.
    assert majority_outcome(["candidate_better", "candidate_better"]) == "candidate_better"
    # Worse split: agree on regression, only on how bad -> conservative "worse_minor".
    assert majority_outcome(["worse_minor", "worse_critical"]) == "worse_minor"
    assert majority_outcome(["worse_critical", "worse_critical"]) == "worse_critical"
    # Cross-boundary conflict stays unresolved.
    assert majority_outcome(["worse_minor", "equivalent"]) == "unresolved"
    # A single valid vote is unchanged by the reconciliation.
    assert majority_outcome(["candidate_better"]) == "candidate_better"
    assert majority_outcome(["worse_minor"]) == "worse_minor"
    assert majority_outcome(["candidate_better", "unresolved"]) == "candidate_better"


def test_human_override_takes_precedence_over_ensemble_rule() -> None:
    # Ensemble alone would resolve this cross-boundary split to "unresolved"; a human
    # override to "equivalent" is authoritative.
    rows = [
        {"pair_id": "p", "judge": "j1", "outcome": "worse_minor"},
        {"pair_id": "p", "judge": "j2", "outcome": "equivalent"},
    ]
    assert aggregate_judgment_rows(rows)["p"] == "unresolved"
    overridden = apply_overrides_to_judgments(
        rows, [{"pair_id": "p", "to_outcome": "equivalent"}]
    )
    assert aggregate_judgment_rows(overridden)["p"] == "equivalent"


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
    # Votes are within-harm-class so the ensemble resolves under the direction-of-harm
    # rule (a cross-boundary vote would abstain to "unresolved"). Each judge is wrong
    # on exactly one pair, keeping judge accuracy at 2/3.
    votes = {
        "p1": {"j1": "candidate_better", "j2": "candidate_better", "j3": "equivalent"},
        "p2": {"j1": "worse_minor", "j2": "worse_minor", "j3": "worse_critical"},
        "p3": {"j1": "candidate_better", "j2": "candidate_better", "j3": "candidate_better"},
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
