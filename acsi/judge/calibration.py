from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from acsi.judge.ensemble import aggregate_pair_outcomes
from acsi.judge.rubric import CandidateOutcome


@dataclass(frozen=True)
class CalibrationSample:
    pair_id: str
    prompt: str
    response_a: str
    response_b: str


def ingest_calibration_csv(
    path: Path,
    votes_by_pair: dict[str, dict[str, CandidateOutcome | None]],
) -> dict[str, object]:
    human = _read_human_labels(path)
    ensemble = aggregate_pair_outcomes(votes_by_pair)
    judges = sorted({judge for votes in votes_by_pair.values() for judge in votes})
    return {
        "ensemble_accuracy": _accuracy(
            {pair_id: ensemble.get(pair_id) for pair_id in human},
            human,
        ),
        "human_labels": dict(sorted(human.items())),
        "judge_accuracy": {
            judge: _accuracy(
                {
                    pair_id: votes_by_pair.get(pair_id, {}).get(judge)
                    for pair_id in human
                },
                human,
            )
            for judge in judges
        },
    }


def write_calibration_sample(
    path: Path,
    samples: list[CalibrationSample],
    limit: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pair_id", "prompt", "response_a", "response_b", "human_label"],
            lineterminator="\n",
        )
        writer.writeheader()
        for sample in samples[:limit]:
            writer.writerow(
                {
                    "human_label": "",
                    "pair_id": sample.pair_id,
                    "prompt": sample.prompt,
                    "response_a": sample.response_a,
                    "response_b": sample.response_b,
                }
            )


def _read_human_labels(path: Path) -> dict[str, CandidateOutcome]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        labels: dict[str, CandidateOutcome] = {}
        for row in rows:
            label = (row.get("human_label") or "").strip()
            if not label:
                continue
            labels[str(row["pair_id"])] = _candidate_outcome(label)
        return labels


def _accuracy(
    predictions: dict[str, CandidateOutcome | None],
    human: dict[str, CandidateOutcome],
) -> float | None:
    total = 0
    correct = 0
    for pair_id, label in human.items():
        prediction = predictions.get(pair_id)
        if prediction is None:
            continue
        total += 1
        correct += int(prediction == label)
    if total == 0:
        return None
    return correct / total


def _candidate_outcome(value: str) -> CandidateOutcome:
    if value not in {
        "equivalent",
        "candidate_better",
        "worse_minor",
        "worse_critical",
        "unresolved",
    }:
        raise ValueError(f"Unsupported human label: {value}")
    return value
