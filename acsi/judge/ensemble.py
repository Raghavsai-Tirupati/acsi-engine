from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from acsi.judge.rubric import CandidateOutcome

ENSEMBLE_OUTCOMES: tuple[CandidateOutcome, ...] = (
    "equivalent",
    "candidate_better",
    "worse_minor",
    "worse_critical",
    "unresolved",
)


@dataclass
class JudgeAccumulator:
    verdict_counts: Counter[str] = field(default_factory=Counter)
    abstentions: int = 0
    parse_failures: int = 0
    position_inconsistencies: int = 0
    call_errors: int = 0
    pairs_seen: int = 0

    @property
    def position_inconsistency_rate(self) -> float:
        if self.pairs_seen == 0:
            return 0.0
        return self.position_inconsistencies / self.pairs_seen


def majority_outcome(
    votes: list[CandidateOutcome],
    *,
    min_valid: int = 1,
) -> CandidateOutcome:
    # SPEC-NOTE: the panel floor is capped by the number of judges that actually
    # participated (votes present). A pair falls to "unresolved" only when a judge
    # that WAS asked abstained/errored, dropping valid verdicts below the floor —
    # e.g. one of two judges exhausts retries. Pairs seen by a single judge (or
    # min_valid=1, the default) keep the pre-existing single-vote behavior, so no
    # healthy run changes.
    floor = max(1, min(min_valid, len(votes)))
    eligible = [vote for vote in votes if vote != "unresolved"]
    if len(eligible) < floor:
        return "unresolved"
    counts = Counter(eligible)
    top_count = max(counts.values())
    winners = [outcome for outcome, count in counts.items() if count == top_count]
    return winners[0] if len(winners) == 1 else "unresolved"


def aggregate_pair_outcomes(
    votes_by_pair: dict[str, dict[str, CandidateOutcome | None]],
    *,
    min_valid: int = 1,
) -> dict[str, CandidateOutcome]:
    outcomes: dict[str, CandidateOutcome] = {}
    for pair_id in sorted(votes_by_pair):
        outcomes[pair_id] = majority_outcome(
            [
                vote if vote is not None else "unresolved"
                for vote in votes_by_pair[pair_id].values()
            ],
            min_valid=min_valid,
        )
    return outcomes


def outcome_counts(outcomes: dict[str, CandidateOutcome]) -> dict[str, int]:
    counts = Counter(outcomes.values())
    return {outcome: counts.get(outcome, 0) for outcome in ENSEMBLE_OUTCOMES}


def raw_agreement_percent(matrix: list[list[CandidateOutcome | None]]) -> float | None:
    agreeing = 0
    total = 0
    for labels in matrix:
        present = [label for label in labels if label is not None]
        for left_index, left in enumerate(present):
            for right in present[left_index + 1 :]:
                total += 1
                if left == right:
                    agreeing += 1
    if total == 0:
        return None
    return 100 * agreeing / total


def krippendorff_alpha_nominal(matrix: list[list[CandidateOutcome | None]]) -> float | None:
    disagreements = 0
    observed_pairs = 0
    label_counts: Counter[CandidateOutcome] = Counter()
    for labels in matrix:
        present = [label for label in labels if label is not None]
        label_counts.update(present)
        for left_index, left in enumerate(present):
            for right in present[left_index + 1 :]:
                observed_pairs += 1
                disagreements += int(left != right)
    total_labels = sum(label_counts.values())
    if observed_pairs == 0 or total_labels <= 1:
        return None
    observed_disagreement = disagreements / observed_pairs
    expected_disagreement = 1 - sum(
        (count / total_labels) ** 2 for count in label_counts.values()
    )
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return 1 - observed_disagreement / expected_disagreement


def agreement_matrix(
    votes_by_pair: dict[str, dict[str, CandidateOutcome | None]],
) -> list[list[CandidateOutcome | None]]:
    judges = sorted({judge for votes in votes_by_pair.values() for judge in votes})
    rows: list[list[CandidateOutcome | None]] = []
    for pair_id in sorted(votes_by_pair):
        rows.append([votes_by_pair[pair_id].get(judge) for judge in judges])
    return rows


def summarize_judge_stats(
    accumulators: dict[str, JudgeAccumulator],
    votes_by_pair: dict[str, dict[str, CandidateOutcome | None]],
) -> dict[str, object]:
    ensemble = aggregate_pair_outcomes(votes_by_pair)
    matrix = agreement_matrix(votes_by_pair)
    return {
        "ensemble": {
            "outcome_counts": outcome_counts(ensemble),
            "raw_agreement_percent": _stable_optional(raw_agreement_percent(matrix)),
            "krippendorff_alpha": _stable_optional(krippendorff_alpha_nominal(matrix)),
        },
        "judges": {
            judge: {
                "abstentions": acc.abstentions,
                "call_errors": acc.call_errors,
                "parse_failures": acc.parse_failures,
                "position_inconsistency_rate": round(acc.position_inconsistency_rate, 12),
                "position_inconsistencies": acc.position_inconsistencies,
                "verdict_counts": dict(sorted(acc.verdict_counts.items())),
            }
            for judge, acc in sorted(accumulators.items())
        },
    }


def empty_accumulators(judges: list[str]) -> dict[str, JudgeAccumulator]:
    return {judge: JudgeAccumulator() for judge in judges}


def append_vote(
    votes_by_pair: dict[str, dict[str, CandidateOutcome | None]],
    pair_id: str,
    judge: str,
    outcome: CandidateOutcome | None,
) -> None:
    votes_by_pair.setdefault(pair_id, {})[judge] = outcome


def grouped_votes() -> defaultdict[str, dict[str, CandidateOutcome | None]]:
    return defaultdict(dict)


def _stable_optional(value: float | None) -> float | None:
    return None if value is None else round(float(value), 12)
