from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from acsi.judge.ensemble import aggregate_pair_outcomes
from acsi.judge.rubric import CandidateOutcome

OVERRIDABLE_OUTCOMES: set[str] = {
    "equivalent",
    "candidate_better",
    "worse_minor",
    "worse_critical",
    "unresolved",
}


@dataclass(frozen=True)
class OverrideAppendResult:
    row: dict[str, Any]
    from_outcome: CandidateOutcome


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_overrides(run_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(run_dir / "overrides.jsonl")


def append_override(
    run_dir: Path,
    *,
    pair_id: str,
    from_outcome: str,
    to_outcome: str,
    note: str | None = None,
    ts: str | None = None,
) -> dict[str, Any]:
    if to_outcome not in OVERRIDABLE_OUTCOMES:
        raise ValueError(f"Unsupported override outcome: {to_outcome}")
    row: dict[str, Any] = {
        "from_outcome": from_outcome,
        "pair_id": pair_id,
        "to_outcome": to_outcome,
        "ts": ts or stable_utc_now(),
    }
    if note:
        row["note"] = note
    path = run_dir / "overrides.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        )
        handle.write("\n")
    return row


def stable_utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def aggregate_judgment_rows(
    judgment_rows: list[dict[str, Any]],
    *,
    min_valid: int = 1,
) -> dict[str, CandidateOutcome]:
    votes: dict[str, dict[str, CandidateOutcome | None]] = {}
    for row in judgment_rows:
        outcome = row.get("outcome")
        parsed_outcome = None if outcome is None else str(outcome)
        votes.setdefault(str(row["pair_id"]), {})[str(row["judge"])] = parsed_outcome  # type: ignore[assignment]
    return aggregate_pair_outcomes(votes, min_valid=min_valid)


def latest_overrides(override_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in override_rows:
        pair_id = str(row.get("pair_id", ""))
        if pair_id:
            latest[pair_id] = row
    return latest


def apply_overrides_to_judgments(
    judgment_rows: list[dict[str, Any]],
    override_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest = latest_overrides(override_rows)
    if not latest:
        return [dict(row) for row in judgment_rows]
    effective: list[dict[str, Any]] = []
    for row in judgment_rows:
        copied = dict(row)
        override = latest.get(str(row.get("pair_id")))
        if override:
            copied["outcome"] = str(override["to_outcome"])
        effective.append(copied)
    return effective


def human_overrides_payload(override_rows: list[dict[str, Any]]) -> dict[str, Any]:
    latest = latest_overrides(override_rows)
    items: list[dict[str, Any]] = []
    for pair_id in sorted(latest):
        row = latest[pair_id]
        item = {
            "from": row.get("from_outcome"),
            "pair_id": pair_id,
            "to": row.get("to_outcome"),
        }
        if row.get("note"):
            item["note"] = row["note"]
        items.append(item)
    return {"count": len(items), "items": items}
