from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

FIRST_NAMES = [
    "Avery",
    "Jordan",
    "Taylor",
    "Morgan",
    "Riley",
    "Casey",
    "Quinn",
    "Rowan",
    "Skyler",
    "Hayden",
]
ROLES = [
    "clinic intake volunteer",
    "transport coordinator",
    "meal delivery volunteer",
    "front desk support",
    "community outreach helper",
]
SKILLS = [
    "Spanish fluency",
    "HIPAA training",
    "night shift availability",
    "dispatch experience",
    "first aid certification",
]


def generate_records(count: int = 300, seed: int = 42) -> list[dict[str, object]]:
    rng = random.Random(seed)
    base_ts = datetime(2026, 7, 15, 18, 22, 3, tzinfo=UTC)
    records: list[dict[str, object]] = []
    stray_start = int(count * 0.9)
    for index in range(count):
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(FIRST_NAMES)}"
        role = rng.choice(ROLES)
        skill = rng.choice(SKILLS)
        hours = rng.choice([4, 8, 12, 16, 20])
        templated = index < stray_start
        if templated:
            prompt = (
                "Summarize this volunteer application as JSON with keys "
                "candidate, role_fit, availability, risks, and next_step.\n\n"
                f"Applicant: {name}\nPreferred role: {role}\nWeekly hours: {hours}\n"
                f"Notable skill: {skill}\nBackground: fabricated fixture applicant {index}."
            )
            template_id = "volunteer-json-summary-v1"
        else:
            prompt = (
                f"Create a concise coordinator note for applicant {name}, who can give "
                f"{hours} hours weekly and mentioned {skill} for {role}."
            )
            template_id = None

        summary = {
            "candidate": name,
            "role_fit": role,
            "availability": f"{hours} hours weekly",
            "risks": [],
            "next_step": "schedule coordinator screen",
        }
        trace_id = str(uuid5(NAMESPACE_URL, f"acsi-fixture-{seed}-{index}"))
        records.append(
            {
                "trace_id": trace_id,
                "ts": (base_ts + timedelta(seconds=index)).isoformat().replace("+00:00", "Z"),
                "source": "jsonl",
                "workload": "volunteer-application-summary",
                "request": {
                    "provider": "anthropic",
                    "model": "claude-haiku-4-5-20251001",
                    "system": "Return only compact JSON for coordinator review.",
                    "messages": [{"role": "user", "content": prompt}],
                    "tools": None,
                    "params": {"temperature": 0.2, "max_tokens": 512},
                },
                "response": {
                    "text": json.dumps(summary, sort_keys=True),
                    "tool_calls": None,
                    "finish_reason": "end_turn",
                    "usage": {"input_tokens": len(prompt.split()), "output_tokens": 40},
                    "latency_ms": 1800 + (index % 30),
                    "served_model": "claude-haiku-4-5-20251001",
                },
                "meta": {
                    "tags": ["fixture", "prod-shaped"],
                    "pii_scrubbed": False,
                    "template_id": template_id,
                },
            }
        )
    return records


def generate_invalid_lines(main_records: list[dict[str, object]], seed: int = 42) -> list[str]:
    lines: list[str] = []

    for index in range(5):
        record = _clone_with_trace_id(
            main_records[index],
            f"acsi-invalid-multi-turn-{seed}-{index}",
        )
        request = record["request"]
        assert isinstance(request, dict)
        messages = request["messages"]
        assert isinstance(messages, list)
        messages.append({"role": "assistant", "content": "extra turn"})
        lines.append(json.dumps(record, sort_keys=True))

    lines.extend(["{not valid json", "[malformed"])

    for index in range(3):
        lines.append(json.dumps(deepcopy(main_records[index]), sort_keys=True))

    for index in range(2):
        record = _clone_with_trace_id(
            main_records[index + 10],
            f"acsi-invalid-empty-response-{seed}-{index}",
        )
        record["response"] = {}
        lines.append(json.dumps(record, sort_keys=True))

    for index in range(3):
        record = _clone_with_trace_id(
            main_records[index + 20],
            f"acsi-valid-backfill-{seed}-{index}",
        )
        record["source"] = "backfill"
        record["response"] = {}
        meta = record["meta"]
        assert isinstance(meta, dict)
        meta["template_id"] = None
        lines.append(json.dumps(record, sort_keys=True))

    return lines


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clone_with_trace_id(record: dict[str, object], trace_key: str) -> dict[str, object]:
    cloned = deepcopy(record)
    cloned["trace_id"] = str(uuid5(NAMESPACE_URL, trace_key))
    return cloned


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ACSI trace fixtures.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/synthetic_traces.jsonl"),
    )
    parser.add_argument(
        "--invalid-output",
        type=Path,
        default=Path("tests/fixtures/invalid.jsonl"),
    )
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = generate_records(count=args.count, seed=args.seed)
    write_jsonl(args.output, records)
    write_lines(args.invalid_output, generate_invalid_lines(records, seed=args.seed))


if __name__ == "__main__":
    main()
