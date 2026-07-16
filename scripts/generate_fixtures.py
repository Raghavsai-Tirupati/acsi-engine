from __future__ import annotations

import argparse
import json
import random
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


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ACSI trace fixtures.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/synthetic_traces.jsonl"),
    )
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_jsonl(args.output, generate_records(count=args.count, seed=args.seed))


if __name__ == "__main__":
    main()
