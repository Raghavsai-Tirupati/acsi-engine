from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from acsi.capture.python import AsyncJsonlWriter, capture_event
from acsi.config import load_workload_manifest
from acsi.replay.clients import CompletionClient, CompletionRequest, FakeClient, LiveClient
from acsi.replay.runner import estimate_call_cost_usd

DEFAULT_BENCHMARK_DIR = Path("benchmarks/oss-issues")
DEFAULT_CORPUS = DEFAULT_BENCHMARK_DIR / "corpus.jsonl"
DEFAULT_MANIFEST = DEFAULT_BENCHMARK_DIR / "acsi.yaml"
DEFAULT_SYSTEM_PROMPT = DEFAULT_BENCHMARK_DIR / "system_prompt.txt"
DEFAULT_OUTPUT = DEFAULT_BENCHMARK_DIR / "traces.jsonl"
TEMPLATE_ID = "oss-issue-summary-v1"
FALLBACK_INPUT_PRICE_PER_TOKEN = 15.0 / 1_000_000
FALLBACK_OUTPUT_PRICE_PER_TOKEN = 75.0 / 1_000_000


@dataclass(frozen=True)
class TraceGenerationResult:
    estimated_usd: float
    output_path: Path
    resolved_served_models: list[str]
    skipped_existing: int
    total_corpus_items: int
    written: int

    def to_payload(self) -> dict[str, object]:
        return {
            "estimated_usd": round(self.estimated_usd, 6),
            "output_path": str(self.output_path),
            "resolved_served_models": self.resolved_served_models,
            "skipped_existing": self.skipped_existing,
            "total_corpus_items": self.total_corpus_items,
            "written": self.written,
        }


def generate_traces(
    *,
    corpus_path: Path = DEFAULT_CORPUS,
    manifest_path: Path = DEFAULT_MANIFEST,
    system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT,
    output_path: Path = DEFAULT_OUTPUT,
    limit: int | None = None,
    max_usd: float = 15.0,
    yes: bool = False,
    fake: bool = False,
    client: CompletionClient | None = None,
    now: str | None = None,
) -> TraceGenerationResult:
    if not fake and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "Set ANTHROPIC_API_KEY to generate benchmark traces against the live baseline, "
            "or pass --fake for offline generation."
        )
    if limit is not None and limit < 0:
        raise ValueError("--limit must be non-negative.")

    manifest = load_workload_manifest(manifest_path)
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    corpus_items = _read_corpus(corpus_path)
    existing_keys = _existing_keys(output_path)
    pending = [
        item
        for item in corpus_items
        if (str(item["source_repo"]), int(item["issue_number"])) not in existing_keys
    ]
    skipped_existing = len(corpus_items) - len(pending)
    selected = pending if limit is None else pending[:limit]
    estimated_usd = _estimate_generation_cost(
        selected,
        manifest.baseline.provider,
        manifest.baseline.model,
    )
    if estimated_usd > max_usd:
        raise RuntimeError(
            f"Estimated generation cost ${estimated_usd:.4f} exceeds --max-usd ${max_usd:.4f}."
        )
    if not yes:
        raise RuntimeError(
            f"Estimated generation cost is ${estimated_usd:.4f}. Re-run with --yes to proceed."
        )

    active_client = client or (FakeClient(seed=42) if fake else LiveClient())
    writer = AsyncJsonlWriter(output_path)
    served_models: set[str] = set()
    timestamp = now or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    written = 0
    try:
        for item in selected:
            request = _completion_request(
                item,
                system_prompt,
                manifest.baseline.provider,
                manifest.baseline.model,
            )
            response = active_client.complete(request)
            served_models.add(response.served_model)
            capture_event(
                writer,
                _trace_payload(
                    item=item,
                    request=request,
                    response=response,
                    workload=manifest.workload,
                    timestamp=timestamp,
                ),
            )
            written += 1
    finally:
        writer.close()

    return TraceGenerationResult(
        estimated_usd=estimated_usd,
        output_path=output_path,
        resolved_served_models=sorted(served_models),
        skipped_existing=skipped_existing,
        total_corpus_items=len(corpus_items),
        written=written,
    )


def _read_corpus(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} line {line_number} is not a JSON object.")
            items.append(payload)
    return items


def _existing_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            meta = payload.get("meta", {})
            if not isinstance(meta, dict):
                continue
            source_repo = meta.get("source_repo")
            issue_number = meta.get("issue_number")
            if isinstance(source_repo, str) and isinstance(issue_number, int):
                keys.add((source_repo, issue_number))
    return keys


def _completion_request(
    item: dict[str, Any],
    system_prompt: str,
    provider: str,
    model: str,
) -> CompletionRequest:
    return CompletionRequest(
        provider=provider,
        model=model,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"Title: {item['title']}\n\nBody: {item['body']}",
            }
        ],
        params={"max_tokens": 700, "temperature": 0.2},
    )


def _trace_payload(
    *,
    item: dict[str, Any],
    request: CompletionRequest,
    response: Any,
    workload: str,
    timestamp: str,
) -> dict[str, Any]:
    issue_number = int(item["issue_number"])
    source_repo = str(item["source_repo"])
    return {
        "meta": {
            "html_url": str(item["html_url"]),
            "issue_number": issue_number,
            "pii_scrubbed": False,
            "source_repo": source_repo,
            "tags": ["benchmark", "oss-issue"],
            "template_id": TEMPLATE_ID,
            "truncated": bool(item["truncated"]),
        },
        "request": {
            "messages": request.messages,
            "model": request.model,
            "params": request.params,
            "provider": request.provider,
            "system": request.system,
            "tools": None,
        },
        "response": {
            "finish_reason": response.finish_reason,
            "latency_ms": response.latency_ms,
            "served_model": response.served_model,
            "text": response.text,
            "tool_calls": response.tool_calls,
            "usage": response.usage,
        },
        "source": "capture",
        "trace_id": str(uuid5(NAMESPACE_URL, f"oss-issue-summary:{source_repo}:{issue_number}")),
        "ts": timestamp,
        "workload": workload,
    }


def _estimate_generation_cost(items: list[dict[str, Any]], provider: str, model: str) -> float:
    total = 0.0
    for item in items:
        prompt = f"{item['title']}\n{item['body']}"
        input_tokens = max(1, (len(prompt) + 3) // 4)
        output_tokens = max(120, min(700, (len(prompt) + 19) // 20))
        estimate = estimate_call_cost_usd(
            provider,
            model,
            input_tokens,
            output_tokens,
            fake=False,
        )
        if estimate <= 0:
            estimate = (
                input_tokens * FALLBACK_INPUT_PRICE_PER_TOKEN
                + output_tokens * FALLBACK_OUTPUT_PRICE_PER_TOKEN
            )
        total += estimate
    return total


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OSS issue benchmark TraceRecords.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-usd", type=float, default=15.0)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--fake", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        result = generate_traces(
            corpus_path=args.corpus,
            manifest_path=args.manifest,
            system_prompt_path=args.system_prompt,
            output_path=args.output,
            limit=args.limit,
            max_usd=args.max_usd,
            yes=args.yes,
            fake=args.fake,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result.to_payload(), sort_keys=True))
    if result.resolved_served_models:
        print("Resolved baseline served_model: " + ", ".join(result.resolved_served_models))


if __name__ == "__main__":
    main()
