from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from acsi.schemas import Message, TraceRecord, TraceRequest, TraceResponse

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
HONORIFIC_NAME_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
)
ENTITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", EMAIL_RE),
    ("PHONE", PHONE_RE),
    ("SSN", SSN_RE),
    ("NAME", HONORIFIC_NAME_RE),
)


@dataclass(frozen=True)
class ScrubResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ScrubRunResult:
    records: list[TraceRecord]
    report: dict[str, object]
    sha256: str


class RegexScrubber:
    def __init__(self) -> None:
        self._placeholders: dict[str, dict[str, str]] = {
            label: {} for label, _pattern in ENTITY_PATTERNS
        }
        self.counts: dict[str, int] = {label.lower(): 0 for label, _pattern in ENTITY_PATTERNS}

    def scrub_text(self, text: str | None) -> str | None:
        if text is None:
            return None
        scrubbed = text
        for label, pattern in ENTITY_PATTERNS:
            scrubbed = pattern.sub(
                lambda match, active_label=label: self._placeholder(
                    active_label,
                    match.group(0),
                ),
                scrubbed,
            )
        return scrubbed

    def _placeholder(self, label: str, value: str) -> str:
        values = self._placeholders[label]
        if value not in values:
            values[value] = f"[{label}_{len(values) + 1}]"
        self.counts[label.lower()] += 1
        return values[value]


def regex_scrub(text: str) -> ScrubResult:
    scrubber = RegexScrubber()
    scrubbed = scrubber.scrub_text(text) or ""
    return ScrubResult(text=scrubbed, counts=_nonzero_counts(scrubber.counts))


def scrub_traces(records: list[TraceRecord]) -> ScrubRunResult:
    scrubber = RegexScrubber()
    scrubbed = [_scrub_record(record, scrubber) for record in records]
    content = _records_jsonl(scrubbed)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    report: dict[str, object] = {
        "counts": _nonzero_counts(scrubber.counts),
        "entity_types": sorted(scrubber.counts),
        "records": len(scrubbed),
        "sha256": digest,
    }
    return ScrubRunResult(records=scrubbed, report=report, sha256=digest)


def write_scrub_artifacts(
    result: ScrubRunResult,
    *,
    traces_path: Path,
    report_path: Path,
) -> str:
    content = _records_jsonl(result.records)
    traces_path.parent.mkdir(parents=True, exist_ok=True)
    with traces_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    with Path(f"{traces_path}.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{digest}\n")

    report_payload = dict(result.report)
    report_payload["sha256"] = digest
    report_content = json.dumps(report_payload, sort_keys=True, separators=(",", ":"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{report_content}\n")
    report_digest = hashlib.sha256(f"{report_content}\n".encode()).hexdigest()
    with Path(f"{report_path}.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{report_digest}\n")
    return digest


def _scrub_record(record: TraceRecord, scrubber: RegexScrubber) -> TraceRecord:
    request = TraceRequest(
        provider=record.request.provider,
        model=record.request.model,
        system=scrubber.scrub_text(record.request.system),
        messages=[
            Message(role=message.role, content=scrubber.scrub_text(message.content) or "")
            for message in record.request.messages
        ],
        tools=record.request.tools,
        params=record.request.params,
    )
    response = TraceResponse(
        text=scrubber.scrub_text(record.response.text),
        tool_calls=record.response.tool_calls,
        finish_reason=record.response.finish_reason,
        usage=record.response.usage,
        latency_ms=record.response.latency_ms,
        served_model=record.response.served_model,
    )
    meta = record.meta.model_copy(deep=True)
    meta.pii_scrubbed = True
    return record.model_copy(
        update={
            "request": request,
            "response": response,
            "meta": meta,
        }
    )


def _nonzero_counts(counts: dict[str, int]) -> dict[str, int]:
    return {key: value for key, value in sorted(counts.items()) if value}


def _records_jsonl(records: list[TraceRecord]) -> str:
    lines = [
        json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        for record in sorted(records, key=lambda item: str(item.trace_id))
    ]
    return "".join(f"{line}\n" for line in lines)
