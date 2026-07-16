from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.table import Table

from acsi.schemas import TraceRecord, TraceSource

EXCLUSION_REASONS = ("multi_turn", "invalid_response", "duplicates", "invalid_schema")


@dataclass(frozen=True)
class SourceLocation:
    source: str
    line_number: int | None = None


@dataclass
class ExcludedRecord:
    reason: str
    location: SourceLocation
    payload: dict[str, Any]
    message: str | None = None

    def to_json_line(self) -> str:
        body: dict[str, Any] = {
            "reason": self.reason,
            "source": self.location.source,
            "record": self.payload,
        }
        if self.location.line_number is not None:
            body["line_number"] = self.location.line_number
        trace_id = self.payload.get("trace_id")
        if trace_id is not None:
            body["trace_id"] = trace_id
        if self.message:
            body["message"] = self.message
        return json.dumps(body, sort_keys=True, separators=(",", ":"))


@dataclass
class ImportSummary:
    lines_read: int = 0
    malformed: int = 0
    valid: int = 0
    exclusions: Counter[str] = field(default_factory=Counter)
    template_ids: Counter[str] = field(default_factory=Counter)
    templateless: int = 0
    source_counts: Counter[str] = field(default_factory=Counter)
    input_tokens: int = 0
    output_tokens: int = 0
    _date_min: datetime | None = None
    _date_max: datetime | None = None

    def record_valid(self, record: TraceRecord) -> None:
        self.valid += 1
        self.source_counts[str(record.source)] += 1
        if record.meta.template_id:
            self.template_ids[record.meta.template_id] += 1
        else:
            self.templateless += 1

        if record.response.usage:
            self.input_tokens += record.response.usage.input_tokens
            self.output_tokens += record.response.usage.output_tokens

        if self._date_min is None or record.ts < self._date_min:
            self._date_min = record.ts
        if self._date_max is None or record.ts > self._date_max:
            self._date_max = record.ts

    def record_exclusion(self, reason: str) -> None:
        self.exclusions[reason] += 1

    def to_payload(
        self,
        output_path: Path | None = None,
        sha256: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "lines_read": self.lines_read,
            "malformed": self.malformed,
            "valid": self.valid,
            "exclusions": {reason: self.exclusions.get(reason, 0) for reason in EXCLUSION_REASONS},
            "template_ids": dict(sorted(self.template_ids.items())),
            "distinct_template_ids": len(self.template_ids),
            "templateless": self.templateless,
            "source_counts": dict(sorted(self.source_counts.items())),
            "date_range": {
                "min": _format_datetime(self._date_min),
                "max": _format_datetime(self._date_max),
            },
            "token_totals": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
        }
        if output_path is not None:
            payload["output"] = str(output_path)
        if sha256 is not None:
            payload["sha256"] = sha256
        return payload


@dataclass
class ImportResult:
    records: list[TraceRecord] = field(default_factory=list)
    exclusions: list[ExcludedRecord] = field(default_factory=list)
    summary: ImportSummary = field(default_factory=ImportSummary)


class ImportAccumulator:
    def __init__(self) -> None:
        self.result = ImportResult()
        self._seen_trace_ids: set[str] = set()

    def add_malformed(self) -> None:
        self.result.summary.lines_read += 1
        self.result.summary.malformed += 1

    def add_payload(self, payload: dict[str, Any], location: SourceLocation) -> None:
        self.result.summary.lines_read += 1
        pre_validation_reason = _pre_validation_exclusion_reason(payload)
        if pre_validation_reason:
            self._exclude(pre_validation_reason, location, payload)
            return

        try:
            record = TraceRecord.model_validate(payload)
        except ValidationError as exc:
            self._exclude("invalid_schema", location, payload, _compact_validation_error(exc))
            return

        trace_id = str(record.trace_id)
        if trace_id in self._seen_trace_ids:
            self._exclude("duplicates", location, payload)
            return

        self._seen_trace_ids.add(trace_id)
        self.result.records.append(record)
        self.result.summary.record_valid(record)

    def extend_payloads(self, payloads: Iterable[dict[str, Any]], source: str) -> None:
        for payload in payloads:
            self.add_payload(payload, SourceLocation(source=source))

    def _exclude(
        self,
        reason: str,
        location: SourceLocation,
        payload: dict[str, Any],
        message: str | None = None,
    ) -> None:
        self.result.summary.record_exclusion(reason)
        self.result.exclusions.append(
            ExcludedRecord(reason=reason, location=location, payload=payload, message=message)
        )


def choose_output_path(
    records: list[TraceRecord],
    explicit_out: Path | None,
    workload: str | None = None,
) -> Path:
    if explicit_out is not None:
        return explicit_out
    if workload:
        return Path(".acsi") / "traces" / f"{workload}.jsonl"
    workloads = {record.workload for record in records}
    if not workloads:
        raise ValueError("No valid traces were imported; pass --out to choose an output path.")
    if len(workloads) > 1:
        raise ValueError("Multiple workloads imported; pass --out to choose an output path.")
    workload = next(iter(workloads))
    return Path(".acsi") / "traces" / f"{workload}.jsonl"


def write_import_artifacts(result: ImportResult, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(_record_json_line(record) + "\n" for record in result.records)
    output_path.write_text(content, encoding="utf-8")

    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    Path(f"{output_path}.sha256").write_text(f"{digest}\n", encoding="utf-8")

    exclusions_content = "".join(exclusion.to_json_line() + "\n" for exclusion in result.exclusions)
    Path(f"{output_path}.exclusions.jsonl").write_text(exclusions_content, encoding="utf-8")
    return digest


def inventory_table(payload: dict[str, Any]) -> Table:
    table = Table(title="ACSI Import Inventory")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Lines read", str(payload["lines_read"]))
    table.add_row("Malformed", str(payload["malformed"]))
    table.add_row("Valid", str(payload["valid"]))
    table.add_row("Exclusions", json.dumps(payload["exclusions"], sort_keys=True))
    table.add_row("Distinct template_ids", str(payload["distinct_template_ids"]))
    table.add_row("Templateless", str(payload["templateless"]))
    table.add_row("Date range", json.dumps(payload["date_range"], sort_keys=True))
    table.add_row("Token totals", json.dumps(payload["token_totals"], sort_keys=True))
    table.add_row("SHA-256", str(payload.get("sha256", "")))
    return table


def _pre_validation_exclusion_reason(payload: dict[str, Any]) -> str | None:
    request = payload.get("request")
    if not isinstance(request, dict):
        return "invalid_schema"

    messages = request.get("messages")
    if not isinstance(messages, list):
        return "invalid_schema"
    if len(messages) != 1 or not isinstance(messages[0], dict) or messages[0].get("role") != "user":
        return "multi_turn"

    source = payload.get("source")
    response = payload.get("response")
    if source != TraceSource.BACKFILL and not _has_response_content(response):
        return "invalid_response"

    return None


def _has_response_content(response: Any) -> bool:
    if not isinstance(response, dict):
        return False
    return bool(response.get("text")) or bool(response.get("tool_calls"))


def _record_json_line(record: TraceRecord) -> str:
    return json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _compact_validation_error(exc: ValidationError) -> str:
    first_error = exc.errors()[0]
    location = ".".join(str(part) for part in first_error.get("loc", ()))
    message = first_error.get("msg", "invalid schema")
    return f"{location}: {message}" if location else str(message)


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")
