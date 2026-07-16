from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from acsi.schemas import TraceRecord


def iter_trace_records(path: Path) -> Iterable[TraceRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield TraceRecord.model_validate_json(stripped)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid TraceRecord: {exc}") from exc

