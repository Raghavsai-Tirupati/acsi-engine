from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from acsi.importers.common import ImportAccumulator, ImportResult, SourceLocation

WarningCallback = Callable[[str], None]


def import_jsonl_paths(paths: list[Path], warn: WarningCallback | None = None) -> ImportResult:
    accumulator = ImportAccumulator()
    for path in paths:
        _import_jsonl_path(path, accumulator, warn)
    return accumulator.result


def _import_jsonl_path(
    path: Path,
    accumulator: ImportAccumulator,
    warn: WarningCallback | None,
) -> None:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                accumulator.add_malformed()
                if warn:
                    warn(f"Skipping malformed JSON in {path} at line {line_number}.")
                continue

            if not isinstance(payload, dict):
                accumulator.add_payload(
                    {"value": payload},
                    SourceLocation(source=str(path), line_number=line_number),
                )
                continue

            accumulator.add_payload(
                payload,
                SourceLocation(source=str(path), line_number=line_number),
            )
