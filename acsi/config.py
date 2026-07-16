from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from acsi.schemas import WorkloadManifest


def load_workload_manifest(path: Path) -> WorkloadManifest:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _load_yaml(text, path)
    return WorkloadManifest.model_validate(payload)


def _load_yaml(text: str, path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise ValueError(
            f"{path} is not JSON and PyYAML is unavailable; use JSON-compatible YAML."
        ) from exc
    return yaml.safe_load(text)
