from __future__ import annotations

import json
from typing import Any


def normalized_text(value: str) -> str:
    return " ".join(value.split()).strip().lower()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))

