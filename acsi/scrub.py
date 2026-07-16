from __future__ import annotations

import re
from dataclasses import dataclass, field

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
SIMPLE_NAME_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Dr)\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b")


@dataclass(frozen=True)
class ScrubResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)


def regex_scrub(text: str) -> ScrubResult:
    counts: dict[str, int] = {}

    def replace(pattern: re.Pattern[str], label: str, value: str) -> str:
        matches = pattern.findall(value)
        if matches:
            counts[label] = counts.get(label, 0) + len(matches)
        return pattern.sub(f"[REDACTED_{label.upper()}]", value)

    scrubbed = replace(EMAIL_RE, "email", text)
    scrubbed = replace(PHONE_RE, "phone", scrubbed)
    scrubbed = replace(SSN_RE, "ssn", scrubbed)
    scrubbed = replace(SIMPLE_NAME_RE, "name", scrubbed)
    return ScrubResult(text=scrubbed, counts=counts)

