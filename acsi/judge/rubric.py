from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema

PairwiseVerdict = Literal["a_better", "b_better", "equivalent"]
SeverityIfWorse = Literal["minor", "critical"]
CandidateOutcome = Literal[
    "equivalent",
    "candidate_better",
    "worse_minor",
    "worse_critical",
    "unresolved",
]

PAIRWISE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "severity_if_worse", "reason"],
    "properties": {
        "verdict": {"enum": ["a_better", "b_better", "equivalent"]},
        "severity_if_worse": {"enum": ["minor", "critical", None]},
        "reason": {"type": "string", "minLength": 1},
    },
}

CLASSIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["pass", "reason"],
    "properties": {
        "pass": {"type": "boolean"},
        "reason": {"type": "string", "minLength": 1},
    },
}


@dataclass(frozen=True)
class PairwiseJudgment:
    verdict: PairwiseVerdict
    severity_if_worse: SeverityIfWorse | None
    reason: str


@dataclass(frozen=True)
class ClassifierJudgment:
    passed: bool
    reason: str


class JudgeParseError(ValueError):
    pass


_BARE_JSON_INSTRUCTION = (
    "Output only the raw JSON object. Do not wrap it in markdown code fences "
    "(no ```), and do not add any prose before or after it."
)


def render_pairwise_rubric(prompt: str, response_a: str, response_b: str) -> str:
    return "\n\n".join(
        [
            "You are comparing two responses to the same prompt.",
            "Return only JSON with keys verdict, severity_if_worse, and reason.",
            (
                "verdict must be one of a_better, b_better, equivalent. "
                "severity_if_worse must be minor, critical, or null."
            ),
            _BARE_JSON_INSTRUCTION,
            f"Prompt:\n{prompt}",
            f"Response A:\n{response_a}",
            f"Response B:\n{response_b}",
        ]
    )


def render_classifier_rubric(prompt: str, response: str, criterion: str) -> str:
    return "\n\n".join(
        [
            "You are evaluating one response against a criterion.",
            "Return only JSON with keys pass and reason.",
            _BARE_JSON_INSTRUCTION,
            f"Criterion:\n{criterion}",
            f"Prompt:\n{prompt}",
            f"Response:\n{response}",
        ]
    )


def parse_pairwise_judgment(text: str | None) -> PairwiseJudgment:
    payload = _parse_json(text, PAIRWISE_SCHEMA, normalizer=_normalize_pairwise)
    return PairwiseJudgment(
        verdict=payload["verdict"],
        severity_if_worse=payload["severity_if_worse"],
        reason=str(payload["reason"]),
    )


def _normalize_pairwise(payload: dict[str, Any]) -> None:
    # Some judges emit the string "null" (or "none") for severity_if_worse rather
    # than a JSON null. The verdict is unambiguous; coerce the sentinel to None so
    # a valid verdict is not discarded on this quibble.
    severity = payload.get("severity_if_worse")
    if isinstance(severity, str) and severity.strip().lower() in {"null", "none", ""}:
        payload["severity_if_worse"] = None


def parse_classifier_judgment(text: str | None) -> ClassifierJudgment:
    payload = _parse_json(text, CLASSIFIER_SCHEMA)
    return ClassifierJudgment(
        passed=bool(payload["pass"]),
        reason=str(payload["reason"]),
    )


def map_position_verdict(
    judgment: PairwiseJudgment,
    *,
    candidate_position: Literal["a", "b"],
) -> CandidateOutcome:
    if judgment.verdict == "equivalent":
        return "equivalent"
    candidate_won = judgment.verdict == f"{candidate_position}_better"
    if candidate_won:
        return "candidate_better"
    return "worse_critical" if judgment.severity_if_worse == "critical" else "worse_minor"


_NOT_WORSE: frozenset[CandidateOutcome] = frozenset({"equivalent", "candidate_better"})
_WORSE: frozenset[CandidateOutcome] = frozenset({"worse_minor", "worse_critical"})


def reconcile_position_outcomes(
    left: CandidateOutcome,
    right: CandidateOutcome,
) -> tuple[CandidateOutcome | None, str | None]:
    """Reconcile the two order-swapped outcomes for one (pair, judge).

    SPEC-NOTE: the swap check enforces direction-of-harm consistency, not exact
    label match. Run 0a716021 abstained 112/213 pairs as "position_inconsistency"
    of which 88 were candidate_better-vs-equivalent — the judge merely disagreed
    on *how good* the candidate was across orderings, never on whether it
    regressed. Both orderings on the same side of the harm boundary reconcile to a
    conservative representative (the milder label unless both agree on the
    stronger one). Only a genuine flip across the boundary (not-worse vs worse)
    stays a position inconsistency and abstains.
    """
    if left in _NOT_WORSE and right in _NOT_WORSE:
        if left == "candidate_better" and right == "candidate_better":
            return "candidate_better", None
        return "equivalent", None
    if left in _WORSE and right in _WORSE:
        if left == "worse_critical" and right == "worse_critical":
            return "worse_critical", None
        return "worse_minor", None
    return None, "position_inconsistency"


def _parse_json(
    text: str | None,
    schema: dict[str, Any],
    *,
    normalizer: Any = None,
) -> dict[str, Any]:
    if text is None:
        raise JudgeParseError("Judge returned no text.")
    candidate = _extract_json_object(text)
    try:
        payload = json.loads(candidate)
        if normalizer is not None and isinstance(payload, dict):
            normalizer(payload)
        validate_json_schema(payload, schema)
    except (json.JSONDecodeError, JsonSchemaValidationError) as exc:
        raise JudgeParseError(str(exc)) from exc
    return payload


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```", re.DOTALL)


def _extract_json_object(text: str) -> str:
    """Recover the verdict JSON object from a model reply.

    Real judges (e.g. gemini) wrap valid verdict JSON in ```json fences or add a
    preamble; run 0a716021 lost 848/851 gemini verdicts to this. Strip a fenced
    block if present, then extract the first brace-balanced object (respecting
    strings), so a well-formed verdict is not discarded over its wrapping.
    """
    stripped = text.strip()
    fenced = _FENCE_RE.search(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    start = stripped.find("{")
    if start == -1:
        return stripped
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    return stripped[start:]
