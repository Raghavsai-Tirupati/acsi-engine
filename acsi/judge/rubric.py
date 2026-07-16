from __future__ import annotations

import json
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


def render_pairwise_rubric(prompt: str, response_a: str, response_b: str) -> str:
    return "\n\n".join(
        [
            "You are comparing two responses to the same prompt.",
            "Return only JSON with keys verdict, severity_if_worse, and reason.",
            (
                "verdict must be one of a_better, b_better, equivalent. "
                "severity_if_worse must be minor, critical, or null."
            ),
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
            f"Criterion:\n{criterion}",
            f"Prompt:\n{prompt}",
            f"Response:\n{response}",
        ]
    )


def parse_pairwise_judgment(text: str | None) -> PairwiseJudgment:
    payload = _parse_json(text, PAIRWISE_SCHEMA)
    return PairwiseJudgment(
        verdict=payload["verdict"],
        severity_if_worse=payload["severity_if_worse"],
        reason=str(payload["reason"]),
    )


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


def _parse_json(text: str | None, schema: dict[str, Any]) -> dict[str, Any]:
    if text is None:
        raise JudgeParseError("Judge returned no text.")
    try:
        payload = json.loads(text)
        validate_json_schema(payload, schema)
    except (json.JSONDecodeError, JsonSchemaValidationError) as exc:
        raise JudgeParseError(str(exc)) from exc
    return payload
