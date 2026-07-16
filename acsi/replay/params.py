from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParamRule:
    provider: str
    model_prefix: str
    key: str
    action: str
    reason: str


PARAM_RULES: tuple[ParamRule, ...] = (
    ParamRule(
        provider="anthropic",
        model_prefix="claude-sonnet-5",
        key="temperature",
        action="strip",
        reason="Claude Sonnet 5 rejects non-default sampling params.",
    ),
    ParamRule(
        provider="anthropic",
        model_prefix="claude-sonnet-5",
        key="top_p",
        action="strip",
        reason="Claude Sonnet 5 rejects non-default sampling params.",
    ),
    ParamRule(
        provider="anthropic",
        model_prefix="claude-sonnet-5",
        key="top_k",
        action="strip",
        reason="Claude Sonnet 5 rejects non-default sampling params.",
    ),
)


def transform_params(
    provider: str, model: str, params: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    transformed = dict(params)
    changes: list[dict[str, Any]] = []
    for rule in PARAM_RULES:
        rule_applies = (
            provider == rule.provider
            and model.startswith(rule.model_prefix)
            and rule.key in transformed
        )
        if rule_applies:
            original = transformed.pop(rule.key)
            changes.append(
                {
                    "provider": provider,
                    "model": model,
                    "path": f"params.{rule.key}",
                    "action": rule.action,
                    "original": original,
                    "transformed": None,
                    "reason": rule.reason,
                }
            )
    return transformed, changes
