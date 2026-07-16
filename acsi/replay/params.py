from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

ParamAction = Literal["strip", "clamp", "rename"]

ANTHROPIC_SAMPLING_REASON = (
    "Provider returns HTTP 400 for non-default temperature/top_p/top_k, verified July 2026."
)


@dataclass(frozen=True)
class ParamRule:
    provider: str
    model_prefix: str
    key: str
    action: ParamAction
    reason: str
    min_value: float | None = None
    max_value: float | None = None
    new_key: str | None = None


@dataclass(frozen=True)
class AppliedParamTransform:
    provider: str
    model: str
    path: str
    action: ParamAction
    original: Any | None
    transformed: Any | None
    reason: str

    def dedupe_key(self) -> tuple[str, str, str, str, str]:
        return (self.provider, self.model, self.path, self.action, self.reason)


PARAM_RULES: tuple[ParamRule, ...] = tuple(
    ParamRule(
        provider="anthropic",
        model_prefix=model_prefix,
        key=key,
        action="strip",
        reason=ANTHROPIC_SAMPLING_REASON,
    )
    for model_prefix in ("claude-sonnet-5", "claude-opus-4-7")
    for key in ("temperature", "top_p", "top_k")
)


def transform_params(
    provider: str,
    model: str,
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[AppliedParamTransform]]:
    transformed = dict(params)
    changes: list[AppliedParamTransform] = []
    for rule in PARAM_RULES:
        if not _rule_applies(rule, provider, model, transformed):
            continue
        original = transformed.get(rule.key)
        if rule.action == "strip":
            transformed.pop(rule.key)
            new_value = None
        elif rule.action == "clamp":
            new_value = _clamp_value(original, rule)
            transformed[rule.key] = new_value
        elif rule.action == "rename":
            if not rule.new_key:
                raise ValueError("rename rules require new_key.")
            new_value = transformed.pop(rule.key)
            transformed[rule.new_key] = new_value
        else:
            raise ValueError(f"Unsupported param action: {rule.action}")
        changes.append(
            AppliedParamTransform(
                provider=provider,
                model=model,
                path=f"params.{rule.key}",
                action=rule.action,
                original=original,
                transformed=new_value,
                reason=rule.reason,
            )
        )
    return transformed, changes


def summarize_param_transforms(
    transforms: list[AppliedParamTransform],
) -> list[dict[str, Any]]:
    counts = Counter(transform.dedupe_key() for transform in transforms)
    first_by_key: dict[tuple[str, str, str, str, str], AppliedParamTransform] = {}
    for transform in transforms:
        first_by_key.setdefault(transform.dedupe_key(), transform)
    summaries: list[dict[str, Any]] = []
    for key in sorted(counts):
        transform = first_by_key[key]
        summaries.append(
            {
                "provider": transform.provider,
                "model": transform.model,
                "path": transform.path,
                "action": transform.action,
                "original": transform.original,
                "transformed": transform.transformed,
                "reason": transform.reason,
                "count": counts[key],
            }
        )
    return summaries


def _rule_applies(
    rule: ParamRule,
    provider: str,
    model: str,
    params: dict[str, Any],
) -> bool:
    return (
        provider == rule.provider
        and model.startswith(rule.model_prefix)
        and rule.key in params
    )


def _clamp_value(value: Any, rule: ParamRule) -> Any:
    if not isinstance(value, int | float):
        return value
    if rule.min_value is not None:
        value = max(value, rule.min_value)
    if rule.max_value is not None:
        value = min(value, rule.max_value)
    return value
