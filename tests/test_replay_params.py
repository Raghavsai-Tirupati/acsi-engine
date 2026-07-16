from __future__ import annotations

from acsi.replay.params import (
    ANTHROPIC_SAMPLING_REASON,
    summarize_param_transforms,
    transform_params,
)


def test_sonnet_five_strips_non_default_sampling_params() -> None:
    params, transforms = transform_params(
        "anthropic",
        "claude-sonnet-5",
        {"temperature": 0.2, "top_p": 0.9, "top_k": 50, "max_tokens": 200},
    )

    assert params == {"max_tokens": 200}
    assert {transform.path for transform in transforms} == {
        "params.temperature",
        "params.top_p",
        "params.top_k",
    }
    assert all(transform.reason == ANTHROPIC_SAMPLING_REASON for transform in transforms)


def test_param_transform_summary_deduplicates_with_counts() -> None:
    _, first = transform_params("anthropic", "claude-sonnet-5", {"temperature": 0.2})
    _, second = transform_params("anthropic", "claude-sonnet-5", {"temperature": 0.7})

    summary = summarize_param_transforms([*first, *second])

    assert summary == [
        {
            "provider": "anthropic",
            "model": "claude-sonnet-5",
            "path": "params.temperature",
            "action": "strip",
            "original": 0.2,
            "transformed": None,
            "reason": ANTHROPIC_SAMPLING_REASON,
            "count": 2,
        }
    ]
