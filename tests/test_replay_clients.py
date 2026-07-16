from __future__ import annotations

import pytest

from acsi.replay.clients import (
    CompletionRequest,
    FakeClient,
    LiveClient,
    RateLimitError,
    RegressionRule,
    live_client_keys_present,
)


def test_fake_client_noise_is_seeded_by_prompt_and_sample() -> None:
    request = CompletionRequest(
        provider="anthropic",
        model="fake-model",
        system=None,
        messages=[{"role": "user", "content": "summarize me"}],
        sample_index=0,
    )

    first = FakeClient(seed=7, noise=0.5).complete(request)
    second = FakeClient(seed=7, noise=0.5).complete(request)
    no_noise = FakeClient(seed=7, noise=0.0).complete(request)
    all_noise = FakeClient(seed=7, noise=1.0).complete(request)

    assert first == second
    assert no_noise.text and no_noise.text.startswith("summary:")
    assert all_noise.text and all_noise.text.startswith("paraphrased summary:")


def test_fake_client_regression_transforms_response() -> None:
    client = FakeClient(
        regressions=[
            RegressionRule(
                predicate=lambda prompt: "break-json" in prompt,
                transform=lambda _prompt, _text: "{broken",
            )
        ]
    )
    response = client.complete(
        CompletionRequest(
            provider="anthropic",
            model="fake-model",
            system=None,
            messages=[{"role": "user", "content": "please break-json"}],
        )
    )

    assert response.text == "{broken"


def test_fake_client_can_raise_retryable_rate_limit() -> None:
    client = FakeClient(fail_rate_limit_every=3)
    request = CompletionRequest(
        provider="anthropic",
        model="fake-model",
        system=None,
        messages=[{"role": "user", "content": "hello"}],
    )

    client.complete(request)
    client.complete(request)
    with pytest.raises(RateLimitError):
        client.complete(request)


@pytest.mark.skipif(
    not live_client_keys_present(),
    reason="live provider smoke test needs API keys",
)
def test_live_client_smoke() -> None:
    response = LiveClient().complete(
        CompletionRequest(
            provider="openai",
            model="gpt-4o-mini",
            system=None,
            messages=[{"role": "user", "content": "Return the word ok."}],
            params={"max_tokens": 3},
        )
    )

    assert response.served_model
