from __future__ import annotations

from acsi.diff.deterministic import (
    DiffResponse,
    deterministic_pair_equivalence,
    is_refusal,
    json_valid,
    latency_ms,
    length_chars,
    regex_matches,
)


def test_deterministic_pair_normalized_text_equal() -> None:
    result = deterministic_pair_equivalence(
        DiffResponse(text="  hello   world\n"),
        DiffResponse(text="hello world"),
    )

    assert result.equivalent
    assert result.reason == "normalized_text_equal"


def test_deterministic_pair_canonical_json_equal() -> None:
    result = deterministic_pair_equivalence(
        DiffResponse(text='{"b": 2, "a": 1}'),
        DiffResponse(text='{"a":1,"b":2}'),
    )

    assert result.equivalent
    assert result.reason == "canonical_json_equal"


def test_deterministic_detectors() -> None:
    response = DiffResponse(
        text='{"ok": true}',
        finish_reason="stop",
        latency_ms=123,
    )

    assert json_valid(response)
    assert regex_matches(response, r'"ok"')
    assert length_chars(response) == 12
    assert latency_ms(response) == 123
    assert not is_refusal(response)
    assert is_refusal(DiffResponse(text="Sorry, but I cannot comply."))
    assert is_refusal(DiffResponse(text="blocked", finish_reason="content_filter"))
