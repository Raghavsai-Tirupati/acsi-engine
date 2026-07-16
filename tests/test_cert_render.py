from __future__ import annotations

import pytest

from acsi.cert.render import assert_no_banned_words


def test_banned_certificate_words_are_rejected() -> None:
    with pytest.raises(ValueError, match="guaranteed"):
        assert_no_banned_words("This is guaranteed.")


def test_allowed_certificate_language_passes() -> None:
    assert_no_banned_words("This certifies the sampled workload against stated assertions.")

