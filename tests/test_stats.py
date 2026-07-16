from __future__ import annotations

import pytest

from acsi.stats import rule_of_three


def test_rule_of_three_formats_bound() -> None:
    assert rule_of_three(1000) == "<= 0.3% at n=1,000"


def test_rule_of_three_requires_positive_n() -> None:
    with pytest.raises(ValueError, match="positive"):
        rule_of_three(0)

