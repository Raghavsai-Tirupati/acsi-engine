from __future__ import annotations

import pytest

from acsi.stats import (
    mcnemar_exact,
    percentile_bootstrap_ci,
    rule_of_three,
    rule_of_three_upper_bound,
)


def test_percentile_bootstrap_ci_hand_computed_constant_values() -> None:
    ci = percentile_bootstrap_ci([0, 0, 1, 1], b=200, seed=1)

    assert ci.mean == 0.5
    assert ci.lower == 0.0
    assert ci.upper == 1.0
    assert ci.confidence == 0.95


def test_rule_of_three_formats_bound() -> None:
    assert rule_of_three(1000) == "<= 0.3% at n=1,000"
    assert rule_of_three_upper_bound(1000) == 0.003


def test_rule_of_three_requires_positive_n() -> None:
    with pytest.raises(ValueError, match="positive"):
        rule_of_three(0)


def test_mcnemar_exact_matches_hand_computed_binomial() -> None:
    assert mcnemar_exact(1, 3) == 0.625
    assert mcnemar_exact(0, 0) == 1.0
