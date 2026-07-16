from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float
    confidence: float


def percentile_bootstrap_ci(
    values: list[float] | np.ndarray,
    *,
    b: int = 2_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> ConfidenceInterval:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("values must not be empty.")
    if b <= 0:
        raise ValueError("b must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, array.size, size=(b, array.size))
    means = array[sample_indices].mean(axis=1)
    alpha = 1 - confidence
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return ConfidenceInterval(
        mean=float(array.mean()),
        lower=lower,
        upper=upper,
        confidence=confidence,
    )


def rule_of_three_upper_bound(n: int) -> float:
    if n <= 0:
        raise ValueError("n must be positive.")
    return 3 / n


def rule_of_three(n: int) -> str:
    percent = rule_of_three_upper_bound(n) * 100
    return f"<= {percent:.1f}% at n={n:,}"


def mcnemar_exact(b: int, c: int) -> float:
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be nonnegative.")
    discordant = b + c
    if discordant == 0:
        return 1.0
    return float(binomtest(min(b, c), discordant, p=0.5, alternative="two-sided").pvalue)
