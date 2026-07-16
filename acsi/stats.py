from __future__ import annotations


def rule_of_three(n: int) -> str:
    if n <= 0:
        raise ValueError("n must be positive.")
    percent = (3 / n) * 100
    return f"<= {percent:.1f}% at n={n:,}"

