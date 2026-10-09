"""Dependency-free policy-evaluation acceptance helpers."""
from __future__ import annotations


def count_critical_regressions(
    attacker_win_ci: dict[str, float],
    defender_success_ci: dict[str, float],
    *,
    reward_acceptable: bool,
) -> int:
    """Count dimensions whose full 95% confidence interval shows regression.

    A CI overlapping zero is inconclusive and fails the improvement gate, but
    is not labelled as a demonstrated regression. A fully negative role CI or
    reward failure counts as a critical regression in the safety record.
    """
    role_regressions = int(attacker_win_ci["upper_95pct"] < 0) + int(
        defender_success_ci["upper_95pct"] < 0
    )
    return role_regressions + int(not reward_acceptable)
