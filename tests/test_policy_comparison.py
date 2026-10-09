from scripts.evaluate_tracked_candidate_vs_production import count_critical_regressions


def test_count_critical_regression_when_role_ci_is_entirely_negative():
    attacker_ci = {"lower_95pct": -0.01, "upper_95pct": 0.05}
    defender_ci = {"lower_95pct": -0.14, "upper_95pct": -0.06}

    # The attacker result is inconclusive; only the defender role demonstrates
    # a regression whose entire 95% confidence interval is below zero.
    assert count_critical_regressions(
        attacker_ci, defender_ci, reward_acceptable=True
    ) == 1


def test_no_critical_regression_when_role_intervals_overlap_zero():
    overlapping = {"lower_95pct": -0.02, "upper_95pct": 0.03}

    assert count_critical_regressions(
        overlapping, overlapping, reward_acceptable=True
    ) == 0


def test_reward_regression_is_counted_separately():
    positive = {"lower_95pct": 0.01, "upper_95pct": 0.05}

    assert count_critical_regressions(
        positive, positive, reward_acceptable=False
    ) == 1
