"""Aggregate multiple independent DQN training seeds using clustered paired bootstrap CIs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Sequence


def percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires non-empty values")
    position = max(0.0, min(1.0, p)) * (len(sorted_values) - 1)
    lo, hi = int(position), int(-(-position // 1))
    fraction = position - lo
    return float(sorted_values[lo] * (1.0 - fraction) + sorted_values[hi] * fraction)


def extract_deltas(report: dict[str, Any], metric_list: str) -> list[float]:
    baseline = report["baseline"][metric_list]
    candidate = report["candidate"][metric_list]
    if not baseline or len(baseline) != len(candidate):
        raise ValueError(f"missing or mismatched paired metric {metric_list}")
    return [float(c) - float(b) for b, c in zip(baseline, candidate)]


def hierarchical_bootstrap_ci(
    reports: list[dict[str, Any]],
    metric_list: str,
    *,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, float]:
    """Resample training-seed clusters, then held-out episode pairs inside each cluster."""
    cluster_deltas = [extract_deltas(report, metric_list) for report in reports]
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        selected_clusters = rng.choices(cluster_deltas, k=len(cluster_deltas))
        selected_deltas = []
        for cluster in selected_clusters:
            selected_deltas.extend(rng.choices(cluster, k=len(cluster)))
        estimates.append(mean(selected_deltas))
    estimates.sort()
    return {
        "lower_95pct": percentile(estimates, 0.025),
        "upper_95pct": percentile(estimates, 0.975),
        "replicates": replicates,
        "resampling_unit": "training_seed cluster, then paired evaluation episode",
    }


def aggregate_metric(
    reports: list[dict[str, Any]],
    metric_list: str,
    mean_key: str,
    *,
    seed: int,
) -> dict[str, Any]:
    baseline_values = [
        float(value)
        for report in reports
        for value in report["baseline"][metric_list]
    ]
    candidate_values = [
        float(value)
        for report in reports
        for value in report["candidate"][metric_list]
    ]
    return {
        "baseline_mean_across_all_seed_episode_pairs": mean(baseline_values),
        "candidate_mean_across_all_seed_episode_pairs": mean(candidate_values),
        "candidate_minus_baseline_mean": mean(
            [c - b for b, c in zip(baseline_values, candidate_values)]
        ),
        "clustered_paired_bootstrap_95pct_ci": hierarchical_bootstrap_ci(
            reports, metric_list, seed=seed
        ),
        "pair_count": len(baseline_values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", default="artifacts/policy-evaluation-suite.json")
    parser.add_argument("--seed", type=int, default=20261099)
    args = parser.parse_args()
    if len(args.reports) < 2:
        parser.error("at least two independent training reports are required")

    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
    reports.sort(key=lambda item: int(item["environment"]["training_seed"]))
    training_seeds = [int(report["environment"]["training_seed"]) for report in reports]
    if len(set(training_seeds)) != len(training_seeds):
        parser.error("each report must use a distinct training seed")
    reference_eval_seeds = reports[0]["environment"]["evaluation_seeds"]
    reference_scenario = reports[0]["environment"]["scenario"]
    for report in reports:
        if report["environment"]["evaluation_seeds"] != reference_eval_seeds:
            parser.error("all training replicates must use the same fixed holdout seed list")
        if report["environment"]["scenario"] != reference_scenario:
            parser.error("all training replicates must use the same environment scenario")
        if report["environment"]["training_used_persisted_replay"]:
            parser.error("replicate reports must disable pre-existing replay restoration")

    per_run = []
    for report in reports:
        paired = report["paired_analysis"]
        win = paired["attacker_win_rate"]
        per_run.append({
            "training_seed": report["environment"]["training_seed"],
            "evaluation_seed_start": report["environment"]["evaluation_seed"],
            "evaluation_episodes": report["environment"]["evaluation_episodes_per_policy"],
            "baseline_attacker_win_rate": report["baseline"]["attacker_win_rate"],
            "candidate_attacker_win_rate": report["candidate"]["attacker_win_rate"],
            "attacker_win_rate_difference": win["candidate_minus_baseline_win_rate"],
            "per_seed_paired_bootstrap_95pct_ci": win["paired_bootstrap_95pct_ci"],
            "exact_two_sided_mcnemar_p_value": win["exact_two_sided_mcnemar_p_value"],
            "attacker_reward_difference": paired["mean_attacker_reward"]["candidate_minus_baseline_mean"],
            "defender_reward_difference": paired["mean_defender_reward"]["candidate_minus_baseline_mean"],
            "promotion_decision": report["promotion_decision"]["status"],
        })

    aggregate_win = aggregate_metric(
        reports, "attacker_wins_by_episode", "attacker_win_rate", seed=args.seed + 1
    )
    positive_seed_estimates = sum(
        1 for item in per_run if item["attacker_win_rate_difference"] > 0
    )
    aggregate_ci = aggregate_win["clustered_paired_bootstrap_95pct_ci"]
    robust_improvement = (
        positive_seed_estimates == len(per_run)
        and aggregate_win["candidate_minus_baseline_mean"] > 0
        and aggregate_ci["lower_95pct"] > 0
    )

    output = {
        "experiment": "multi-seed 1v1 random-initialization baseline vs trained DQN candidate",
        "training_seed_count": len(reports),
        "training_seeds": training_seeds,
        "shared_holdout_evaluation_seeds": reference_eval_seeds,
        "evaluation_episodes_per_training_seed_and_policy": reports[0]["environment"]["evaluation_episodes_per_policy"],
        "total_paired_evaluation_episodes_across_training_seeds": sum(
            int(report["environment"]["evaluation_episodes_per_policy"]) for report in reports
        ),
        "scenario": reference_scenario,
        "per_training_seed": per_run,
        "aggregate": {
            "attacker_win_rate": aggregate_win,
            "mean_attacker_reward": aggregate_metric(
                reports, "attacker_rewards_by_episode", "mean_attacker_reward", seed=args.seed + 2
            ),
            "mean_defender_reward": aggregate_metric(
                reports, "defender_rewards_by_episode", "mean_defender_reward", seed=args.seed + 3
            ),
            "training_seeds_with_positive_win_rate_difference": positive_seed_estimates,
            "improvement_demonstrated_across_training_seeds": robust_improvement,
            "interpretation": (
                "evidence_of_improvement_within_this_fixed_1v1_suite_across_all_training_seeds"
                if robust_improvement
                else "improvement_not_demonstrated_consistently_across_training_seeds"
            ),
        },
        "promotion_decision": {
            "status": "not_promoted",
            "reason": "Even a positive multi-seed result on this limited 1v1 suite is not a production promotion test or a comparison against a verified production checkpoint.",
        },
        "limitations": [
            "This benchmark's baseline is intentionally the random-initialized pair that the candidate starts from; it measures a training effect, not performance versus the tracked production policy.",
            "Three training seeds and one 1v1 scenario are still a bounded experiment and do not establish generalization to other team sizes.",
            "Clustered bootstrap resamples independent training-seed runs first, then paired evaluation episodes within each run.",
            "This experiment does not verify the durability of Render's production filesystem or backend restarts.",
        ],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "training_seeds": training_seeds,
        "total_paired_evaluation_episodes_across_training_seeds": output["total_paired_evaluation_episodes_across_training_seeds"],
        "attacker_win_rate_difference": aggregate_win["candidate_minus_baseline_mean"],
        "clustered_paired_bootstrap_95pct_ci": aggregate_ci,
        "positive_training_seed_differences": positive_seed_estimates,
        "interpretation": output["aggregate"]["interpretation"],
        "promotion": "not_promoted",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
