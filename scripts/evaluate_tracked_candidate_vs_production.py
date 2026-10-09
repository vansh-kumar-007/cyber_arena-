"""Compare a tracked candidate checkpoint pair to the policy loaded in production.

The baseline checkpoint filenames are confirmed by the Render startup logs:
models/final_marl_attacker_1v1.pt and
models/final_marl_defender_1v1_defender.pt.
The candidate pair is the separately tracked marl_1v1 checkpoint pair. This
script never promotes weights or changes the active policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from configs.network_config import ATTACK_TYPES, DEFENSE_TYPES
from env.network_env import NetworkEnvironment
from evaluate_dqn import evaluate_pair
from utils.model_registry import (
    ACTION_SCHEMA_VERSION,
    ENVIRONMENT_VERSION,
    FEATURE_SCHEMA_VERSION,
)

BASELINE_ID = "production-final-marl-1v1-accdf1b"
CANDIDATE_ID = "tracked-marl-1v1-checkpoint-accdf1b"
DEFAULT_EPISODES_PER_SEED_BLOCK = 100
DEFAULT_SEED_BLOCKS = (20361010, 20362010, 20363010)
BOOTSTRAP_REPLICATES = 10_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires non-empty values")
    position = max(0.0, min(1.0, p)) * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction)


def hierarchical_paired_bootstrap_ci(
    delta_blocks: list[list[float]], *, seed: int, replicates: int = BOOTSTRAP_REPLICATES
) -> dict[str, float]:
    """Resample three holdout-seed blocks, then paired episode deltas in each block."""
    if len(delta_blocks) < 3 or not delta_blocks or any(not block for block in delta_blocks):
        raise ValueError("at least three non-empty matched evaluation seed blocks are required")
    rng = random.Random(seed)
    estimates: list[float] = []
    count_blocks = len(delta_blocks)
    for _ in range(replicates):
        selected_blocks = rng.choices(delta_blocks, k=count_blocks)
        values: list[float] = []
        for block in selected_blocks:
            values.extend(rng.choices(block, k=len(block)))
        estimates.append(mean(values))
    estimates.sort()
    return {
        "lower_95pct": percentile(estimates, 0.025),
        "upper_95pct": percentile(estimates, 0.975),
        "replicates": replicates,
        "resampling_unit": "holdout seed block, then matched episode pair",
    }


def exact_mcnemar_p_value(better: int, worse: int) -> float:
    discordant = better + worse
    if discordant == 0:
        return 1.0
    tail = sum(math_comb(discordant, k) for k in range(min(better, worse) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def math_comb(n: int, k: int) -> int:
    # Avoid importing another statistics package solely for the exact paired test.
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - i + 1) // i
    return result


def group_deltas(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    episodes_per_block: int,
) -> list[list[float]]:
    if len(baseline_values) != len(candidate_values) or not baseline_values:
        raise ValueError("baseline and candidate results need equal non-empty seed-aligned values")
    if len(baseline_values) % episodes_per_block:
        raise ValueError("evaluation count must be divisible by episodes_per_block")
    deltas = [float(candidate) - float(baseline) for baseline, candidate in zip(baseline_values, candidate_values)]
    return [
        deltas[offset:offset + episodes_per_block]
        for offset in range(0, len(deltas), episodes_per_block)
    ]


def paired_metric(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    *,
    episodes_per_block: int,
    seed: int,
    favorable_direction: str = "higher",
) -> dict[str, Any]:
    if favorable_direction not in {"higher", "lower"}:
        raise ValueError("favorable_direction must be higher or lower")
    blocks = group_deltas(baseline_values, candidate_values, episodes_per_block)
    signed_blocks = blocks if favorable_direction == "higher" else [[-x for x in b] for b in blocks]
    all_deltas = [v for b in blocks for v in b]
    return {
        "baseline_mean": mean(float(v) for v in baseline_values),
        "candidate_mean": mean(float(v) for v in candidate_values),
        "candidate_minus_baseline_mean": mean(all_deltas),
        "favorable_direction": favorable_direction,
        "paired_hierarchical_bootstrap_95pct_ci": hierarchical_paired_bootstrap_ci(
            signed_blocks, seed=seed
        ) if favorable_direction == "higher" else {
            **{
                "lower_95pct": -hierarchical_paired_bootstrap_ci(blocks, seed=seed)["upper_95pct"],
                "upper_95pct": -hierarchical_paired_bootstrap_ci(blocks, seed=seed)["lower_95pct"],
                "replicates": BOOTSTRAP_REPLICATES,
                "resampling_unit": "holdout seed block, then matched episode pair",
            }
        },
        "paired_episode_count": len(all_deltas),
    }


def run_seed_blocks(
    *,
    name: str,
    attacker_path: Path,
    defender_path: Path,
    seed_blocks: Sequence[int],
    episodes_per_block: int,
) -> dict[str, Any]:
    reports = [
        evaluate_pair(
            name=f"{name}-seed-block-{block_seed}",
            attacker_path=str(attacker_path),
            defender_path=str(defender_path),
            episodes=episodes_per_block,
            seed=block_seed,
        )
        for block_seed in seed_blocks
    ]
    aggregate: dict[str, Any] = {}
    for key in (
        "episode_seeds",
        "attacker_wins_by_episode",
        "attacker_rewards_by_episode",
        "defender_rewards_by_episode",
        "episode_steps_by_episode",
        "detections_by_episode",
    ):
        aggregate[key] = [value for report in reports for value in report[key]]
    aggregate["attacker_win_rate"] = mean(aggregate["attacker_wins_by_episode"])
    aggregate["mean_attacker_reward"] = mean(aggregate["attacker_rewards_by_episode"])
    aggregate["mean_defender_reward"] = mean(aggregate["defender_rewards_by_episode"])
    aggregate["inference_latency_ms"] = {
        role: {
            "p50": float(np.percentile(
                [report["inference_latency_ms"][f"{role}_p50"] for report in reports], 50
            )),
            "p95": max(report["inference_latency_ms"][f"{role}_p95"] for report in reports),
            "p95_per_block": [report["inference_latency_ms"][f"{role}_p95"] for report in reports],
        }
        for role in ("attacker", "defender")
    }
    aggregate["inference_latency_ms"]["p95_max_across_roles"] = max(
        aggregate["inference_latency_ms"]["attacker"]["p95"],
        aggregate["inference_latency_ms"]["defender"]["p95"],
    )
    aggregate["safety_checks"] = {
        "invalid_action_count": sum(report["safety_checks"]["invalid_action_count"] for report in reports),
        "non_finite_output_count": sum(report["safety_checks"]["non_finite_output_count"] for report in reports),
        "forward_passes": {
            "attacker": sum(report["safety_checks"]["attacker_q_forward_passes"] for report in reports),
            "defender": sum(report["safety_checks"]["defender_q_forward_passes"] for report in reports),
        },
    }
    aggregate["seed_blocks"] = [
        {
            "base_seed": seed,
            "episode_count": episodes_per_block,
            "attacker_win_rate": report["attacker_win_rate"],
            "mean_attacker_reward": report["mean_attacker_reward"],
            "mean_defender_reward": report["mean_defender_reward"],
        }
        for seed, report in zip(seed_blocks, reports)
    ]
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-attacker",
        default="models/final_marl_attacker_1v1.pt",
    )
    parser.add_argument(
        "--baseline-defender",
        default="models/final_marl_defender_1v1_defender.pt",
    )
    parser.add_argument("--candidate-attacker", default="models/marl_1v1_attacker.pt")
    parser.add_argument("--candidate-defender", default="models/marl_1v1_defender.pt")
    parser.add_argument("--episodes-per-seed-block", type=int, default=DEFAULT_EPISODES_PER_SEED_BLOCK)
    parser.add_argument("--seed-blocks", type=int, nargs=3, default=list(DEFAULT_SEED_BLOCKS))
    parser.add_argument("--output", default="artifacts/production-policy-comparison.json")
    args = parser.parse_args()

    if not 100 <= args.episodes_per_seed_block <= 1000:
        parser.error("--episodes-per-seed-block must be between 100 and 1000")
    if len(set(args.seed_blocks)) != 3 or any(seed < 0 for seed in args.seed_blocks):
        parser.error("--seed-blocks must be three distinct non-negative integers")

    paths = {
        "baseline_attacker": Path(args.baseline_attacker),
        "baseline_defender": Path(args.baseline_defender),
        "candidate_attacker": Path(args.candidate_attacker),
        "candidate_defender": Path(args.candidate_defender),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        parser.error("checkpoint file(s) not found: " + ", ".join(missing))

    started = time.monotonic()
    environment = NetworkEnvironment(n_attackers=1, n_defenders=1, seed=args.seed_blocks[0], max_steps=50)
    state_size = len(environment.reset(seed=args.seed_blocks[0]))
    if state_size != 37 or len(ATTACK_TYPES) != 12 or len(DEFENSE_TYPES) != 12:
        raise RuntimeError("runtime feature/action schema differs from this evaluator's acceptance contract")

    # Every comparison below reuses the same ordered 300 episode seeds. The candidate's
    # attacker and defender are isolated against the corresponding current production
    # opponent so role improvements are measured independently rather than self-play only.
    baseline = run_seed_blocks(
        name="production-stable-policy",
        attacker_path=paths["baseline_attacker"],
        defender_path=paths["baseline_defender"],
        seed_blocks=args.seed_blocks,
        episodes_per_block=args.episodes_per_seed_block,
    )
    candidate_attacker = run_seed_blocks(
        name="candidate-attacker-vs-production-defender",
        attacker_path=paths["candidate_attacker"],
        defender_path=paths["baseline_defender"],
        seed_blocks=args.seed_blocks,
        episodes_per_block=args.episodes_per_seed_block,
    )
    candidate_defender = run_seed_blocks(
        name="production-attacker-vs-candidate-defender",
        attacker_path=paths["baseline_attacker"],
        defender_path=paths["candidate_defender"],
        seed_blocks=args.seed_blocks,
        episodes_per_block=args.episodes_per_seed_block,
    )

    attacker_win = paired_metric(
        baseline["attacker_wins_by_episode"],
        candidate_attacker["attacker_wins_by_episode"],
        episodes_per_block=args.episodes_per_seed_block,
        seed=20261081,
    )
    baseline_defender_success = [1 - int(value) for value in baseline["attacker_wins_by_episode"]]
    candidate_defender_success = [1 - int(value) for value in candidate_defender["attacker_wins_by_episode"]]
    defender_success = paired_metric(
        baseline_defender_success,
        candidate_defender_success,
        episodes_per_block=args.episodes_per_seed_block,
        seed=20261082,
    )
    attacker_reward = paired_metric(
        baseline["attacker_rewards_by_episode"],
        candidate_attacker["attacker_rewards_by_episode"],
        episodes_per_block=args.episodes_per_seed_block,
        seed=20261083,
    )
    defender_reward = paired_metric(
        baseline["defender_rewards_by_episode"],
        candidate_defender["defender_rewards_by_episode"],
        episodes_per_block=args.episodes_per_seed_block,
        seed=20261084,
    )

    holdout_seeds = baseline["episode_seeds"]
    baseline_hashes = {
        "attacker": sha256_file(paths["baseline_attacker"]),
        "defender": sha256_file(paths["baseline_defender"]),
    }
    candidate_hashes = {
        "attacker": sha256_file(paths["candidate_attacker"]),
        "defender": sha256_file(paths["candidate_defender"]),
    }
    candidate_latency_p95 = max(
        candidate_attacker["inference_latency_ms"]["attacker"]["p95"],
        candidate_defender["inference_latency_ms"]["defender"]["p95"],
    )
    candidate_invalid = (
        candidate_attacker["safety_checks"]["attacker_invalid_action_count"]
        + candidate_defender["safety_checks"]["defender_invalid_action_count"]
    )
    candidate_non_finite = (
        candidate_attacker["safety_checks"]["attacker_non_finite_output_count"]
        + candidate_defender["safety_checks"]["defender_non_finite_output_count"]
    )
    runtime_compatible = state_size == 37 and len(ATTACK_TYPES) == 12 and len(DEFENSE_TYPES) == 12
    # A weak/overlapping confidence interval is itself a rejection; never turn it
    # into a promotion decision, even when the point estimate is positive.
    acceptance_errors: list[str] = []
    if attacker_win["paired_hierarchical_bootstrap_95pct_ci"]["lower_95pct"] <= 0:
        acceptance_errors.append("attacker cross-play lower 95% CI is not strictly positive")
    if defender_success["paired_hierarchical_bootstrap_95pct_ci"]["lower_95pct"] <= 0:
        acceptance_errors.append("defender cross-play lower 95% CI is not strictly positive")
    if candidate_latency_p95 > 100.0:
        acceptance_errors.append("candidate p95 inference latency exceeds 100 ms")
    if candidate_invalid:
        acceptance_errors.append("candidate generated an invalid action")
    if candidate_non_finite:
        acceptance_errors.append("candidate generated a non-finite Q-value output")
    if not runtime_compatible:
        acceptance_errors.append("candidate/runtime schema compatibility check failed")

    reward_acceptable = (
        attacker_reward["paired_hierarchical_bootstrap_95pct_ci"]["lower_95pct"] > -5.0
        and defender_reward["paired_hierarchical_bootstrap_95pct_ci"]["lower_95pct"] > -5.0
    )
    if not reward_acceptable:
        acceptance_errors.append("attacker or defender reward lower 95% CI shows a regression worse than -5.0")

    paired_episodes = len(holdout_seeds)
    report = {
        "experiment": "tracked candidate vs production-loaded 1v1 policy, role-separated matched-seed cross-play",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": os.environ.get("GITHUB_SHA", "local-run"),
        "production_policy_identity": {
            "model_id": BASELINE_ID,
            "attacker_path": paths["baseline_attacker"].as_posix(),
            "defender_path": paths["baseline_defender"].as_posix(),
            "checkpoint_sha256": baseline_hashes,
            "identity_basis": "Render startup log recorded these exact checkpoint filenames on main commit accdf1b5846682773b6c5133f0c6df205265eeee; SHA-256 values were computed from the tracked checkpoint bytes in this evaluation checkout.",
        },
        "candidate_identity": {
            "model_id": CANDIDATE_ID,
            "attacker_path": paths["candidate_attacker"].as_posix(),
            "defender_path": paths["candidate_defender"].as_posix(),
            "checkpoint_sha256": candidate_hashes,
            "training_run_id": None,
            "training_metadata_note": "These are tracked candidate checkpoint files; no linked training run metadata is recorded in this repository.",
        },
        "environment": {
            "scenario_context": "1v1",
            "state_size": state_size,
            "attacker_action_count": len(ATTACK_TYPES),
            "defender_action_count": len(DEFENSE_TYPES),
            "max_steps_per_episode": 50,
            "holdout_seed_blocks": list(args.seed_blocks),
            "episodes_per_seed_block": args.episodes_per_seed_block,
            "paired_episodes": paired_episodes,
            "holdout_evaluation_seeds": holdout_seeds,
            "holdout_seed_set_sha256": sha256_json(holdout_seeds),
        },
        "baseline_production_selfplay": baseline,
        "candidate_attacker_vs_production_defender": candidate_attacker,
        "production_attacker_vs_candidate_defender": candidate_defender,
        "paired_analysis": {
            "attacker_crossplay_win_rate_delta": attacker_win,
            "defender_crossplay_success_rate_delta": defender_success,
            "attacker_reward_delta": attacker_reward,
            "defender_reward_delta": defender_reward,
        },
        "acceptance_evidence": {
            "acceptance_criteria_version": 1,
            "evaluation_kind": "candidate_comparison",
            "scenario_context": "1v1",
            "baseline_model_id": BASELINE_ID,
            "evaluation_suite_id": "cyberarena-production-policy-crossplay",
            "evaluation_suite_version": "1",
            "holdout_seed_set_sha256": sha256_json(holdout_seeds),
            "evaluation_seed_count": len(args.seed_blocks),
            "paired_episodes": paired_episodes,
            "attacker_crossplay_win_rate_delta_ci95": [
                attacker_win["paired_hierarchical_bootstrap_95pct_ci"]["lower_95pct"],
                attacker_win["paired_hierarchical_bootstrap_95pct_ci"]["upper_95pct"],
            ],
            "defender_crossplay_success_rate_delta_ci95": [
                defender_success["paired_hierarchical_bootstrap_95pct_ci"]["lower_95pct"],
                defender_success["paired_hierarchical_bootstrap_95pct_ci"]["upper_95pct"],
            ],
            "attacker_mean_reward_delta_ci95": [
                attacker_reward["paired_hierarchical_bootstrap_95pct_ci"]["lower_95pct"],
                attacker_reward["paired_hierarchical_bootstrap_95pct_ci"]["upper_95pct"],
            ],
            "defender_mean_reward_delta_ci95": [
                defender_reward["paired_hierarchical_bootstrap_95pct_ci"]["lower_95pct"],
                defender_reward["paired_hierarchical_bootstrap_95pct_ci"]["upper_95pct"],
            ],
            "inference_p95_latency_ms": candidate_latency_p95,
            "safety_checks": {
                "schema_compatible": runtime_compatible,
                "runtime_load_smoke_passed": True,
                "invalid_action_count": candidate_invalid,
                "non_finite_output_count": candidate_non_finite,
                "critical_regression_count": 0 if reward_acceptable else 1,
            },
            "environment_version": ENVIRONMENT_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "baseline_checkpoint_sha256": baseline_hashes,
            "candidate_checkpoint_sha256": candidate_hashes,
            "criteria_passed": not acceptance_errors,
            "rejection_reasons": acceptance_errors,
        },
        "promotion_decision": {
            "status": "not_promoted",
            "reason": "Offline evaluation is evidence only. No production model pointer was modified; a passing result still requires durable registry storage and an explicit operator promotion.",
        },
        "runtime": {
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "inference_latency_measurement_note": "Latency was measured on the CI evaluation host, not directly on Render production.",
        },
        "limitations": [
            "Candidate artifacts are tracked checkpoints without a linked training run ID/config record.",
            "The role-separated cross-play isolates candidate attacker behavior against the stable defender and candidate defender behavior against the stable attacker.",
            "This offline run cannot verify Render filesystem durability, live runtime health, or production request latency.",
            "A positive result applies to the fixed 1v1 seed suite, not automatically to 2v2/3v2 scenarios.",
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(out_path),
        "baseline_model_id": BASELINE_ID,
        "candidate_model_id": CANDIDATE_ID,
        "baseline_attacker_sha256": baseline_hashes["attacker"],
        "baseline_defender_sha256": baseline_hashes["defender"],
        "candidate_attacker_sha256": candidate_hashes["attacker"],
        "candidate_defender_sha256": candidate_hashes["defender"],
        "paired_episodes": paired_episodes,
        "attacker_delta_ci95": report["acceptance_evidence"]["attacker_crossplay_win_rate_delta_ci95"],
        "defender_delta_ci95": report["acceptance_evidence"]["defender_crossplay_success_rate_delta_ci95"],
        "candidate_p95_inference_latency_ms": candidate_latency_p95,
        "acceptance_criteria_passed": not acceptance_errors,
        "rejection_reasons": acceptance_errors,
        "promotion": "not_promoted",
    }, indent=2))
    # Report failure of performance acceptance in the artifact; do not fail CI
    # solely because the research candidate does not beat production. A load/schema
    # defect instead raises before a report can be emitted.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
