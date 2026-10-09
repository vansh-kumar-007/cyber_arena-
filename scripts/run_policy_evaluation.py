"""Run a bounded CPU-only baseline-vs-trained policy experiment.

The baseline is the same randomly initialized policy pair used to start training,
not a production checkpoint. The report deliberately does not promote a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Sequence

import numpy as np
import torch

# Running this file directly sets sys.path[0] to scripts/, so add the repository
# root before importing the project's modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_dqn
from agents.dqn_attacker import DQNAttacker
from agents.dqn_defender import DQNDefender
from env.network_env import NetworkEnvironment
from evaluate_dqn import evaluate_pair
from utils.experience_memory import ExperienceMemory

DEFAULT_SEED = 20261010
DEFAULT_TRAIN_EPISODES = 100
DEFAULT_EVAL_EPISODES = 100
BOOTSTRAP_REPLICATES = 10_000


def percentile(sorted_values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile for a non-empty sorted sequence."""
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = max(0.0, min(1.0, p)) * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction)


def bootstrap_mean_interval(
    differences: Sequence[float], *, seed: int, replicates: int = BOOTSTRAP_REPLICATES
) -> dict[str, float]:
    """Paired non-parametric bootstrap CI for a per-seed difference."""
    if not differences:
        raise ValueError("at least one paired difference is required")
    rng = random.Random(seed)
    count = len(differences)
    estimates = sorted(
        mean(rng.choices(differences, k=count)) for _ in range(replicates)
    )
    return {
        "lower_95pct": percentile(estimates, 0.025),
        "upper_95pct": percentile(estimates, 0.975),
        "replicates": replicates,
    }


def exact_mcnemar_p_value(better: int, worse: int) -> float:
    """Exact two-sided McNemar test from discordant paired binary outcomes."""
    if better < 0 or worse < 0:
        raise ValueError("discordant outcome counts cannot be negative")
    discordant = better + worse
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(better, worse) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_win_analysis(
    baseline_wins: Sequence[int], candidate_wins: Sequence[int], *, seed: int
) -> dict[str, object]:
    if len(baseline_wins) != len(candidate_wins) or not baseline_wins:
        raise ValueError("baseline/candidate outcomes must be non-empty and equally sized")
    if any(value not in (0, 1) for value in [*baseline_wins, *candidate_wins]):
        raise ValueError("win outcomes must be binary")
    deltas = [candidate - baseline for baseline, candidate in zip(baseline_wins, candidate_wins)]
    candidate_only = sum(1 for base, candidate in zip(baseline_wins, candidate_wins) if base == 0 and candidate == 1)
    baseline_only = sum(1 for base, candidate in zip(baseline_wins, candidate_wins) if base == 1 and candidate == 0)
    return {
        "paired_episodes": len(deltas),
        "baseline_win_rate": mean(baseline_wins),
        "candidate_win_rate": mean(candidate_wins),
        "candidate_minus_baseline_win_rate": mean(deltas),
        "paired_bootstrap_95pct_ci": bootstrap_mean_interval(deltas, seed=seed),
        "discordant_pairs_candidate_only": candidate_only,
        "discordant_pairs_baseline_only": baseline_only,
        "exact_two_sided_mcnemar_p_value": exact_mcnemar_p_value(candidate_only, baseline_only),
    }


def paired_metric_analysis(
    baseline_values: Sequence[float], candidate_values: Sequence[float], *, seed: int
) -> dict[str, object]:
    if len(baseline_values) != len(candidate_values) or not baseline_values:
        raise ValueError("baseline/candidate metrics must be non-empty and equally sized")
    differences = [candidate - baseline for baseline, candidate in zip(baseline_values, candidate_values)]
    return {
        "baseline_mean": mean(baseline_values),
        "candidate_mean": mean(candidate_values),
        "candidate_minus_baseline_mean": mean(differences),
        "paired_bootstrap_95pct_ci": bootstrap_mean_interval(differences, seed=seed),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_policy(brain) -> None:
    # Match the actual trainer hyperparameters before saving the starting baseline.
    brain.gamma = 0.95
    brain.learning_rate = 0.0005
    brain.target_update_freq = 5
    for group in brain.optimizer.param_groups:
        group["lr"] = brain.learning_rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--output", default="artifacts/policy-evaluation.json")
    parser.add_argument("--work-dir", default="artifacts/policy-evaluation-work")
    args = parser.parse_args()

    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.eval_seed is not None and args.eval_seed < 0:
        parser.error("--eval-seed must be non-negative")
    if not 1 <= args.train_episodes <= 500:
        parser.error("--train-episodes must be between 1 and 500")
    if not 20 <= args.eval_episodes <= 1000:
        parser.error("--eval-episodes must be between 20 and 1000")

    started = time.monotonic()
    output_path = Path(args.output)
    work_dir = Path(args.work_dir)
    model_dir = work_dir / "models"
    baseline_dir = work_dir / "baseline"
    for folder in (output_path.parent, model_dir, baseline_dir):
        folder.mkdir(parents=True, exist_ok=True)

    context = "1v1"
    # Use the random-initialization baseline deliberately to isolate the training
    # effect. It is not a comparison against the tracked production checkpoint pair.
    train_dqn._seed_everything(args.seed)
    probe_env = NetworkEnvironment(n_attackers=1, n_defenders=1, seed=args.seed, max_steps=50)
    state_size = len(probe_env.reset(seed=args.seed))
    baseline_attacker = DQNAttacker(state_size=state_size)
    baseline_defender = DQNDefender(state_size=state_size)
    configure_policy(baseline_attacker)
    configure_policy(baseline_defender)
    baseline_attacker_path = baseline_dir / "baseline_attacker.pt"
    baseline_defender_path = baseline_dir / "baseline_defender.pt"
    baseline_attacker.save(str(baseline_attacker_path))
    baseline_defender.save(str(baseline_defender_path))

    memory_path = work_dir / "agent-memory.sqlite3"
    memory = ExperienceMemory(memory_path)
    original_project_root = train_dqn.PROJECT_ROOT
    # Route model/checkpoint writes into the isolated CI artifact workspace, never
    # into the checked-out repository (models/ is gitignored in normal use).
    train_dqn.PROJECT_ROOT = work_dir
    try:
        training_metrics, trained_attacker, trained_defender = train_dqn.train_marl(
            n_attackers=1,
            n_defenders=1,
            num_episodes=args.train_episodes,
            save_models=True,
            seed=args.seed,
            memory=memory,
            resume=False,
            restore_replay=False,
        )
        memory_summary = memory.summary()
    finally:
        memory.close()
        train_dqn.PROJECT_ROOT = original_project_root

    candidate_attacker_path = model_dir / f"final_marl_attacker_{context}.pt"
    candidate_defender_path = model_dir / f"final_marl_defender_{context}_defender.pt"
    expected = [
        baseline_attacker_path, baseline_defender_path,
        candidate_attacker_path, candidate_defender_path,
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError("training failed to create required policy artifacts: " + ", ".join(missing))

    evaluation_seed = args.eval_seed if args.eval_seed is not None else args.seed + 100_000
    # Identical held-out environment seed set; it does not overlap the training seed
    # interval used by this bounded experiment.
    baseline_eval = evaluate_pair(
        name="random-initialized-baseline",
        attacker_path=str(baseline_attacker_path),
        defender_path=str(baseline_defender_path),
        episodes=args.eval_episodes,
        seed=evaluation_seed,
    )
    candidate_eval = evaluate_pair(
        name="trained-candidate",
        attacker_path=str(candidate_attacker_path),
        defender_path=str(candidate_defender_path),
        episodes=args.eval_episodes,
        seed=evaluation_seed,
    )

    win_analysis = paired_win_analysis(
        baseline_eval["attacker_wins_by_episode"],
        candidate_eval["attacker_wins_by_episode"],
        seed=args.seed + 17,
    )
    attacker_reward_analysis = paired_metric_analysis(
        baseline_eval["attacker_rewards_by_episode"],
        candidate_eval["attacker_rewards_by_episode"],
        seed=args.seed + 23,
    )
    defender_reward_analysis = paired_metric_analysis(
        baseline_eval["defender_rewards_by_episode"],
        candidate_eval["defender_rewards_by_episode"],
        seed=args.seed + 29,
    )
    if (
        win_analysis["candidate_minus_baseline_win_rate"] > 0
        and win_analysis["paired_bootstrap_95pct_ci"]["lower_95pct"] > 0
        and win_analysis["exact_two_sided_mcnemar_p_value"] < 0.05
    ):
        win_result = "improvement_evidence_for_attacker_win_rate_on_this_suite"
    elif (
        win_analysis["paired_bootstrap_95pct_ci"]["lower_95pct"] <= 0
        and win_analysis["paired_bootstrap_95pct_ci"]["upper_95pct"] >= 0
    ):
        win_result = "improvement_not_demonstrated_for_attacker_win_rate"
    else:
        win_result = "candidate_regressed_on_attacker_win_rate_or_evidence_is_mixed"

    checkpoints = {}
    for name, path in zip(
        ("baseline_attacker", "baseline_defender", "candidate_attacker", "candidate_defender"),
        expected,
    ):
        checkpoints[name] = {"file": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}

    report = {
        "experiment": "bounded 1v1 self-play baseline vs trained DQN candidate",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": os.environ.get("GITHUB_SHA", "local-run"),
        "environment": {
            "scenario": context,
            "state_size": state_size,
            "action_size_attacker": 12,
            "action_size_defender": 12,
            "max_steps_per_episode": 50,
            "training_seed": args.seed,
            "evaluation_seed": evaluation_seed,
            "evaluation_seeds": baseline_eval["episode_seeds"],
            "training_episodes": args.train_episodes,
            "evaluation_episodes_per_policy": args.eval_episodes,
            "training_used_persisted_replay": False,
            "candidate_started_from_same_random_initialization_as_baseline": True,
        },
        "algorithm": {
            "family": "Double DQN with prioritized experience replay",
            "gamma": 0.95,
            "learning_rate": 0.0005,
            "batch_size": 64,
            "target_update_frequency_episodes": 5,
            "replay_capacity": train_dqn.REPLAY_CAPACITY,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.05,
            "reward_normalization": "clip(raw_reward / 30, -3, 3)",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "cuda_available": torch.cuda.is_available(),
        },
        "checkpoint_artifacts": checkpoints,
        "persistent_memory": {
            "backend": memory_summary.get("backend", "sqlite"),
            "summary": memory_summary,
            "note": "This run uses a temporary CI workspace; it does not test production host disk durability.",
        },
        "baseline": baseline_eval,
        "candidate": candidate_eval,
        "paired_analysis": {
            "attacker_win_rate": win_analysis,
            "mean_attacker_reward": attacker_reward_analysis,
            "mean_defender_reward": defender_reward_analysis,
            "interpretation": win_result,
        },
        "training_run_summary": {
            "episodes_requested": args.train_episodes,
            "attacker_final_epsilon": trained_attacker.epsilon,
            "defender_final_epsilon": trained_defender.epsilon,
            "attacker_completed_episodes": trained_attacker.episode_count,
            "defender_completed_episodes": trained_defender.episode_count,
            "training_attacker_success_count": sum(training_metrics.attacker_success),
        },
        "promotion_decision": {
            "status": "not_promoted",
            "reason": "This isolated baseline experiment provides evidence only for the tested seed suite; deployment promotion requires separate approval and multi-seed acceptance criteria.",
        },
        "limitations": [
            "This experiment intentionally compares against the same random initialization that starts candidate training; it does not measure performance versus the tracked production policy.",
            "A single 1v1 scenario and one training seed do not establish generalization to other team sizes or environments.",
            "Candidate and baseline win-rate differences are paired by held-out environment seed; the bootstrap CI and exact McNemar test quantify sampling uncertainty, not all sources of uncertainty.",
            "This benchmark does not verify production database durability or restart recovery.",
        ],
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "training_episodes": args.train_episodes,
        "evaluation_episodes_per_policy": args.eval_episodes,
        "baseline_attacker_win_rate": baseline_eval["attacker_win_rate"],
        "candidate_attacker_win_rate": candidate_eval["attacker_win_rate"],
        "attacker_win_rate_difference": win_analysis["candidate_minus_baseline_win_rate"],
        "paired_bootstrap_95pct_ci": win_analysis["paired_bootstrap_95pct_ci"],
        "mcnemar_p_value": win_analysis["exact_two_sided_mcnemar_p_value"],
        "interpretation": win_result,
        "promotion": "not_promoted",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
