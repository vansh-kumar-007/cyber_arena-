"""Compare two DQN checkpoint pairs on identical, fixed-seed holdout episodes.

Example:
python evaluate_dqn.py \
  --baseline-attacker models/final_attacker.pt \
  --baseline-defender models/final_defender.pt \
  --candidate-attacker models/final_marl_attacker_1v1.pt \
  --candidate-defender models/final_marl_defender_1v1_defender.pt \
  --episodes 100 --seed 1234 --output evaluation.json

The reported difference is meaningful only for the same environment/action contract;
run multiple seeds and report uncertainty before drawing a general conclusion.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch

from agents.dqn_attacker import DQNAttacker
from agents.dqn_defender import DQNDefender
from env.network_env import NetworkEnvironment


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for a Bernoulli win rate."""
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def load_policy(attacker_path: str, defender_path: str, state_size: int):
    attacker = DQNAttacker(state_size=state_size)
    defender = DQNDefender(state_size=state_size)
    attacker.load(attacker_path)
    defender.load(defender_path)
    attacker.epsilon = 0.0
    defender.epsilon = 0.0
    attacker.online_net.eval()
    defender.online_net.eval()
    return attacker, defender


def evaluate_pair(
    *, name: str, attacker_path: str, defender_path: str,
    episodes: int, seed: int,
) -> dict[str, Any]:
    probe_env = NetworkEnvironment(n_attackers=1, n_defenders=1, seed=seed, max_steps=50)
    state_size = len(probe_env.reset(seed=seed))
    attacker, defender = load_policy(attacker_path, defender_path, state_size)

    rewards_att: list[float] = []
    rewards_def: list[float] = []
    steps_list: list[int] = []
    detections_list: list[int] = []
    wins = 0
    for index in range(episodes):
        episode_seed = (seed + index) % (2**32 - 1)
        env = NetworkEnvironment(n_attackers=1, n_defenders=1, seed=episode_seed, max_steps=50)
        state = env.reset(seed=episode_seed)
        total_att, total_def = 0.0, 0.0
        done = False
        step_count = 0
        while not done and step_count < env.max_steps:
            att_action = attacker.choose_action(state)
            def_action = defender.choose_action(state)
            state, att_reward, def_reward, done = env.step([att_action], [def_action])
            total_att += float(att_reward)
            total_def += float(def_reward)
            step_count += 1
        info = env.get_info()
        rewards_att.append(total_att)
        rewards_def.append(total_def)
        steps_list.append(step_count)
        detections_list.append(int(info["detection_count"]))
        wins += int(bool(info["attacker_won"]))

    return {
        "policy": name,
        "attacker_checkpoint": str(attacker_path),
        "defender_checkpoint": str(defender_path),
        "episodes": episodes,
        "base_seed": seed,
        "episode_seeds": [((seed + i) % (2**32 - 1)) for i in range(episodes)],
        "attacker_win_rate": wins / episodes,
        "attacker_win_rate_95pct_wilson": wilson_interval(wins, episodes),
        "mean_attacker_reward": mean(rewards_att),
        "mean_defender_reward": mean(rewards_def),
        "median_episode_steps": median(steps_list),
        "mean_detections_per_episode": mean(detections_list),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-attacker", required=True)
    parser.add_argument("--baseline-defender", required=True)
    parser.add_argument("--candidate-attacker", required=True)
    parser.add_argument("--candidate-defender", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None, help="Optional JSON results path")
    args = parser.parse_args()

    if args.episodes < 1 or args.episodes > 10000:
        parser.error("--episodes must be between 1 and 10000")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    paths = [
        args.baseline_attacker, args.baseline_defender,
        args.candidate_attacker, args.candidate_defender,
    ]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        parser.error("checkpoint file(s) not found: " + ", ".join(missing))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    baseline = evaluate_pair(
        name="baseline",
        attacker_path=args.baseline_attacker,
        defender_path=args.baseline_defender,
        episodes=args.episodes,
        seed=args.seed,
    )
    candidate = evaluate_pair(
        name="candidate",
        attacker_path=args.candidate_attacker,
        defender_path=args.candidate_defender,
        episodes=args.episodes,
        seed=args.seed,
    )
    report = {
        "evaluation": "fixed-seed paired checkpoint comparison",
        "environment": "CyberArena 1v1; identical per-episode environment seeds; 50-step cap",
        "seed": args.seed,
        "episodes_per_policy": args.episodes,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": {
            "attacker_win_rate": candidate["attacker_win_rate"] - baseline["attacker_win_rate"],
            "mean_attacker_reward": candidate["mean_attacker_reward"] - baseline["mean_attacker_reward"],
            "mean_defender_reward": candidate["mean_defender_reward"] - baseline["mean_defender_reward"],
            "median_episode_steps": candidate["median_episode_steps"] - baseline["median_episode_steps"],
            "mean_detections_per_episode": candidate["mean_detections_per_episode"] - baseline["mean_detections_per_episode"],
        },
        "interpretation_note": (
            "Use as a controlled comparison on the listed seed set, not a universal performance claim. "
            "The Wilson interval describes sampling uncertainty in attacker win rate; compare multiple seed sets "
            "before making research claims."
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved evaluation report to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
