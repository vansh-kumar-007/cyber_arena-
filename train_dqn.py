"""Train shared-team Double DQN policies with bounded, persistent replay.

The deployed API remains inference-only. This script is the explicit route for
gradient updates, allowing training to be reproducible and auditable.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

from agents.dqn_attacker import DQNAttacker
from agents.dqn_defender import DQNDefender
from configs.network_config import ATTACK_TYPES, DEFENSE_TYPES
from env.network_env import NetworkEnvironment
from utils.experience_memory import ExperienceMemory
from utils.metrics import Metrics

PROJECT_ROOT = Path(__file__).resolve().parent
REPLAY_CAPACITY = 10_000


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _restore_replay(memory: ExperienceMemory, agent, *, agent_name: str, context: str) -> int:
    """Reload compatible transitions into PER; priorities are reinitialized uniformly."""
    restored = 0
    for item in memory.load_transitions(agent=agent_name, context=context, limit=REPLAY_CAPACITY):
        state = item["state"]
        next_state = item["next_state"]
        if len(state) != agent.state_size or len(next_state) != agent.state_size:
            continue
        if item["action"] >= agent.action_size:
            continue
        agent.memory.push(
            tuple(state), item["action"], item["reward"], tuple(next_state),
            float(item["terminal"]),
        )
        restored += 1
    return restored


def _record_training_transition(
    memory: ExperienceMemory,
    *,
    agent_name: str,
    context: str,
    action: int,
    state: tuple[float, ...],
    reward: float,
    next_state: tuple[float, ...],
    done: bool,
    episode: int,
    step: int,
    action_name: str,
) -> None:
    """Store an audit record and the matching replay transition."""
    outcome = "success" if reward > 0 else "failure" if reward < 0 else "neutral"
    record = memory.record(
        agent=agent_name,
        task=f"{agent_name} training decision in {context}",
        state_summary={"state_vector": list(state)},
        action={"id": action, "name": action_name},
        outcome=outcome,
        reward=float(reward),
        lesson=(
            f"Observed normalized team reward {reward:.4f} after the {action_name} "
            f"action at episode {episode}, step {step}. This is an outcome observation, "
            "not proof that the action caused the reward."
        ),
        tags=[context, action_name, outcome],
        metadata={"episode": episode, "step": step, "context": context, "terminal": done},
    )
    memory.remember_transition(
        agent=agent_name,
        context=context,
        state=state,
        action=action,
        reward=float(reward),
        next_state=next_state,
        terminal=done,
        experience_id=record["id"],
        capacity=REPLAY_CAPACITY,
    )


def train_marl(
    n_attackers: int = 2,
    n_defenders: int = 2,
    num_episodes: int = 1000,
    save_models: bool = True,
    *,
    seed: int | None = None,
    memory: ExperienceMemory | None = None,
    resume: bool = False,
    restore_replay: bool = True,
):
    """Train a shared DQN per team.

    Args:
        seed: Seeds Python, NumPy, PyTorch, and environment randomness.
        memory: Optional externally-managed memory database.
        resume: Restore latest per-configuration checkpoint weights/optimizer.
        restore_replay: Reload stored transitions for this exact team-size scenario.
    """
    if not isinstance(num_episodes, int) or isinstance(num_episodes, bool) or num_episodes < 1:
        raise ValueError("num_episodes must be a positive integer")
    if seed is not None:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer or None")
        _seed_everything(seed)

    owns_memory = memory is None
    memory = memory or ExperienceMemory()
    context = f"{n_attackers}v{n_defenders}"
    env = NetworkEnvironment(
        n_attackers=n_attackers, n_defenders=n_defenders, seed=seed, max_steps=50
    )
    state_size = len(env.reset())

    print("=" * 60)
    print("   CyberArena RL — Multi-Agent DQN Training (MARL)")
    print(f"   {n_attackers} Attackers vs {n_defenders} Defenders")
    print(f"   Scenario: {context} | State size: {state_size} | Seed: {seed}")
    print("   Architecture: Centralized Training, Decentralized Execution")
    print("=" * 60)

    # One shared brain per team; every individual action contributes a transition.
    attacker_brain = DQNAttacker(state_size=state_size)
    defender_brain = DQNDefender(state_size=state_size)
    for brain in (attacker_brain, defender_brain):
        brain.gamma = 0.95
        brain.learning_rate = 0.0005
        brain.target_update_freq = 5
        for group in brain.optimizer.param_groups:
            group["lr"] = brain.learning_rate

    if restore_replay:
        att_restored = _restore_replay(memory, attacker_brain, agent_name="attacker", context=context)
        def_restored = _restore_replay(memory, defender_brain, agent_name="defender", context=context)
        if att_restored or def_restored:
            print(f"Restored replay: attacker={att_restored}, defender={def_restored}")

    model_dir = PROJECT_ROOT / "models"
    attacker_checkpoint = model_dir / f"marl_{context}_attacker.pt"
    defender_checkpoint = model_dir / f"marl_{context}_defender.pt"
    if resume and attacker_checkpoint.is_file() and defender_checkpoint.is_file():
        try:
            attacker_brain.load(str(attacker_checkpoint))
            defender_brain.load(str(defender_checkpoint))
            # Preserve the current training config even if an older optimizer state
            # stored a mismatched learning-rate value.
            for brain in (attacker_brain, defender_brain):
                brain.gamma = 0.95
                brain.learning_rate = 0.0005
                brain.target_update_freq = 5
                for group in brain.optimizer.param_groups:
                    group["lr"] = brain.learning_rate
            print(f"Resumed checkpoints for {context}")
        except (RuntimeError, KeyError, ValueError, OSError) as exc:
            print(f"Checkpoint resume failed; continuing from fresh weights: {exc}")

    metrics = Metrics()
    best_reward = float("-inf")
    print(f"Training for {num_episodes} episodes...")

    for episode in range(1, num_episodes + 1):
        episode_seed = ((seed + episode - 1) % (2**32 - 1)) if seed is not None else None
        state = env.reset(seed=episode_seed)
        attacker_brain.reset_episode_reward()
        defender_brain.reset_episode_reward()
        done = False
        step = 0
        att_losses = []
        def_losses = []

        while not done and step < env.max_steps:
            step += 1
            att_actions = [attacker_brain.choose_action(state) for _ in range(n_attackers)]
            def_actions = [defender_brain.choose_action(state) for _ in range(n_defenders)]
            next_state, att_reward, def_reward, done = env.step(att_actions, def_actions)

            att_reward_n = float(np.clip(att_reward / 30.0, -3.0, 3.0))
            def_reward_n = float(np.clip(def_reward / 30.0, -3.0, 3.0))

            for att_action in att_actions:
                attacker_brain.remember(state, att_action, att_reward_n, next_state, float(done))
                _record_training_transition(
                    memory, agent_name="attacker", context=context, action=att_action,
                    state=state, reward=att_reward_n, next_state=next_state, done=bool(done),
                    episode=episode, step=step, action_name=ATTACK_TYPES[att_action]["name"],
                )
            for def_action in def_actions:
                defender_brain.remember(state, def_action, def_reward_n, next_state, float(done))
                _record_training_transition(
                    memory, agent_name="defender", context=context, action=def_action,
                    state=state, reward=def_reward_n, next_state=next_state, done=bool(done),
                    episode=episode, step=step, action_name=DEFENSE_TYPES[def_action]["name"],
                )

            att_loss = attacker_brain.learn()
            def_loss = defender_brain.learn()
            if att_loss is not None:
                att_losses.append(att_loss)
            if def_loss is not None:
                def_losses.append(def_loss)
            state = next_state

        attacker_brain.decay_epsilon()
        defender_brain.decay_epsilon()
        if episode % attacker_brain.target_update_freq == 0:
            attacker_brain.update_target_network()
            defender_brain.update_target_network()

        info = env.get_info()
        metrics.record(
            ep_attacker_reward=attacker_brain.episode_reward,
            ep_defender_reward=defender_brain.episode_reward,
            attacker_won=info["attacker_won"],
            detections=info["detection_count"],
            steps=step,
        )

        if episode % 100 == 0:
            recent_wins = metrics.attacker_success[-100:]
            win_rate = sum(recent_wins) / len(recent_wins) * 100
            avg_att_loss = float(np.mean(att_losses)) if att_losses else 0.0
            avg_def_loss = float(np.mean(def_losses)) if def_losses else 0.0
            att_stats = attacker_brain.get_stats()
            def_stats = defender_brain.get_stats()
            print(f"Episode {episode}/{num_episodes} [{context}]")
            print(f"  Attacker reward={att_stats['episode_reward']:.2f} epsilon={att_stats['epsilon']} loss={avg_att_loss:.4f}")
            print(f"  Defender reward={def_stats['episode_reward']:.2f} epsilon={def_stats['epsilon']} loss={avg_def_loss:.4f}")
            print(f"  Attacker win rate (last 100): {win_rate:.1f}% | {att_stats['mode']}")
            if save_models and attacker_brain.episode_reward > best_reward:
                best_reward = attacker_brain.episode_reward
                model_dir.mkdir(parents=True, exist_ok=True)
                attacker_brain.save(str(attacker_checkpoint))
                defender_brain.save(str(defender_checkpoint))

    print("=" * 60)
    print("   MARL Training Complete!")
    print("=" * 60)
    if metrics.attacker_rewards:
        metrics.summary(last_n=min(100, len(metrics.attacker_rewards)))

    if save_models:
        model_dir.mkdir(parents=True, exist_ok=True)
        attacker_brain.save(str(model_dir / f"final_marl_attacker_{context}.pt"))
        defender_brain.save(str(model_dir / f"final_marl_defender_{context}_defender.pt"))

    print(f"Persistent memory summary: {memory.summary()}")
    if owns_memory:
        memory.close()
    return metrics, attacker_brain, defender_brain


if __name__ == "__main__":
    configs = [(1, 1), (2, 2), (3, 2)]
    for n_att, n_def in configs:
        train_marl(n_attackers=n_att, n_defenders=n_def, num_episodes=500)
