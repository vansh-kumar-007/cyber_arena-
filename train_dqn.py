# train_dqn.py
# Multi-Agent DQN Training
# Architecture: Centralized Training, Decentralized Execution (CTDE)
# Teams share one brain — each agent acts independently using shared weights

import os
import numpy as np
from env.network_env import NetworkEnvironment
from agents.dqn_attacker import DQNAttacker
from agents.dqn_defender import DQNDefender
from utils.metrics import Metrics

def train_marl(n_attackers=2, n_defenders=2, num_episodes=1000, save_models=True):
    print("=" * 60)
    print("   CyberArena RL — Multi-Agent DQN Training (MARL)")
    print(f"   {n_attackers} Attackers vs {n_defenders} Defenders")
    print("   Architecture: Centralized Training, Decentralized Execution")
    print("=" * 60)

    env = NetworkEnvironment(n_attackers=n_attackers, n_defenders=n_defenders)
    sample_state = env.reset()
    state_size = len(sample_state)

    print(f"\nState size: {state_size}")
    print(f"Attackers: {n_attackers} | Defenders: {n_defenders}")

    # One shared brain per team (CTDE)
    attacker_brain = DQNAttacker(state_size=state_size)
    defender_brain = DQNDefender(state_size=state_size)

    attacker_brain.gamma = 0.95
    attacker_brain.learning_rate = 0.0005
    attacker_brain.target_update_freq = 5

    defender_brain.gamma = 0.95
    defender_brain.learning_rate = 0.0005
    defender_brain.target_update_freq = 5

    metrics = Metrics()
    best_reward = float('-inf')

    print(f"\nTraining for {num_episodes} episodes...\n")

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        attacker_brain.reset_episode_reward()
        defender_brain.reset_episode_reward()
        done = False
        step = 0
        att_losses = []
        def_losses = []

        while not done and step < 50:
            step += 1

            # Each agent on the team uses shared brain
            # but acts on different nodes independently
            att_actions = [
                attacker_brain.choose_action(state)
                for _ in range(n_attackers)
            ]
            def_actions = [
                defender_brain.choose_action(state)
                for _ in range(n_defenders)
            ]

            next_state, att_reward, def_reward, done = env.step(
                att_actions, def_actions
            )

            # Normalize rewards
            att_reward_n = np.clip(att_reward / 30.0, -3.0, 3.0)
            def_reward_n = np.clip(def_reward / 30.0, -3.0, 3.0)

            # Shared brain learns from all agents' experiences
            for att_action in att_actions:
                attacker_brain.remember(state, att_action,
                    att_reward_n, next_state, float(done))
            for def_action in def_actions:
                defender_brain.remember(state, def_action,
                    def_reward_n, next_state, float(done))

            att_loss = attacker_brain.learn()
            def_loss = defender_brain.learn()

            if att_loss: att_losses.append(att_loss)
            if def_loss: def_losses.append(def_loss)

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
            steps=step
        )

        if episode % 100 == 0:
            att_stats = attacker_brain.get_stats()
            def_stats = defender_brain.get_stats()
            recent_wins = metrics.attacker_success[-100:]
            win_rate = sum(recent_wins) / len(recent_wins) * 100
            avg_att_loss = np.mean(att_losses) if att_losses else 0
            avg_def_loss = np.mean(def_losses) if def_losses else 0

            print(f"Episode {episode}/{num_episodes} "
                  f"[{n_attackers}v{n_defenders}]")
            print(f"  Attacker → Reward: {att_stats['episode_reward']:>7.2f} | "
                  f"Epsilon: {att_stats['epsilon']} | "
                  f"Loss: {avg_att_loss:.4f}")
            print(f"  Defender → Reward: {def_stats['episode_reward']:>7.2f} | "
                  f"Epsilon: {def_stats['epsilon']} | "
                  f"Loss: {avg_def_loss:.4f}")
            print(f"  Win Rate (last 100): {win_rate:.1f}% | "
                  f"Mode: {att_stats['mode']}\n")

            if save_models and attacker_brain.episode_reward > best_reward:
                best_reward = attacker_brain.episode_reward
                os.makedirs("models", exist_ok=True)
                attacker_brain.save(f"models/marl_{n_attackers}v{n_defenders}_attacker.pt")
                defender_brain.save(f"models/marl_{n_attackers}v{n_defenders}_defender.pt")

    print("=" * 60)
    print("   MARL Training Complete!")
    print("=" * 60)
    metrics.summary(last_n=100)

    if save_models:
        os.makedirs("models", exist_ok=True)
        attacker_brain.save(f"models/final_marl_attacker_{n_attackers}v{n_defenders}.pt")
        defender_brain.save(f"models/final_marl_defender_{n_attackers}v{n_defenders}_defender.pt")

    return metrics, attacker_brain, defender_brain

if __name__ == "__main__":
    # Train different configurations
    configs = [(1,1), (2,2), (3,2)]

    for n_att, n_def in configs:
        print(f"\n{'='*60}")
        print(f"Starting {n_att}v{n_def} configuration...")
        print(f"{'='*60}")
        train_marl(
            n_attackers=n_att,
            n_defenders=n_def,
            num_episodes=500
        )