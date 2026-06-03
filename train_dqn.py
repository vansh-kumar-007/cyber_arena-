# train_dqn.py
# Training loop for DQN agents.
# Run this instead of main.py to use neural networks.

import os
import numpy as np
from env.network_env import NetworkEnvironment
from agents.dqn_attacker import DQNAttacker
from agents.dqn_defender import DQNDefender
from utils.metrics import Metrics

def train_dqn(num_episodes=2000, save_models=True):
    print("=" * 55)
    print("   CyberArena RL — DQN Training")
    print("   Neural Network Mode (PyTorch)")
    print("=" * 55)

    env = NetworkEnvironment()
    sample_state = env.reset()
    state_size = len(sample_state)
    print(f"\nState size: {state_size}")
    print(f"Action size: 12 (each agent)")

    attacker = DQNAttacker(state_size=state_size)
    defender = DQNDefender(state_size=state_size)

    # ── KEY FIX: tune agent hyperparameters ──
    attacker.gamma = 0.95
    attacker.learning_rate = 0.0005
    attacker.optimizer = __import__('torch').optim.Adam(
        attacker.online_net.parameters(), lr=attacker.learning_rate)
    attacker.target_update_freq = 5

    defender.gamma = 0.95
    defender.learning_rate = 0.0005
    defender.optimizer = __import__('torch').optim.Adam(
        defender.online_net.parameters(), lr=defender.learning_rate)
    defender.target_update_freq = 5

    metrics = Metrics()
    print(f"\nTraining for {num_episodes} episodes...\n")
    best_attacker_reward = float('-inf')

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        attacker.reset_episode_reward()
        defender.reset_episode_reward()
        done = False
        step = 0
        att_loss_ep = []
        def_loss_ep = []

        while not done and step < 50:
            step += 1

            # Choose actions
            att_action = attacker.choose_action(state)
            def_action = defender.choose_action(state)

            # Step environment
            next_state, att_reward, def_reward, done = env.step(
                att_action, def_action
            )
            # Normalize rewards to similar scale — critical for DQN stability
            att_reward = np.clip(att_reward / 50.0, -2.0, 2.0)
            def_reward = np.clip(def_reward / 50.0, -2.0, 2.0)

            # Store experiences
            attacker.remember(state, att_action, att_reward, next_state,
                            float(done))
            defender.remember(state, def_action, def_reward, next_state,
                            float(done))

            # Learn from experiences
            att_loss = attacker.learn()
            def_loss = defender.learn()

            if att_loss: att_loss_ep.append(att_loss)
            if def_loss: def_loss_ep.append(def_loss)

            state = next_state

        # End of episode
        attacker.decay_epsilon()
        defender.decay_epsilon()

        # Update target networks periodically
        if episode % attacker.target_update_freq == 0:
            attacker.update_target_network()
            defender.update_target_network()

        # Record metrics
        info = env.get_info()
        metrics.record(
            ep_attacker_reward=attacker.episode_reward,
            ep_defender_reward=defender.episode_reward,
            attacker_won=info["attacker_won"],
            detections=info["detection_count"],
            steps=step
        )

        # Print progress every 100 episodes
        if episode % 100 == 0:
            att_stats = attacker.get_stats()
            def_stats = defender.get_stats()
            recent_wins = metrics.attacker_success[-100:]
            win_rate = sum(recent_wins) / len(recent_wins) * 100
            avg_att_loss = np.mean(att_loss_ep) if att_loss_ep else 0
            avg_def_loss = np.mean(def_loss_ep) if def_loss_ep else 0

            print(f"Episode {episode}/{num_episodes}")
            print(f"  Attacker → Reward: {att_stats['episode_reward']:>8.1f} | "
                  f"Epsilon: {att_stats['epsilon']} | "
                  f"Loss: {avg_att_loss:.4f} | "
                  f"Memory: {att_stats['memory_size']}")
            print(f"  Defender → Reward: {def_stats['episode_reward']:>8.1f} | "
                  f"Epsilon: {def_stats['epsilon']} | "
                  f"Loss: {avg_def_loss:.4f} | "
                  f"Memory: {def_stats['memory_size']}")
            print(f"  Win Rate (last 100): {win_rate:.1f}%\n")

            # Save best model
            if save_models and attacker.episode_reward > best_attacker_reward:
                best_attacker_reward = attacker.episode_reward
                os.makedirs("models", exist_ok=True)
                attacker.save("models/best_attacker.pt")
                defender.save("models/best_defender.pt")

    # Training complete
    print("=" * 55)
    print("   DQN Training Complete!")
    print("=" * 55)
    metrics.summary(last_n=100)

    # Save final models
    if save_models:
        os.makedirs("models", exist_ok=True)
        attacker.save("models/final_attacker.pt")
        defender.save("models/final_defender.pt")
        print("\nFinal models saved to models/")

    return metrics, attacker, defender

if __name__ == "__main__":
    metrics, attacker, defender = train_dqn(num_episodes=1000)