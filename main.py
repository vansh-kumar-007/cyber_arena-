# main.py
# This is the CONDUCTOR of the entire simulation.
# It connects the environment, attacker, and defender together
# and runs thousands of training episodes.

from env.network_env import NetworkEnvironment
from agents.attacker import AttackerAgent
from agents.defender import DefenderAgent
from utils.logger import Logger
from utils.metrics import Metrics
from configs.hyperparams import NUM_EPISODES, MAX_STEPS

def train():
    print("="*50)
    print("   CyberArena RL — Training Started")
    print("="*50)

    # Initialize everything
    env = NetworkEnvironment()
    attacker = AttackerAgent()
    defender = DefenderAgent()
    logger = Logger(print_logs=False)  # Set True to see every step
    metrics = Metrics()

    # ─── MAIN TRAINING LOOP ───────────────────────────────────────────────
    for episode in range(1, NUM_EPISODES + 1):

        state = env.reset()
        attacker.reset_episode_reward()
        defender.reset_episode_reward()
        done = False
        step = 0

        while not done and step < MAX_STEPS:
            step += 1

            # Both agents choose their actions based on current state
            attacker_action = attacker.choose_action(state)
            defender_action = defender.choose_action(state)

            # Environment processes both actions and returns results
            next_state, att_reward, def_reward, done = env.step(
                attacker_action, defender_action
            )

            # Both agents learn from what just happened
            attacker.update(state, attacker_action, att_reward, next_state)
            defender.update(state, defender_action, def_reward, next_state)

            # Log important events
            info = env.get_info()
            if att_reward > 0:
                logger.log(episode, step,
                    f"Attacker gained {att_reward:.1f} reward | "
                    f"Position: {info['attacker_position']}"
                )
            if def_reward > 0:
                logger.log(episode, step,
                    f"Defender gained {def_reward:.1f} reward | "
                    f"Detection: {info['detection_score']}"
                )

            state = next_state

        # ─── END OF EPISODE ───────────────────────────────────────────────

        # Decay epsilon — agents explore less as they learn more
        attacker.decay_epsilon()
        defender.decay_epsilon()

        # Record metrics for this episode
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
            print(f"\nEpisode {episode}/{NUM_EPISODES}")
            print(f"  Attacker → Reward: {att_stats['episode_reward']:>8.1f} | "
                  f"Epsilon: {att_stats['epsilon']} | "
                  f"Q-states known: {att_stats['q_table_size']}")
            print(f"  Defender → Reward: {def_stats['episode_reward']:>8.1f} | "
                  f"Epsilon: {def_stats['epsilon']} | "
                  f"Q-states known: {def_stats['q_table_size']}")

    # ─── TRAINING COMPLETE ────────────────────────────────────────────────
    print("\n" + "="*50)
    print("   Training Complete!")
    print("="*50)
    metrics.summary(last_n=100)
    logger.save("battle_log.txt")
    print("\nBattle log saved to battle_log.txt")

    return metrics, attacker, defender

if __name__ == "__main__":
    metrics, attacker, defender = train()