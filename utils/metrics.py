# utils/metrics.py
# This file tracks performance numbers over time.
# These numbers will later become the charts in our dashboard.

class Metrics:
    def __init__(self):
        self.attacker_rewards = []      # Total attacker reward per episode
        self.defender_rewards = []      # Total defender reward per episode
        self.attacker_success = []      # Did attacker reach a critical node? (1 or 0)
        self.defender_detections = []   # How many times defender detected attack
        self.episode_lengths = []       # How many steps each episode lasted

    def record(self, ep_attacker_reward, ep_defender_reward,
               attacker_won, detections, steps):
        self.attacker_rewards.append(ep_attacker_reward)
        self.defender_rewards.append(ep_defender_reward)
        self.attacker_success.append(1 if attacker_won else 0)
        self.defender_detections.append(detections)
        self.episode_lengths.append(steps)

    def summary(self, last_n=100):
        """Print a summary of the last N episodes"""
        recent_att = self.attacker_rewards[-last_n:]
        recent_def = self.defender_rewards[-last_n:]
        recent_win = self.attacker_success[-last_n:]

        print(f"\n=== Last {last_n} Episodes Summary ===")
        print(f"Avg Attacker Reward : {sum(recent_att)/len(recent_att):.2f}")
        print(f"Avg Defender Reward : {sum(recent_def)/len(recent_def):.2f}")
        print(f"Attacker Win Rate   : {sum(recent_win)/len(recent_win)*100:.1f}%")
        print("="*38)