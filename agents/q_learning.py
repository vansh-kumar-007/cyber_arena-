# agents/q_learning.py
# This is the BRAIN of both agents.
# Q-Learning works like this:
# - Agent tries actions and gets rewards
# - It remembers which actions gave good rewards in which situations
# - Over time it gets smarter by updating a "Q-table"
# The Q-table is basically: "In situation X, action Y gave me Z reward"

import random
import numpy as np
from configs.hyperparams import (
    ALPHA, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY
)

class QLearningAgent:
    def __init__(self, n_actions, name="Agent"):
        self.n_actions = n_actions      # How many actions this agent can take
        self.name = name                # "Attacker" or "Defender"
        self.q_table = {}               # The memory: state → action values
        self.epsilon = EPSILON_START    # Exploration rate (starts high)
        self.alpha = ALPHA              # Learning rate
        self.gamma = GAMMA              # Discount factor

        # Track total reward this episode
        self.episode_reward = 0

    def get_q_values(self, state):
        """
        Get Q-values for a state.
        If we've never seen this state before, initialize with zeros.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def choose_action(self, state):
        """
        Epsilon-greedy policy:
        - With probability epsilon → pick a RANDOM action (explore)
        - Otherwise → pick the BEST known action (exploit)

        Early in training epsilon is high so agent explores a lot.
        Over time epsilon decays so agent uses what it learned.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Random action
        else:
            q_values = self.get_q_values(state)
            return int(np.argmax(q_values))               # Best known action

    def update(self, state, action, reward, next_state):
        """
        The core Q-Learning update rule:
        Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]

        In plain English:
        - Look at what reward we actually got
        - Add an estimate of future rewards (gamma * max future Q)
        - Move our current estimate a little bit toward that target
        """
        self.episode_reward += reward

        current_q = self.get_q_values(state)[action]
        future_q = np.max(self.get_q_values(next_state))

        # The target is what we WISH we had predicted
        target = reward + self.gamma * future_q

        # Update: move current estimate toward target
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        """
        Call this at the end of each episode.
        Slowly reduces exploration so agent relies more on what it learned.
        """
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def reset_episode_reward(self):
        """Reset the episode reward counter at the start of each episode"""
        self.episode_reward = 0

    def get_stats(self):
        """Return useful info about this agent"""
        return {
            "name": self.name,
            "epsilon": round(self.epsilon, 4),
            "q_table_size": len(self.q_table),
            "episode_reward": round(self.episode_reward, 2)
        }