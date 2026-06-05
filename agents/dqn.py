# agents/dqn.py
# Deep Q-Network — replaces the Q-table with a neural network.
#
# Instead of storing Q(s,a) in a dictionary,
# we train a neural network to PREDICT Q(s,a) for any state.
#
# Two key innovations over basic Q-Learning:
# 1. Experience Replay — store past experiences, learn from random batches
# 2. Target Network — a frozen copy of the network for stable training

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ─── NEURAL NETWORK ARCHITECTURE ─────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    This is the actual neural network.
    Input:  state vector (what the agent sees)
    Output: Q-value for each possible action
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()

        # 3-layer neural network
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),   # Input layer
            nn.ReLU(),                             # Activation
            nn.Linear(hidden_size, hidden_size),  # Hidden layer
            nn.ReLU(),                             # Activation
            nn.Linear(hidden_size, action_size)   # Output layer
        )

    def forward(self, x):
        """Pass state through network to get Q-values"""
        return self.network(x)


# ─── EXPERIENCE REPLAY BUFFER ─────────────────────────────────────────────────

class ReplayBuffer:
    """
    Stores past experiences (state, action, reward, next_state, done).
    During training we sample RANDOM batches from this buffer.

    Why random? To break correlation between consecutive experiences
    which would make training unstable.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


# ─── DQN AGENT ────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    The full DQN agent with:
    - Online network (learns every step)
    - Target network (updates every N steps for stability)
    - Experience replay buffer
    - Epsilon-greedy exploration
    """
    def __init__(self, state_size, action_size, name="DQNAgent"):
        self.state_size = state_size
        self.action_size = action_size
        self.name = name

        # Hyperparameters
        self.gamma = 0.99          # Discount factor
        self.epsilon = 1.0         # Start with full exploration
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update_freq = 10  # Update target network every N episodes

        # Two networks — online and target
        self.online_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)

        # Copy online weights to target at start
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target net is never trained directly

        # Optimizer
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=self.learning_rate
        )

        # Loss function
        self.criterion = nn.MSELoss()
        self.use_double_dqn = True   # Double DQN enabled

        # Replay buffer
        self.memory = ReplayBuffer(capacity=10000)

        # Tracking
        self.episode_reward = 0
        self.losses = []
        self.episode_count = 0

    def choose_action(self, state):
        """
        Epsilon-greedy: random action OR best network action
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Convert state to tensor and get Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.episode_reward += reward
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """
        Sample a random batch from memory and train the network.
        This is the core DQN update.
        """
        if len(self.memory) < self.batch_size:
            return None  # Not enough experiences yet

        # Sample random batch
        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.batch_size)

        # Current Q-values from online network
        current_q = self.online_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Double DQN — decouple action selection from evaluation
        # Step 1: online network selects the best action
        # Step 2: target network evaluates that action
        # This prevents overestimation of Q-values
        with torch.no_grad():
            # Online net picks best action for next state
            best_actions = self.online_net(next_states).argmax(1).unsqueeze(1)
            # Target net evaluates that specific action
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss and backpropagate
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), max_norm=1.0
        )

        self.optimizer.step()
        self.losses.append(loss.item())
        return loss.item()

    def update_target_network(self):
        """Copy online network weights to target network"""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_episode_reward(self):
        self.episode_reward = 0
        self.episode_count += 1

    def get_stats(self):
        avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
        return {
            "name": self.name,
            "epsilon": round(self.epsilon, 4),
            "memory_size": len(self.memory),
            "episode_reward": round(self.episode_reward, 2),
            "avg_loss": round(avg_loss, 4),
            "mode": "Double DQN" if self.use_double_dqn else "DQN",
        }

    def get_network_weights(self):
        """
        Returns network weights for visualization.
        We'll use this to show the neural network learning in real time.
        """
        weights = []
        for name, param in self.online_net.named_parameters():
            if 'weight' in name:
                w = param.detach().numpy()
                weights.append({
                    "layer": name,
                    "shape": list(w.shape),
                    "mean": float(np.mean(np.abs(w))),
                    "max": float(np.max(np.abs(w))),
                    "values": w.tolist()
                })
        return weights

    def save(self, path):
        """Save trained model to disk"""
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load trained model from disk"""
        checkpoint = torch.load(path)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon = checkpoint["epsilon"]
        self.episode_count = checkpoint["episode_count"]
        print(f"Model loaded from {path}")