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
    Prioritized Experience Replay Buffer.
    Experiences with higher TD error are sampled more frequently.
    This means the agent learns faster from its biggest mistakes.
    """
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha      # Priority exponent (0=uniform, 1=full priority)
        self.beta = beta        # Importance sampling correction
        self.beta_increment = 0.001
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        # New experiences get max priority so they're sampled at least once
        max_priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample based on priority probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Importance sampling weights to correct for bias
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
            indices,
            torch.FloatTensor(weights)
        )

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors after learning"""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6  # Small epsilon prevents 0 priority

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
        self.memory = ReplayBuffer(capacity=10000, alpha=0.6, beta=0.4)

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
        Double DQN + Prioritized Experience Replay learning step.
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample with priorities
        result = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, indices, weights = result

        # Current Q-values
        current_q = self.online_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            best_actions = self.online_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # TD errors for priority update
        td_errors = (target_q - current_q).detach().numpy()
        self.memory.update_priorities(indices, td_errors)

        # Weighted loss — important experiences contribute more
        loss = (weights * (current_q - target_q) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
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
            "mode": "Double DQN + PER",
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