# api/simulation.py
# This file wraps our RL environment and DQN agents
# so FastAPI can expose them as API endpoints.

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.network_env import NetworkEnvironment
from agents.dqn_attacker import DQNAttacker
from agents.dqn_defender import DQNDefender
from configs.network_config import ATTACK_TYPES, DEFENSE_TYPES

class SimulationManager:
    def __init__(self):
        self.env = NetworkEnvironment()
        sample_state = self.env.reset()
        self.state_size = len(sample_state)

        # Initialize agents
        self.attacker = DQNAttacker(state_size=self.state_size)
        self.defender = DQNDefender(state_size=self.state_size)

        # Load trained models if they exist
        self._load_models()

        # Set to exploitation mode (no random actions)
        self.attacker.epsilon = 0.05
        self.defender.epsilon = 0.05

        # Current state
        self.current_state = self.env.reset()
        self.episode_count = 0
        self.is_done = False

    def _load_models(self):
        """Load trained DQN models if available — handles size mismatch gracefully"""
        att_path = "models/best_attacker.pt"
        def_path = "models/best_defender.pt"

        if os.path.exists(att_path):
            try:
                self.attacker.load(att_path)
                print(f"✅ Loaded attacker model from {att_path}")
            except RuntimeError as e:
                print(f"⚠️ Could not load attacker model (size mismatch) — using fresh weights")
                print(f"   Reason: {e}")
        else:
            print("⚠️ No trained attacker model found, using random weights")

        if os.path.exists(def_path):
            try:
                self.defender.load(def_path)
                print(f"✅ Loaded defender model from {def_path}")
            except RuntimeError as e:
                print(f"⚠️ Could not load defender model (size mismatch) — using fresh weights")
                print(f"   Reason: {e}")
        else:
            print("⚠️ No trained defender model found, using random weights")

    def reset(self):
        """Reset environment for new episode"""
        self.current_state = self.env.reset()
        self.is_done = False
        self.episode_count += 1
        return self._get_full_state()

    def step(self):
        """Run one step of the simulation"""
        if self.is_done:
            return self._get_full_state()

        state = self.current_state

        # Get actions from DQN agents
        att_action = self.attacker.choose_action(state)
        def_action = self.defender.choose_action(state)

        # Get Q-values for visualization
        import torch
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            att_q_values = self.attacker.online_net(state_tensor).numpy()[0].tolist()
            def_q_values = self.defender.online_net(state_tensor).numpy()[0].tolist()

        # Step environment
        next_state, att_reward, def_reward, done = self.env.step(
            att_action, def_action
        )

        # Learn from this step
        self.attacker.remember(state, att_action, att_reward, next_state, float(done))
        self.defender.remember(state, def_action, def_reward, next_state, float(done))
        self.attacker.learn()
        self.defender.learn()

        self.current_state = next_state
        self.is_done = done

        if done:
            self.attacker.decay_epsilon()
            self.defender.decay_epsilon()
            if self.episode_count % 10 == 0:
                self.attacker.update_target_network()
                self.defender.update_target_network()

        info = self.env.get_info()

        return {
            "state": {
                "compromised": info["compromised"],
                "attacker_position": info["attacker_position"],
                "detection_score": info["detection_score"],
                "blocked_nodes": info["blocked_nodes"],
                "patched_nodes": info["patched_nodes"],
                "honeypots": info["honeypots"],
                "isolated_nodes": info["isolated_nodes"],
                "ids_active": info["ids_active"],
                "attacker_won": info["attacker_won"],
                "attacker_score": info["attacker_score"],
                "defender_score": info["defender_score"],
                "battle_log": info["battle_log"],
                "last_attack": info["last_attack"],
                "last_defense": info["last_defense"],
            },
            "att_action": att_action,
            "def_action": def_action,
            "att_reward": float(att_reward),
            "def_reward": float(def_reward),
            "att_q_values": att_q_values,
            "def_q_values": def_q_values,
            "done": done,
            "step": self.env.current_step,
            "episode": self.episode_count,
            "att_epsilon": round(self.attacker.epsilon, 4),
            "def_epsilon": round(self.defender.epsilon, 4),
        }

    def get_network_weights(self):
        """Get neural network weights for visualization"""
        return {
            "attacker": self.attacker.get_network_weights(),
            "defender": self.defender.get_network_weights(),
        }

    def _get_full_state(self):
        info = self.env.get_info()
        return {
            "state": info,
            "done": self.is_done,
            "step": self.env.current_step,
            "episode": self.episode_count,
        }