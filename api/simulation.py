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
        """Try loading models in order of preference"""
        model_options = [
            ("models/final_marl_attacker_1v1.pt", "models/final_marl_defender_1v1_defender.pt"),
            ("models/best_attacker.pt", "models/best_defender.pt"),
            ("models/final_attacker.pt", "models/final_defender.pt"),
        ]

        for att_path, def_path in model_options:
            if os.path.exists(att_path) and os.path.exists(def_path):
                try:
                    self.attacker.load(att_path)
                    self.defender.load(def_path)
                    print(f"✅ Loaded models: {att_path}")
                    return
                except RuntimeError:
                    print(f"⚠️ Size mismatch for {att_path}, trying next...")
                    continue

        print("⚠️ No compatible models found — using fresh weights")
        print("   API will still work, agents start untrained")

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
        # Environment expects lists (one action per agent)
        n_att = self.env.n_attackers
        n_def = self.env.n_defenders

        att_actions = [self.attacker.choose_action(state) for _ in range(n_att)]
        def_actions = [self.defender.choose_action(state) for _ in range(n_def)]

        # Use first action for logging/visualization
        att_action = att_actions[0]
        def_action = def_actions[0]

        # Get Q-values for visualization
        import torch
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            att_q_values = self.attacker.online_net(state_tensor).numpy()[0].tolist()
            def_q_values = self.defender.online_net(state_tensor).numpy()[0].tolist()

        # Step environment
        next_state, att_reward, def_reward, done = self.env.step(
            att_actions, def_actions
        )

        # Learn from this step
        # Store experience but don't learn during API gameplay
        # Learning happens during training (train_dqn.py)
        # This prevents gradient computation conflicts
        self.attacker.remember(state, att_action, att_reward, next_state, float(done))
        self.defender.remember(state, def_action, def_reward, next_state, float(done))

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
                "attacker_score": info["attacker_score"],    # make sure this is here
                "defender_score": info["defender_score"],    # make sure this is here
                "battle_log": info["battle_log"],
                "last_attack": info["last_attack"],
                "last_defense": info["last_defense"],
                "attacker_positions": info["attacker_positions"],
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