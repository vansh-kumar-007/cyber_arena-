"""Simulation orchestration for the FastAPI service.

API gameplay is inference-only by default. Outcomes are persisted as auditable
episodic memories; DQN policy updates remain part of the explicit training
pipeline so public gameplay traffic cannot silently rewrite deployed weights.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from agents.dqn_attacker import DQNAttacker
from agents.dqn_defender import DQNDefender
from configs.network_config import ATTACK_TYPES, DEFENSE_TYPES
from env.network_env import NetworkEnvironment
from utils.experience_memory import ExperienceMemory

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SimulationManager:
    def __init__(
        self,
        *,
        memory: ExperienceMemory | None = None,
        model_dir: str | Path | None = None,
        seed: int | None = None,
        n_attackers: int = 1,
        n_defenders: int = 1,
    ) -> None:
        self.env = NetworkEnvironment(
            n_attackers=n_attackers, n_defenders=n_defenders, seed=seed, max_steps=50
        )
        sample_state = self.env.reset()
        self.state_size = len(sample_state)
        self.attacker = DQNAttacker(state_size=self.state_size)
        self.defender = DQNDefender(state_size=self.state_size)
        self.memory = memory or ExperienceMemory()
        self.model_dir = Path(model_dir) if model_dir else PROJECT_ROOT / "models"
        self.models_loaded = self._load_models()
        # Inference-only policy; an explicit training run controls exploration.
        self.attacker.epsilon = 0.0
        self.defender.epsilon = 0.0
        self.current_state = self.env.reset()
        self.episode_count = 0
        self.is_done = False
        self.memory_write_errors = 0

    def _load_models(self) -> bool:
        """Load the first complete and architecture-compatible checkpoint pair."""
        context = f"{self.env.n_attackers}v{self.env.n_defenders}"
        active_pointer = self.model_dir / "registry" / "active.json"
        if active_pointer.exists():
            # Once registry-managed promotion is enabled, never silently fall back
            # to legacy filenames if the active pointer/checkpoint is invalid.
            from utils.model_registry import ModelRegistry

            registry = ModelRegistry(self.model_dir)
            try:
                active = registry.active_model()
                if active is None:
                    raise RuntimeError("active model pointer disappeared")
                if active.get("scenario_context") != context:
                    logger.warning("Active model %s is for %s, not %s",
                                   active.get("model_id"), active.get("scenario_context"), context)
                    return False
                candidate_attacker = DQNAttacker(state_size=self.state_size)
                candidate_defender = DQNDefender(state_size=self.state_size)
                candidate_attacker.load(str(self.model_dir / active["attacker_path"]))
                candidate_defender.load(str(self.model_dir / active["defender_path"]))
                self.attacker = candidate_attacker
                self.defender = candidate_defender
                logger.info("Loaded registry active model %s", active["model_id"])
                return True
            except Exception:
                logger.exception("Active model registry is invalid; refusing legacy/random fallback")
                return False

        model_options = [
            (f"final_marl_attacker_{context}.pt", f"final_marl_defender_{context}_defender.pt"),
            (f"marl_{context}_attacker.pt", f"marl_{context}_defender.pt"),
        ]
        # Legacy checkpoint names are not scenario-labelled, so only load them
        # for the original 1v1 configuration instead of silently reusing them
        # for a different multi-agent configuration.
        if context == "1v1":
            model_options.extend([
                ("best_attacker.pt", "best_defender.pt"),
                ("final_attacker.pt", "final_defender.pt"),
            ])
        for attacker_name, defender_name in model_options:
            attacker_path = self.model_dir / attacker_name
            defender_path = self.model_dir / defender_name
            if not attacker_path.is_file() or not defender_path.is_file():
                continue
            try:
                # Load into temporary agents so a half-loaded pair can never leave
                # the running simulation with one trained and one random network.
                candidate_attacker = DQNAttacker(state_size=self.state_size)
                candidate_defender = DQNDefender(state_size=self.state_size)
                candidate_attacker.load(str(attacker_path))
                candidate_defender.load(str(defender_path))
                self.attacker = candidate_attacker
                self.defender = candidate_defender
                logger.info("Loaded compatible model pair: %s / %s", attacker_path.name, defender_path.name)
                return True
            except Exception as exc:
                logger.warning("Skipping incompatible checkpoint pair %s/%s: %s",
                               attacker_path.name, defender_path.name, exc)
        logger.warning("No compatible trained model pair found; API uses untrained weights")
        return False

    def reset(
        self,
        *,
        seed: int | None = None,
        n_attackers: int | None = None,
        n_defenders: int | None = None,
    ) -> dict[str, Any]:
        """Start a new episode, optionally switching to a trained team-size scenario."""
        target_attackers = self.env.n_attackers if n_attackers is None else n_attackers
        target_defenders = self.env.n_defenders if n_defenders is None else n_defenders
        if (target_attackers, target_defenders) != (self.env.n_attackers, self.env.n_defenders):
            self.env = NetworkEnvironment(
                n_attackers=target_attackers,
                n_defenders=target_defenders,
                seed=seed,
                max_steps=50,
            )
            self.state_size = len(self.env.reset(seed=seed))
            self.attacker = DQNAttacker(state_size=self.state_size)
            self.defender = DQNDefender(state_size=self.state_size)
            self.models_loaded = self._load_models()
            self.attacker.epsilon = 0.0
            self.defender.epsilon = 0.0
        self.current_state = self.env.reset(seed=seed)
        self.is_done = False
        self.episode_count += 1
        return self._get_full_state()

    @staticmethod
    def _outcome(reward: float) -> str:
        if reward > 0:
            return "success"
        if reward < 0:
            return "failure"
        return "neutral"

    def _record_experience(
        self, *, agent: str, task: str, state: tuple[float, ...],
        actions: list[dict[str, Any]], reward: float, done: bool, step: int,
    ) -> None:
        action_names = [a["name"] for a in actions]
        outcome = self._outcome(float(reward))
        lesson = (
            f"Observed {agent} action(s) {', '.join(action_names)} with "
            f"team reward {float(reward):.3f} at step {step}. "
            "This is an observed outcome, not proof that the action caused it."
        )
        try:
            self.memory.record(
                agent=agent,
                task=task,
                state_summary={"state_vector": list(state), "step": step},
                action={"actions": actions},
                outcome=outcome,
                reward=float(reward),
                lesson=lesson,
                tags=[agent, *action_names, outcome],
                metadata={"episode": self.episode_count, "step": step, "terminal": done},
            )
        except Exception:
            self.memory_write_errors += 1
            logger.exception("Could not persist %s experience at episode=%s step=%s",
                             agent, self.episode_count, step)

    def step(self) -> dict[str, Any]:
        """Run one inference step and append an auditable result to episodic memory."""
        if self.is_done:
            return self._get_full_state()

        state = self.current_state
        step_number = self.env.current_step + 1
        # Retrieve evidence for the explanation panel. These records are displayed
        # as precedents; they do not alter the neural policy's action selection.
        attacker_actions = [self.attacker.choose_action(state) for _ in range(self.env.n_attackers)]
        defender_actions = [self.defender.choose_action(state) for _ in range(self.env.n_defenders)]
        att_action = attacker_actions[0]
        def_action = defender_actions[0]

        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():
            att_q_values = self.attacker.online_net(state_tensor).cpu().numpy()[0].tolist()
            def_q_values = self.defender.online_net(state_tensor).cpu().numpy()[0].tolist()

        next_state, att_reward, def_reward, done = self.env.step(attacker_actions, defender_actions)
        info = self.env.get_info()
        att_actions_named = [
            {"id": action, "name": ATTACK_TYPES[action]["name"]} for action in attacker_actions
        ]
        def_actions_named = [
            {"id": action, "name": DEFENSE_TYPES[action]["name"]} for action in defender_actions
        ]
        precedent_query = " ".join(
            ["attacker", *(action["name"] for action in att_actions_named),
             "defender", *(action["name"] for action in def_actions_named)]
        )
        precedents = self.memory.retrieve(precedent_query, limit=3)

        def top_actions(values: list[float], action_map: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
            ranked = sorted(enumerate(values), key=lambda pair: pair[1], reverse=True)[:3]
            return [
                {"id": action_id, "name": action_map[action_id]["name"], "q_value": float(value)}
                for action_id, value in ranked
            ]
        self._record_experience(
            agent="attacker", task="simulated network attack decision", state=state,
            actions=att_actions_named, reward=float(att_reward), done=done, step=step_number,
        )
        self._record_experience(
            agent="defender", task="simulated network defense decision", state=state,
            actions=def_actions_named, reward=float(def_reward), done=done, step=step_number,
        )

        self.current_state = next_state
        self.is_done = done
        return {
            "state": self._serialize_state(info),
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
            "decision_context": {
                "explanation": "The DQN selected actions using its current Q-value estimates for the observed state. Q-values estimate relative long-term return; they are not probabilities or guarantees. Stored episodes are historical context only and do not directly override the deployed policy.",
                "attacker_top_actions": top_actions(att_q_values, ATTACK_TYPES),
                "defender_top_actions": top_actions(def_q_values, DEFENSE_TYPES),
                "precedents": precedents,
            },
        }

    @staticmethod
    def _serialize_state(info: dict[str, Any]) -> dict[str, Any]:
        return {
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
            "attacker_positions": info["attacker_positions"],
            "defender_positions": info["defender_positions"],
            "detection_count": info["detection_count"],
            "n_attackers": info["n_attackers"],
            "n_defenders": info["n_defenders"],
        }

    def get_network_weights(self) -> dict[str, Any]:
        return {
            "attacker": self.attacker.get_network_weights(),
            "defender": self.defender.get_network_weights(),
        }

    def status(self) -> dict[str, Any]:
        active_model_id = None
        model_registry_error = False
        active_pointer = self.model_dir / "registry" / "active.json"
        if active_pointer.exists():
            try:
                from utils.model_registry import ModelRegistry
                active = ModelRegistry(self.model_dir).active_model()
                active_model_id = active["model_id"] if active else None
            except Exception:
                # Do not leak checkpoint paths or exception details through the API.
                model_registry_error = True
                logger.exception("Could not read active model registry status")
        return {
            "active_model_id": active_model_id,
            "model_registry_error": model_registry_error,
            "episode": self.episode_count,
            "step": self.env.current_step,
            "is_done": self.is_done,
            "att_epsilon": round(self.attacker.epsilon, 4),
            "def_epsilon": round(self.defender.epsilon, 4),
            "att_memory": len(self.attacker.memory),
            "def_memory": len(self.defender.memory),
            "models_loaded": self.models_loaded,
            "state_size": self.state_size,
            "n_attackers": self.env.n_attackers,
            "n_defenders": self.env.n_defenders,
            "experience_memory": self.memory.summary(),
            "memory_write_errors": self.memory_write_errors,
        }

    def _get_full_state(self) -> dict[str, Any]:
        return {
            "state": self._serialize_state(self.env.get_info()),
            "done": self.is_done,
            "step": self.env.current_step,
            "episode": self.episode_count,
            "models_loaded": self.models_loaded,
            "state_size": self.state_size,
            "n_attackers": self.env.n_attackers,
            "n_defenders": self.env.n_defenders,
        }
