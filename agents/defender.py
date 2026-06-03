# agents/defender.py
from agents.q_learning import QLearningAgent

class DefenderAgent(QLearningAgent):
    def __init__(self):
        super().__init__(n_actions=12, name="Defender")