# agents/dqn_defender.py
from agents.dqn import DQNAgent

class DQNDefender(DQNAgent):
    def __init__(self, state_size):
        super().__init__(
            state_size=state_size,
            action_size=12,
            name="DQN Defender"
        )