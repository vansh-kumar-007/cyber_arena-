# agents/attacker.py
# The attacker agent — uses QLearningAgent as its brain.
# 5 possible actions:
#   0 = Scan network
#   1 = Exploit current node
#   2 = Move laterally
#   3 = Privilege escalation
#   4 = Stay idle

from agents.q_learning import QLearningAgent
from configs.network_config import ATTACK_TYPES


class AttackerAgent(QLearningAgent):
    def __init__(self):
        super().__init__(
            n_actions=len(ATTACK_TYPES),
            name="Attacker"
        )