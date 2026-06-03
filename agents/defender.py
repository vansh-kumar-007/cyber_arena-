# agents/defender.py
# The defender agent — uses QLearningAgent as its brain.
# 5 possible actions:
#   0 = Monitor traffic
#   1 = Block current attacker node
#   2 = Patch a vulnerability
#   3 = Increase detection sensitivity
#   4 = Do nothing

from agents.q_learning import QLearningAgent

class DefenderAgent(QLearningAgent):
    def __init__(self):
        super().__init__(n_actions=5, name="Defender")