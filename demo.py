# demo.py - diagnostic test
from env.network_env import NetworkEnvironment
import random

env = NetworkEnvironment()
state = env.reset()
print('Running 20 random steps...')

for i in range(20):
    att_action = random.randint(0, 4)
    def_action = random.randint(0, 4)
    next_state, att_r, def_r, done = env.step(att_action, def_action)
    info = env.get_info()
    print(f"Step {i+1} | AttAction:{att_action} DefAction:{def_action} | AttReward:{att_r} | Compromised:{info['compromised']} | Done:{done}")
    if done:
        break