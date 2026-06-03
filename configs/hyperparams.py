# configs/hyperparams.py
# These are the "settings" that control how our AI agents learn.

# Learning rate — how fast the agent updates what it knows
ALPHA = 0.1

# Discount factor — how much the agent cares about future rewards
GAMMA = 0.9

# Exploration rate — how often agent tries random actions (starts high, decays over time)
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

# Training settings
NUM_EPISODES = 1000   # How many rounds of training
MAX_STEPS = 50        # Max steps per episode before it ends