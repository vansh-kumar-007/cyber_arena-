# env/reward.py
# This file calculates rewards for both attacker and defender.
# Rewards are what teach the agents what "good" and "bad" behavior is.

def calculate_attacker_reward(event, node_value=0):
    """
    event: a string describing what just happened
    node_value: the value of the node involved (if any)
    """
    if event == "exploit_success":
        return 10 * node_value      # Big reward for hacking a node

    elif event == "critical_node_reached":
        return 50                   # Huge reward for reaching a critical node

    elif event == "detected":
        return -20                  # Punishment for getting caught

    elif event == "wasted_action":
        return -2                   # Small punishment for doing nothing useful

    else:
        return 0                    # Neutral — nothing happened


def calculate_defender_reward(event):
    """
    event: a string describing what just happened
    """
    if event == "attack_detected":
        return 15                   # Reward for spotting an attack

    elif event == "attack_blocked":
        return 20                   # Bigger reward for actually stopping it

    elif event == "false_positive":
        return -10                  # Punishment for blocking innocent traffic

    elif event == "critical_node_compromised":
        return -50                  # Huge punishment if attacker wins

    else:
        return 0                    # Neutral