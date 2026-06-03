# env/reward.py
# Reward shaping for DQN — rewards need to guide learning step by step

def calculate_attacker_reward(event, node_value=0, damage=0, stealth=1.0):
    if event == "exploit_success":
        return (8 * node_value) + (3 * damage)   # Immediate progress reward

    elif event == "critical_node_reached":
        return 40 + (8 * damage)

    elif event == "detected":
        return -10 * (1 - stealth)               # Stealth attacks hurt less

    elif event == "wasted_action":
        return -0.5                               # Very small penalty — don't discourage exploration

    elif event == "ransomware_success":
        return 60

    elif event == "data_exfil_success":
        return 40

    elif event == "ddos_success":
        return 15

    elif event == "move_to_new_node":
        return 1.0                                # Small reward for exploring

    else:
        return 0


def calculate_defender_reward(event, attack_damage=0):
    if event == "attack_detected":
        return 8 + attack_damage

    elif event == "attack_blocked":
        return 12 + (2 * attack_damage)

    elif event == "false_positive":
        return -3

    elif event == "critical_node_compromised":
        return -30 - (3 * attack_damage)

    elif event == "honeypot_triggered":
        return 20

    elif event == "patch_success":
        return 5

    else:
        return 0