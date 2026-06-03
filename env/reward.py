# env/reward.py

def calculate_attacker_reward(event, node_value=0, damage=0, stealth=1.0):
    if event == "exploit_success":
        return (10 * node_value) + (5 * damage)

    elif event == "critical_node_reached":
        return 50 + (10 * damage)

    elif event == "detected":
        return -25 * (1 - stealth)   # Stealthy attacks hurt less when detected

    elif event == "wasted_action":
        return -1

    elif event == "ransomware_success":
        return 80                     # Huge reward for ransomware

    elif event == "data_exfil_success":
        return 60

    elif event == "ddos_success":
        return 20

    else:
        return 0


def calculate_defender_reward(event, attack_damage=0):
    if event == "attack_detected":
        return 15 + (2 * attack_damage)

    elif event == "attack_blocked":
        return 25 + (3 * attack_damage)

    elif event == "false_positive":
        return -5

    elif event == "critical_node_compromised":
        return -50 - (5 * attack_damage)

    elif event == "honeypot_triggered":
        return 30                     # Big reward for catching attacker in honeypot

    elif event == "patch_success":
        return 10

    else:
        return 0