# env/network_env.py
import random
import copy
from configs.network_config import (
    NODES, NETWORK_TOPOLOGY, ATTACKER_START_NODE,
    ATTACK_TYPES, DEFENSE_TYPES
)
from env.reward import calculate_attacker_reward, calculate_defender_reward

class NetworkEnvironment:
    def __init__(self):
        self.topology = NETWORK_TOPOLOGY
        self.n_attacker_actions = len(ATTACK_TYPES)
        self.n_defender_actions = len(DEFENSE_TYPES)
        self.reset()

    def reset(self):
        self.nodes = copy.deepcopy(NODES)
        self.compromised = {node: False for node in self.nodes}
        self.attacker_position = ATTACKER_START_NODE
        self.detection_score = 0.1
        self.blocked_nodes = set()
        self.patched_nodes = set()
        self.honeypots = set()
        self.isolated_nodes = set()
        self.ids_active = False
        self.current_step = 0
        self.attacker_detected = False
        self.attacker_won = False
        self.detection_count = 0
        self.attacker_score = 0
        self.defender_score = 0
        self.last_attack = None
        self.last_defense = None
        self.battle_log = []
        return self._get_state()

    def _get_state(self):
        state = []
        node_list = list(self.nodes.keys())

        for node in node_list:
            state.append(1 if self.compromised[node] else 0)
            state.append(self.nodes[node]["vulnerability"])
            state.append(1 if node in self.blocked_nodes else 0)
            state.append(1 if node in self.honeypots else 0)

        state.append(node_list.index(self.attacker_position) / len(node_list))
        state.append(self.detection_score)
        state.append(self.current_step / 50)
        state.append(1 if self.ids_active else 0)
        state.append(len(self.compromised) / len(node_list))

        return tuple(round(x, 2) for x in state)

    def step(self, attacker_action, defender_action):
        self.current_step += 1
        attacker_reward = 0
        defender_reward = 0
        done = False

        attack = ATTACK_TYPES[attacker_action]
        defense = DEFENSE_TYPES[defender_action]

        self.last_attack = attack
        self.last_defense = defense

        current_node = self.attacker_position

        # ─── ATTACKER ACTS ────────────────────────────────────────────────────

        # Check honeypot trap
        if current_node in self.honeypots:
            self.attacker_detected = True
            self.detection_count += 1
            attacker_reward += calculate_attacker_reward("detected",
                stealth=attack["stealth"])
            defender_reward += calculate_defender_reward("honeypot_triggered",
                attack_damage=attack["damage"])
            self.log(f"🍯 Honeypot triggered! Attacker caught at {current_node}")

        elif attacker_action == 11:
            # Idle — stealth recovery
            attacker_reward += 0

        elif attacker_action == 6:
            # Lateral movement
            neighbors = self.topology.get(current_node, [])
            reachable = [n for n in neighbors
                        if n not in self.blocked_nodes
                        and n not in self.isolated_nodes]
            if reachable:
                new_node = random.choice(reachable)
                self.attacker_position = new_node
                # Reward for moving toward uncompromised nodes
                if not self.compromised[new_node]:
                    attacker_reward += calculate_attacker_reward("move_to_new_node")
                self.log(f"🔀 Attacker moved to {self.attacker_position}")
            else:
                attacker_reward += calculate_attacker_reward("wasted_action")

        else:
            # All attack types
            if current_node in self.blocked_nodes or current_node in self.isolated_nodes:
                attacker_reward += calculate_attacker_reward("wasted_action")
                self.log(f"🚫 Attack blocked — node {current_node} is secured")
            else:
                vuln = self.nodes[current_node]["vulnerability"]
                defense_strength = self.detection_score * 0.15
                ids_bonus = 0.1 if self.ids_active else 0
                success_prob = max(0.1, attack["base_success"] * vuln - defense_strength - ids_bonus)

                if random.random() < success_prob:
                    # Attack succeeded
                    if attacker_action == 2:
                        # DDoS — disrupts but doesn't compromise
                        attacker_reward += calculate_attacker_reward(
                            "ddos_success", node_value=self.nodes[current_node]["value"])
                        self.log(f"💥 DDoS hit {current_node}!")

                    elif attacker_action == 4:
                        # Ransomware
                        self.compromised[current_node] = True
                        attacker_reward += calculate_attacker_reward("ransomware_success")
                        defender_reward += calculate_defender_reward(
                            "critical_node_compromised", attack_damage=8)
                        self.attacker_score += 80
                        self.log(f"💰 RANSOMWARE deployed on {current_node}!")

                    elif attacker_action == 8:
                        # Data exfiltration
                        if self.compromised[current_node]:
                            attacker_reward += calculate_attacker_reward("data_exfil_success")
                            self.attacker_score += 60
                            self.log(f"📤 Data exfiltrated from {current_node}!")
                        else:
                            attacker_reward += calculate_attacker_reward("wasted_action")

                    else:
                        # Standard compromise
                        if not self.compromised[current_node]:
                            self.compromised[current_node] = True
                            node_val = self.nodes[current_node]["value"]
                            attacker_reward += calculate_attacker_reward(
                                "exploit_success",
                                node_value=node_val,
                                damage=attack["damage"],
                                stealth=attack["stealth"]
                            )
                            self.attacker_score += node_val * 10
                            self.log(f"{attack['emoji']} {attack['name']} succeeded on {current_node}!")

                            if self.nodes[current_node]["critical"]:
                                attacker_reward += calculate_attacker_reward(
                                    "critical_node_reached", damage=attack["damage"])
                                defender_reward += calculate_defender_reward(
                                    "critical_node_compromised",
                                    attack_damage=attack["damage"])
                                self.attacker_won = True
                                done = True
                                self.log(f"🚨 CRITICAL NODE {current_node} COMPROMISED!")
                        else:
                            attacker_reward += calculate_attacker_reward("wasted_action")

                    # Detection check
                    detect_chance = self.detection_score * (1 - attack["stealth"])
                    if random.random() < detect_chance:
                        self.attacker_detected = True
                        self.detection_count += 1
                        attacker_reward += calculate_attacker_reward(
                            "detected", stealth=attack["stealth"])
                        defender_reward += calculate_defender_reward(
                            "attack_detected", attack_damage=attack["damage"])
                        self.log(f"👁️ Attack detected! ({attack['name']})")

                else:
                    attacker_reward += calculate_attacker_reward("wasted_action")
                    self.log(f"❌ {attack['name']} failed on {current_node}")

        # ─── DEFENDER ACTS ────────────────────────────────────────────────────

        if not done:
            if defender_action == 0:
                # Monitor
                self.detection_score = min(0.8, self.detection_score + 0.02)
                self.log(f"👁️ Monitoring — detection: {self.detection_score:.2f}")

            elif defender_action == 1:
                # Block IP
                compromised_list = [n for n in self.compromised if self.compromised[n]]
                if compromised_list:
                    target = random.choice(compromised_list)
                    self.blocked_nodes.add(target)
                    defender_reward += calculate_defender_reward(
                        "attack_blocked", attack_damage=attack["damage"])
                    self.defender_score += 25
                    self.log(f"🚫 Blocked {target}")

            elif defender_action == 2:
                # Patch
                unpatched = [n for n in self.nodes if n not in self.patched_nodes]
                if unpatched:
                    target = random.choice(unpatched)
                    self.patched_nodes.add(target)
                    self.nodes[target]["vulnerability"] = max(
                        0.05, self.nodes[target]["vulnerability"] - 0.25)
                    defender_reward += calculate_defender_reward("patch_success")
                    self.defender_score += 10
                    self.log(f"🔧 Patched {target}")

            elif defender_action == 3:
                # Deploy honeypot
                uncompromised = [n for n in self.nodes
                                if not self.compromised[n]
                                and n not in self.honeypots]
                if uncompromised:
                    target = random.choice(uncompromised)
                    self.honeypots.add(target)
                    self.log(f"🍯 Honeypot deployed at {target}")

            elif defender_action == 4:
                # Firewall rule
                self.detection_score = min(0.8, self.detection_score + 0.08)
                self.log(f"🛡️ Firewall rule added")

            elif defender_action == 5:
                # Antivirus scan
                found = [n for n in self.compromised
                        if self.compromised[n]
                        and n not in self.blocked_nodes]
                if found and random.random() < 0.4:
                    target = random.choice(found)
                    self.compromised[target] = False
                    defender_reward += calculate_defender_reward(
                        "attack_blocked", attack_damage=3)
                    self.defender_score += 20
                    self.log(f"🔍 Antivirus cleaned {target}!")

            elif defender_action == 6:
                # Isolate node
                if self.attacker_position not in self.isolated_nodes:
                    self.isolated_nodes.add(self.attacker_position)
                    defender_reward += calculate_defender_reward(
                        "attack_blocked", attack_damage=attack["damage"])
                    self.defender_score += 15
                    self.log(f"🔒 Isolated {self.attacker_position}")

            elif defender_action == 7:
                # Reset credentials — reduces vulnerability
                self.nodes[self.attacker_position]["vulnerability"] = max(
                    0.1, self.nodes[self.attacker_position]["vulnerability"] - 0.15)
                self.log(f"🔑 Credentials reset on {self.attacker_position}")

            elif defender_action == 8:
                # Deploy IDS
                self.ids_active = True
                self.detection_score = min(0.8, self.detection_score + 0.15)
                self.defender_score += 10
                self.log(f"🚨 IDS deployed!")

            elif defender_action == 9:
                # Backup — prevents ransomware from winning
                self.log(f"💾 Systems backed up")

            elif defender_action == 10:
                # Threat intelligence
                self.detection_score = min(0.8, self.detection_score + 0.1)
                self.log(f"🧠 Threat intel gathered")

        # ─── END CONDITIONS ───────────────────────────────────────────────────

        if self.current_step >= 50:
            done = True

        critical_nodes = [n for n in self.nodes if self.nodes[n]["critical"]]
        if all(self.compromised[n] for n in critical_nodes):
            self.attacker_won = True
            done = True

        return self._get_state(), attacker_reward, defender_reward, done

    def log(self, message):
        self.battle_log.append(f"Step {self.current_step}: {message}")
        if len(self.battle_log) > 20:
            self.battle_log.pop(0)

    def get_info(self):
        return {
            "compromised": self.compromised,
            "attacker_position": self.attacker_position,
            "detection_score": round(self.detection_score, 2),
            "blocked_nodes": list(self.blocked_nodes),
            "patched_nodes": list(self.patched_nodes),
            "honeypots": list(self.honeypots),
            "isolated_nodes": list(self.isolated_nodes),
            "ids_active": self.ids_active,
            "attacker_won": self.attacker_won,
            "detection_count": self.detection_count,
            "attacker_score": self.attacker_score,
            "defender_score": self.defender_score,
            "last_attack": self.last_attack,
            "last_defense": self.last_defense,
            "battle_log": self.battle_log
        }