# env/network_env.py
# Multi-Agent version — supports configurable number of attackers and defenders
# Architecture: Centralized Training, Decentralized Execution (CTDE)

import random
import copy
from configs.network_config import (
    NODES, NETWORK_TOPOLOGY, ATTACKER_START_NODE,
    ATTACK_TYPES, DEFENSE_TYPES
)
from env.reward import calculate_attacker_reward, calculate_defender_reward

class NetworkEnvironment:
    def __init__(self, n_attackers=1, n_defenders=1):
        self.topology = NETWORK_TOPOLOGY
        self.n_attackers = n_attackers
        self.n_defenders = n_defenders
        self.n_attacker_actions = len(ATTACK_TYPES)
        self.n_defender_actions = len(DEFENSE_TYPES)

        # Starting positions spread across network
        self.attacker_starts = ["N1", "N3", "N5", "N1"]
        self.defender_starts = ["N2", "N4", "N6", "N2"]

        self.reset()

    def reset(self):
        self.nodes = copy.deepcopy(NODES)
        self.compromised = {node: False for node in self.nodes}
        self.blocked_nodes = set()
        self.patched_nodes = set()
        self.honeypots = set()
        self.isolated_nodes = set()
        self.ids_active = False
        self.detection_score = 0.1
        self.current_step = 0
        self.attacker_won = False
        self.detection_count = 0
        self.attacker_score = 0
        self.defender_score = 0
        self.last_attack = None
        self.last_defense = None
        self.battle_log = []

        # Each attacker starts at different node
        self.attacker_positions = [
            self.attacker_starts[i % len(self.attacker_starts)]
            for i in range(self.n_attackers)
        ]

        # Each defender patrols different node
        self.defender_positions = [
            self.defender_starts[i % len(self.defender_starts)]
            for i in range(self.n_defenders)
        ]

        return self._get_state()

    def _get_state(self):
        """
        State includes: node statuses + all agent positions
        Each agent sees the full global state (centralized observation)
        """
        state = []
        node_list = list(self.nodes.keys())

        # Node statuses
        for node in node_list:
            state.append(1 if self.compromised[node] else 0)
            state.append(self.nodes[node]["vulnerability"])
            state.append(1 if node in self.blocked_nodes else 0)
            state.append(1 if node in self.honeypots else 0)

        # All attacker positions (padded to max 4)
        for i in range(4):
            if i < len(self.attacker_positions):
                pos = node_list.index(self.attacker_positions[i])
                state.append(pos / len(node_list))
            else:
                state.append(-1)  # No agent

        # All defender positions (padded to max 4)
        for i in range(4):
            if i < len(self.defender_positions):
                pos = node_list.index(self.defender_positions[i])
                state.append(pos / len(node_list))
            else:
                state.append(-1)

        state.append(self.detection_score)
        state.append(self.current_step / 50)
        state.append(1 if self.ids_active else 0)
        state.append(self.n_attackers / 4)
        state.append(self.n_defenders / 4)

        return tuple(round(x, 2) for x in state)

    def step(self, attacker_actions, defender_actions):
        """
        attacker_actions: list of actions, one per attacker
        defender_actions: list of actions, one per defender
        """
        self.current_step += 1
        total_att_reward = 0
        total_def_reward = 0
        done = False

        # ─── ALL ATTACKERS ACT ────────────────────────────────────────────────
        for i, (action, position) in enumerate(
            zip(attacker_actions, self.attacker_positions)
        ):
            att_reward, new_pos, done_flag = self._attacker_step(
                i, action, position
            )
            self.attacker_positions[i] = new_pos
            total_att_reward += att_reward
            if done_flag:
                done = True

        # ─── ALL DEFENDERS ACT ────────────────────────────────────────────────
        for i, (action, position) in enumerate(
            zip(defender_actions, self.defender_positions)
        ):
            def_reward, new_pos = self._defender_step(i, action, position)
            self.defender_positions[i] = new_pos
            total_def_reward += def_reward

        # ─── END CONDITIONS ───────────────────────────────────────────────────
        if self.current_step >= 50:
            done = True

        critical_nodes = [n for n in self.nodes if self.nodes[n]["critical"]]
        if all(self.compromised[n] for n in critical_nodes):
            self.attacker_won = True
            done = True

        # Average rewards across agents
        avg_att_reward = total_att_reward / self.n_attackers
        avg_def_reward = total_def_reward / self.n_defenders

        return self._get_state(), avg_att_reward, avg_def_reward, done

    def _attacker_step(self, agent_idx, action, current_node):
        """Single attacker agent step"""
        att_reward = 0
        new_pos = current_node
        done = False
        attack = ATTACK_TYPES[action]
        self.last_attack = attack

        if current_node in self.honeypots:
            self.detection_count += 1
            att_reward += calculate_attacker_reward("detected",
                stealth=attack["stealth"])
            self.log(f"🍯 Attacker {agent_idx+1} caught in honeypot at {current_node}!")
            return att_reward, new_pos, done

        if action == 11:  # Idle
            return 0, new_pos, done

        if action == 6:  # Lateral movement
            neighbors = self.topology.get(current_node, [])
            reachable = [n for n in neighbors
                        if n not in self.blocked_nodes
                        and n not in self.isolated_nodes]
            if reachable:
                new_pos = random.choice(reachable)
                if not self.compromised[new_pos]:
                    att_reward += calculate_attacker_reward("move_to_new_node")
                self.log(f"🔀 Attacker {agent_idx+1} moved to {new_pos}")
            return att_reward, new_pos, done

        # Attack current node
        if current_node not in self.blocked_nodes and \
           current_node not in self.isolated_nodes:
            vuln = self.nodes[current_node]["vulnerability"]
            defense = self.detection_score * 0.1
            ids_penalty = 0.1 if self.ids_active else 0
            success_prob = max(0.1, attack["base_success"] * vuln
                             - defense - ids_penalty)

            if random.random() < success_prob:
                if action == 4:  # Ransomware
                    self.compromised[current_node] = True
                    att_reward += calculate_attacker_reward("ransomware_success")
                    self.attacker_score += 80
                    self.log(f"💰 Attacker {agent_idx+1}: RANSOMWARE on {current_node}!")
                elif action == 8:  # Data exfil
                    if self.compromised[current_node]:
                        att_reward += calculate_attacker_reward("data_exfil_success")
                        self.attacker_score += 60
                        self.log(f"📤 Attacker {agent_idx+1}: Data exfil from {current_node}!")
                    else:
                        att_reward += calculate_attacker_reward("wasted_action")
                elif action == 2:  # DDoS
                    att_reward += calculate_attacker_reward("ddos_success",
                        node_value=self.nodes[current_node]["value"])
                    self.log(f"💥 Attacker {agent_idx+1}: DDoS on {current_node}!")
                else:
                    if not self.compromised[current_node]:
                        self.compromised[current_node] = True
                        node_val = self.nodes[current_node]["value"]
                        att_reward += calculate_attacker_reward(
                            "exploit_success",
                            node_value=node_val,
                            damage=attack["damage"],
                            stealth=attack["stealth"]
                        )
                        self.attacker_score += node_val * 10
                        self.log(f"{attack['emoji']} Attacker {agent_idx+1}: "
                                f"{attack['name']} on {current_node}!")

                        if self.nodes[current_node]["critical"]:
                            att_reward += calculate_attacker_reward(
                                "critical_node_reached",
                                damage=attack["damage"]
                            )
                            self.attacker_won = True
                            done = True
                            self.log(f"🚨 CRITICAL NODE {current_node} COMPROMISED!")
                    else:
                        att_reward += calculate_attacker_reward("wasted_action")

                # Detection check
                detect_chance = self.detection_score * (1 - attack["stealth"])
                if self.ids_active:
                    detect_chance = min(1.0, detect_chance * 1.5)
                if random.random() < detect_chance:
                    self.detection_count += 1
                    att_reward += calculate_attacker_reward("detected",
                        stealth=attack["stealth"])
                    self.log(f"👁️ Attacker {agent_idx+1} detected!")
            else:
                att_reward += calculate_attacker_reward("wasted_action")
        else:
            att_reward += calculate_attacker_reward("wasted_action")

        return att_reward, new_pos, done

    def _defender_step(self, agent_idx, action, current_node):
        """Single defender agent step"""
        def_reward = 0
        new_pos = current_node
        defense = DEFENSE_TYPES[action]
        self.last_defense = defense

        if action == 0:  # Monitor
            self.detection_score = min(0.8, self.detection_score + 0.02)

        elif action == 1:  # Block IP
            compromised = [n for n in self.compromised if self.compromised[n]]
            if compromised:
                target = random.choice(compromised)
                self.blocked_nodes.add(target)
                def_reward += calculate_defender_reward("attack_blocked", 2)
                self.defender_score += 25
                self.log(f"🚫 Defender {agent_idx+1}: Blocked {target}")

        elif action == 2:  # Patch
            unpatched = [n for n in self.nodes if n not in self.patched_nodes]
            if unpatched:
                target = random.choice(unpatched)
                self.patched_nodes.add(target)
                self.nodes[target]["vulnerability"] = max(
                    0.05, self.nodes[target]["vulnerability"] - 0.25)
                def_reward += calculate_defender_reward("patch_success")
                self.defender_score += 10
                self.log(f"🔧 Defender {agent_idx+1}: Patched {target}")

        elif action == 3:  # Honeypot
            uncompromised = [n for n in self.nodes
                           if not self.compromised[n]
                           and n not in self.honeypots]
            if uncompromised:
                target = random.choice(uncompromised)
                self.honeypots.add(target)
                self.log(f"🍯 Defender {agent_idx+1}: Honeypot at {target}")

        elif action == 4:  # Firewall
            self.detection_score = min(0.8, self.detection_score + 0.08)

        elif action == 5:  # Antivirus
            found = [n for n in self.compromised
                    if self.compromised[n]
                    and n not in self.blocked_nodes]
            if found and random.random() < 0.4:
                target = random.choice(found)
                self.compromised[target] = False
                def_reward += calculate_defender_reward("attack_blocked", 3)
                self.defender_score += 20
                self.log(f"🔍 Defender {agent_idx+1}: Cleaned {target}!")

        elif action == 6:  # Isolate
            # Defender moves to most threatened node
            att_positions = set(self.attacker_positions)
            if att_positions:
                target = random.choice(list(att_positions))
                self.isolated_nodes.add(target)
                def_reward += calculate_defender_reward("attack_blocked", 2)
                self.defender_score += 15
                self.log(f"🔒 Defender {agent_idx+1}: Isolated {target}")

        elif action == 7:  # Reset credentials
            self.nodes[current_node]["vulnerability"] = max(
                0.1, self.nodes[current_node]["vulnerability"] - 0.15)

        elif action == 8:  # Deploy IDS
            self.ids_active = True
            self.detection_score = min(0.8, self.detection_score + 0.15)
            self.defender_score += 10
            self.log(f"🚨 Defender {agent_idx+1}: IDS deployed!")

        elif action == 10:  # Threat intel
            self.detection_score = min(0.8, self.detection_score + 0.1)

        return def_reward, new_pos

    def log(self, message):
        self.battle_log.append(f"Step {self.current_step}: {message}")
        if len(self.battle_log) > 20:
            self.battle_log.pop(0)

    def get_info(self):
        return {
            "compromised": self.compromised,
            "attacker_positions": self.attacker_positions,
            "attacker_position": self.attacker_positions[0],
            "defender_positions": self.defender_positions,
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
            "battle_log": self.battle_log,
            "n_attackers": self.n_attackers,
            "n_defenders": self.n_defenders,
        }