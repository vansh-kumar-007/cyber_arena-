# env/network_env.py
import random
from configs.network_config import NODES, NETWORK_TOPOLOGY, ATTACKER_START_NODE
from env.reward import calculate_attacker_reward, calculate_defender_reward
import copy

class NetworkEnvironment:
    def __init__(self):
        self.topology = NETWORK_TOPOLOGY
        self.reset()

    def reset(self):
        # Deep copy nodes so vulnerability changes don't persist between episodes
        self.nodes = copy.deepcopy(NODES)

        self.compromised = {node: False for node in self.nodes}
        self.attacker_position = ATTACKER_START_NODE
        self.detection_score = 0.1
        self.blocked_nodes = set()
        self.patched_nodes = set()
        self.current_step = 0
        self.attacker_detected = False
        self.attacker_won = False
        self.detection_count = 0

        return self._get_state()

    def _get_state(self):
        state = []
        for node in self.nodes:
            state.append(1 if self.compromised[node] else 0)

        node_list = list(self.nodes.keys())
        state.append(node_list.index(self.attacker_position) / len(node_list))
        state.append(self.detection_score)
        state.append(self.current_step / 50)

        return tuple(state)

    def step(self, attacker_action, defender_action):
        self.current_step += 1
        attacker_reward = 0
        defender_reward = 0
        done = False

        # ─── ATTACKER ACTS FIRST ──────────────────────────────────────────
        # (changed order — attacker now acts before defender blocks)

        current_node = self.attacker_position

        if attacker_action == 0:
            # Scan — small exploration reward instead of penalty
            attacker_reward += 0

        elif attacker_action == 1:
            # Exploit current node
            if not self.compromised[current_node]:
                vuln = self.nodes[current_node]["vulnerability"]
                defense = self.detection_score * 0.1
                success_prob = max(0.15, vuln - defense)

                if random.random() < success_prob:
                    self.compromised[current_node] = True
                    node_val = self.nodes[current_node]["value"]
                    attacker_reward += calculate_attacker_reward(
                        "exploit_success", node_val
                    )
                    if self.nodes[current_node]["critical"]:
                        attacker_reward += calculate_attacker_reward(
                            "critical_node_reached"
                        )
                        defender_reward += calculate_defender_reward(
                            "critical_node_compromised"
                        )
                        self.attacker_won = True
                        done = True
                else:
                    attacker_reward += calculate_attacker_reward("wasted_action")
            else:
                # Already compromised — try to move instead
                attacker_reward += calculate_attacker_reward("wasted_action")

        elif attacker_action == 2:
            # Move laterally
            neighbors = self.topology.get(current_node, [])
            reachable = [n for n in neighbors if n not in self.blocked_nodes]
            if reachable:
                self.attacker_position = random.choice(reachable)
            else:
                attacker_reward += calculate_attacker_reward("wasted_action")

        elif attacker_action == 3:
            # Privilege escalation
            neighbors = self.topology.get(current_node, [])
            compromised_neighbors = [n for n in neighbors if self.compromised[n]]
            if compromised_neighbors:
                target = random.choice(compromised_neighbors)
                node_val = self.nodes[target]["value"]
                attacker_reward += calculate_attacker_reward(
                    "exploit_success", node_val
                )
            else:
                attacker_reward += calculate_attacker_reward("wasted_action")

        elif attacker_action == 4:
            attacker_reward += calculate_attacker_reward("wasted_action")

        # ─── DEFENDER ACTS SECOND ─────────────────────────────────────────

        if not done:
            if defender_action == 0:
                # Monitor
                if random.random() < self.detection_score:
                    self.attacker_detected = True
                    self.detection_count += 1
                    attacker_reward += calculate_attacker_reward("detected")
                    defender_reward += calculate_defender_reward("attack_detected")

            elif defender_action == 1:
                # Block a RANDOM compromised node (not always attacker position)
                compromised_list = [n for n in self.compromised if self.compromised[n]]
                if compromised_list:
                    node_to_block = random.choice(compromised_list)
                    self.blocked_nodes.add(node_to_block)
                    defender_reward += calculate_defender_reward("attack_blocked")
                # If nothing compromised yet, blocking does nothing

            elif defender_action == 2:
                # Patch a random unpatched node
                unpatched = [n for n in self.nodes if n not in self.patched_nodes]
                if unpatched:
                    node_to_patch = random.choice(unpatched)
                    self.patched_nodes.add(node_to_patch)
                    self.nodes[node_to_patch]["vulnerability"] = max(
                        0.0, self.nodes[node_to_patch]["vulnerability"] - 0.2
                    )

            elif defender_action == 3:
                self.detection_score = min(0.6, self.detection_score + 0.05)

            elif defender_action == 4:
                pass

        # ─── CHECK END CONDITIONS ─────────────────────────────────────────

        if self.current_step >= 50:
            done = True

        critical_nodes = [n for n in self.nodes if self.nodes[n]["critical"]]
        if all(self.compromised[n] for n in critical_nodes):
            self.attacker_won = True
            done = True

        next_state = self._get_state()
        return next_state, attacker_reward, defender_reward, done

    def get_info(self):
        return {
            "compromised": self.compromised,
            "attacker_position": self.attacker_position,
            "detection_score": round(self.detection_score, 2),
            "blocked_nodes": list(self.blocked_nodes),
            "patched_nodes": list(self.patched_nodes),
            "attacker_won": self.attacker_won,
            "detection_count": self.detection_count
        }