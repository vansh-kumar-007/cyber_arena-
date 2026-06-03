# env/network_env.py
# This is the CORE of the entire simulation.
# Think of it as the "game board" — it manages the state of the network,
# processes actions from both agents, and returns rewards.

import random
from configs.network_config import NODES, NETWORK_TOPOLOGY, ATTACKER_START_NODE
from env.reward import calculate_attacker_reward, calculate_defender_reward

class NetworkEnvironment:
    def __init__(self):
        self.nodes = NODES
        self.topology = NETWORK_TOPOLOGY
        self.reset()

    def reset(self):
        """
        Reset everything back to the start of a new episode.
        Called at the beginning of every training episode.
        """
        # Which nodes have been compromised by attacker (all start as False)
        self.compromised = {node: False for node in self.nodes}

        # Where is the attacker right now
        self.attacker_position = ATTACKER_START_NODE

        # How likely is the defender to detect the attacker (0.0 to 1.0)
        self.detection_score = 0.2

        # Which nodes has the defender blocked
        self.blocked_nodes = set()

        # Which nodes have been patched by defender
        self.patched_nodes = set()

        # Step counter
        self.current_step = 0

        # Was the attacker detected this episode
        self.attacker_detected = False

        # Did the attacker win (reach a critical node)
        self.attacker_won = False

        # How many times defender detected attacker
        self.detection_count = 0

        return self._get_state()

    def _get_state(self):
        """
        Convert the current situation into a list of numbers.
        This is what the RL agents actually see and learn from.
        """
        state = []

        # Is each node compromised? (0 or 1 for each node)
        for node in self.nodes:
            state.append(1 if self.compromised[node] else 0)

        # Where is the attacker? (encoded as a number 0-3)
        node_list = list(self.nodes.keys())
        state.append(node_list.index(self.attacker_position) / len(node_list))

        # Current detection score (already 0-1)
        state.append(self.detection_score)

        # Normalized step count
        state.append(self.current_step / 50)

        return tuple(state)  # Tuples can be used as dictionary keys (for Q-table)

    def step(self, attacker_action, defender_action):
        """
        The main function — both agents take an action,
        the world updates, and rewards are returned.

        Attacker actions:
            0 = Scan network
            1 = Exploit current node
            2 = Move laterally (move to a connected node)
            3 = Privilege escalation (try to compromise a higher value node)
            4 = Stay idle

        Defender actions:
            0 = Monitor traffic
            1 = Block current attacker node
            2 = Patch a vulnerability
            3 = Increase detection sensitivity
            4 = Do nothing
        """
        self.current_step += 1
        attacker_reward = 0
        defender_reward = 0
        done = False

        # ─── DEFENDER ACTS FIRST ───────────────────────────────────────────

        if defender_action == 0:
            # Monitor: small chance to detect attacker
            if random.random() < self.detection_score:
                self.attacker_detected = True
                self.detection_count += 1
                attacker_reward += calculate_attacker_reward("detected")
                defender_reward += calculate_defender_reward("attack_detected")

        elif defender_action == 1:
            # Block the node attacker is currently on
            self.blocked_nodes.add(self.attacker_position)
            defender_reward += calculate_defender_reward("attack_blocked")
            attacker_reward += calculate_attacker_reward("detected")

        elif defender_action == 2:
            # Patch a random unpatched node
            unpatched = [n for n in self.nodes if n not in self.patched_nodes]
            if unpatched:
                node_to_patch = random.choice(unpatched)
                self.patched_nodes.add(node_to_patch)
                # Patching reduces vulnerability
                self.nodes[node_to_patch]["vulnerability"] = max(
                    0.0, self.nodes[node_to_patch]["vulnerability"] - 0.2
                )

        elif defender_action == 3:
            # Increase detection sensitivity
            self.detection_score = min(1.0, self.detection_score + 0.1)

        elif defender_action == 4:
            # Do nothing
            defender_reward += calculate_defender_reward("none")

        # ─── ATTACKER ACTS ─────────────────────────────────────────────────

        current_node = self.attacker_position

        # If attacker's current node is blocked, penalize movement
        if current_node in self.blocked_nodes:
            attacker_reward += calculate_attacker_reward("wasted_action")
        else:
            if attacker_action == 0:
                # Scan: reveals network info (small reward for now, useful later)
                attacker_reward += calculate_attacker_reward("wasted_action")

            elif attacker_action == 1:
                # Exploit: try to compromise current node
                if not self.compromised[current_node]:
                    vuln = self.nodes[current_node]["vulnerability"]
                    defense = self.detection_score * 0.3
                    success_prob = max(0, vuln - defense)

                    if random.random() < success_prob:
                        self.compromised[current_node] = True
                        node_val = self.nodes[current_node]["value"]
                        attacker_reward += calculate_attacker_reward(
                            "exploit_success", node_val
                        )

                        # Extra reward if it's a critical node
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
                    # Node already compromised
                    attacker_reward += calculate_attacker_reward("wasted_action")

            elif attacker_action == 2:
                # Move laterally to a connected node
                neighbors = self.topology.get(current_node, [])
                reachable = [n for n in neighbors if n not in self.blocked_nodes]
                if reachable:
                    self.attacker_position = random.choice(reachable)
                else:
                    attacker_reward += calculate_attacker_reward("wasted_action")

            elif attacker_action == 3:
                # Privilege escalation — try to compromise a connected already-visited node
                neighbors = self.topology.get(current_node, [])
                compromised_neighbors = [
                    n for n in neighbors if self.compromised[n]
                ]
                if compromised_neighbors:
                    target = random.choice(compromised_neighbors)
                    node_val = self.nodes[target]["value"]
                    attacker_reward += calculate_attacker_reward(
                        "exploit_success", node_val
                    )
                else:
                    attacker_reward += calculate_attacker_reward("wasted_action")

            elif attacker_action == 4:
                # Idle
                attacker_reward += calculate_attacker_reward("wasted_action")

        # ─── CHECK END CONDITIONS ──────────────────────────────────────────

        # Episode ends if max steps reached
        if self.current_step >= 50:
            done = True

        # If all critical nodes compromised
        critical_nodes = [n for n in self.nodes if self.nodes[n]["critical"]]
        if all(self.compromised[n] for n in critical_nodes):
            self.attacker_won = True
            done = True

        next_state = self._get_state()
        return next_state, attacker_reward, defender_reward, done

    def get_info(self):
        """Returns a summary of the current environment state — useful for logging"""
        return {
            "compromised": self.compromised,
            "attacker_position": self.attacker_position,
            "detection_score": round(self.detection_score, 2),
            "blocked_nodes": list(self.blocked_nodes),
            "patched_nodes": list(self.patched_nodes),
            "attacker_won": self.attacker_won,
            "detection_count": self.detection_count
        }