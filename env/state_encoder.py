"""Stable encoder for the fixed CyberArena observation vector.

Feature ordering is part of the model contract. Keep it unchanged unless model
checkpoints are retrained and versioned with the new observation schema.
"""
from __future__ import annotations

from typing import Mapping, Sequence


MAX_AGENTS = 4
FEATURES_PER_NODE = 4
GLOBAL_FEATURES = 5


def state_size(node_count: int, max_agents: int = MAX_AGENTS) -> int:
    if node_count < 1 or max_agents < 1:
        raise ValueError("node_count and max_agents must be positive")
    return node_count * FEATURES_PER_NODE + max_agents * 2 + GLOBAL_FEATURES


def encode_state(
    *,
    nodes: Mapping[str, Mapping[str, object]],
    compromised: Mapping[str, bool],
    blocked_nodes: set[str],
    honeypots: set[str],
    attacker_positions: Sequence[str],
    defender_positions: Sequence[str],
    detection_score: float,
    current_step: int,
    ids_active: bool,
    n_attackers: int,
    n_defenders: int,
    max_agents: int = MAX_AGENTS,
) -> tuple[float, ...]:
    """Encode state into the original ordered vector used by saved 37-input models."""
    if len(attacker_positions) > max_agents or len(defender_positions) > max_agents:
        raise ValueError(f"agent counts cannot exceed {max_agents}")
    node_names = list(nodes.keys())
    if not node_names:
        raise ValueError("at least one network node is required")
    if any(position not in node_names for position in (*attacker_positions, *defender_positions)):
        raise ValueError("agent position references an unknown node")
    values: list[float] = []
    for name in node_names:
        values.extend((
            1.0 if compromised[name] else 0.0,
            float(nodes[name]["vulnerability"]),
            1.0 if name in blocked_nodes else 0.0,
            1.0 if name in honeypots else 0.0,
        ))
    for positions in (attacker_positions, defender_positions):
        for i in range(max_agents):
            values.append(node_names.index(positions[i]) / len(node_names) if i < len(positions) else -1.0)
    values.extend((
        float(detection_score),
        float(current_step) / 50.0,  # Preserve checkpoint input semantics.
        1.0 if ids_active else 0.0,
        float(n_attackers) / max_agents,
        float(n_defenders) / max_agents,
    ))
    return tuple(round(value, 2) for value in values)
