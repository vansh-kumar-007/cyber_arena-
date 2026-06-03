# configs/network_config.py
# This file defines what our simulated network looks like.
# Think of it as the "map" of the game.

NODES = {
    "N1": {
        "name": "Web Server",
        "value": 2,           # How valuable this node is to the attacker
        "vulnerability": 0.7, # How easy it is to hack (0 = impossible, 1 = trivial)
        "critical": False     # Is this a critical/high-value target?
    },
    "N2": {
        "name": "DB Server",
        "value": 5,
        "vulnerability": 0.5,
        "critical": True
    },
    "N3": {
        "name": "User PC",
        "value": 1,
        "vulnerability": 0.9,
        "critical": False
    },
    "N4": {
        "name": "Admin Node",
        "value": 10,
        "vulnerability": 0.3,
        "critical": True
    }
}

# Which nodes are connected to each other
# Attacker can only move between connected nodes
NETWORK_TOPOLOGY = {
    "N1": ["N2", "N3"],   # Web Server connects to DB and User PC
    "N2": ["N1", "N4"],   # DB Server connects to Web Server and Admin
    "N3": ["N1"],          # User PC only connects to Web Server
    "N4": ["N2"]           # Admin Node only connects to DB Server
}

# Where the attacker starts
ATTACKER_START_NODE = "N1"