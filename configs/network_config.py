# configs/network_config.py

NODES = {
    "N1": {
        "name": "Web Server",
        "value": 2,
        "vulnerability": 0.8,   # increased
        "critical": False
    },
    "N2": {
        "name": "DB Server",
        "value": 5,
        "vulnerability": 0.6,   # increased
        "critical": True
    },
    "N3": {
        "name": "User PC",
        "value": 1,
        "vulnerability": 0.95,  # increased
        "critical": False
    },
    "N4": {
        "name": "Admin Node",
        "value": 10,
        "vulnerability": 0.4,   # increased
        "critical": True
    }
}

NETWORK_TOPOLOGY = {
    "N1": ["N2", "N3"],
    "N2": ["N1", "N4"],
    "N3": ["N1"],
    "N4": ["N2"]
}

ATTACKER_START_NODE = "N1"