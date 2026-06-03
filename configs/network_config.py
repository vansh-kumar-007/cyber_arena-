# configs/network_config.py

NODES = {
    "N1": {
        "name": "Web Server",
        "value": 2,
        "vulnerability": 0.8,
        "critical": False,
        "type": "web"
    },
    "N2": {
        "name": "DB Server",
        "value": 5,
        "vulnerability": 0.6,
        "critical": True,
        "type": "database"
    },
    "N3": {
        "name": "User PC",
        "value": 1,
        "vulnerability": 0.95,
        "critical": False,
        "type": "endpoint"
    },
    "N4": {
        "name": "Admin Node",
        "value": 10,
        "vulnerability": 0.4,
        "critical": True,
        "type": "admin"
    },
    "N5": {
        "name": "Email Server",
        "value": 3,
        "vulnerability": 0.75,
        "critical": False,
        "type": "email"
    },
    "N6": {
        "name": "Firewall",
        "value": 8,
        "vulnerability": 0.2,
        "critical": True,
        "type": "firewall"
    }
}

NETWORK_TOPOLOGY = {
    "N1": ["N2", "N3", "N5"],
    "N2": ["N1", "N4"],
    "N3": ["N1", "N5"],
    "N4": ["N2", "N6"],
    "N5": ["N1", "N3"],
    "N6": ["N4"]
}

ATTACKER_START_NODE = "N1"

# ─── ATTACK TYPES ─────────────────────────────────────────────────────────────
ATTACK_TYPES = {
    0:  {"name": "Phishing",             "emoji": "🎣", "base_success": 0.7,  "stealth": 0.9, "damage": 1},
    1:  {"name": "Exploit CVE",          "emoji": "💉", "base_success": 0.5,  "stealth": 0.6, "damage": 2},
    2:  {"name": "DDoS",                 "emoji": "💥", "base_success": 0.8,  "stealth": 0.1, "damage": 1},
    3:  {"name": "Malware Deploy",       "emoji": "🦠", "base_success": 0.6,  "stealth": 0.7, "damage": 3},
    4:  {"name": "Ransomware",           "emoji": "💰", "base_success": 0.4,  "stealth": 0.5, "damage": 5},
    5:  {"name": "Social Engineering",   "emoji": "🎭", "base_success": 0.65, "stealth": 0.95,"damage": 2},
    6:  {"name": "Lateral Movement",     "emoji": "🔀", "base_success": 0.75, "stealth": 0.8, "damage": 0},
    7:  {"name": "Privilege Escalation", "emoji": "⬆️", "base_success": 0.5,  "stealth": 0.6, "damage": 3},
    8:  {"name": "Data Exfiltration",    "emoji": "📤", "base_success": 0.6,  "stealth": 0.7, "damage": 4},
    9:  {"name": "Zero Day Exploit",     "emoji": "🌟", "base_success": 0.3,  "stealth": 0.5, "damage": 8},
    10: {"name": "Brute Force",          "emoji": "🔨", "base_success": 0.4,  "stealth": 0.2, "damage": 2},
    11: {"name": "Stay Idle",            "emoji": "😴", "base_success": 1.0,  "stealth": 1.0, "damage": 0},
}

# ─── DEFENSE TYPES ────────────────────────────────────────────────────────────
DEFENSE_TYPES = {
    0:  {"name": "Monitor Traffic",      "emoji": "👁️",  "detect_boost": 0.1,  "cost": 0},
    1:  {"name": "Block IP",             "emoji": "🚫",  "detect_boost": 0.0,  "cost": 1},
    2:  {"name": "Patch Vulnerability",  "emoji": "🔧",  "detect_boost": 0.05, "cost": 2},
    3:  {"name": "Deploy Honeypot",      "emoji": "🍯",  "detect_boost": 0.2,  "cost": 1},
    4:  {"name": "Firewall Rule",        "emoji": "🛡️",  "detect_boost": 0.15, "cost": 1},
    5:  {"name": "Antivirus Scan",       "emoji": "🔍",  "detect_boost": 0.1,  "cost": 1},
    6:  {"name": "Isolate Node",         "emoji": "🔒",  "detect_boost": 0.0,  "cost": 2},
    7:  {"name": "Reset Credentials",    "emoji": "🔑",  "detect_boost": 0.05, "cost": 1},
    8:  {"name": "Deploy IDS",           "emoji": "🚨",  "detect_boost": 0.25, "cost": 2},
    9:  {"name": "Backup Systems",       "emoji": "💾",  "detect_boost": 0.0,  "cost": 1},
    10: {"name": "Threat Intelligence",  "emoji": "🧠",  "detect_boost": 0.15, "cost": 1},
    11: {"name": "Do Nothing",           "emoji": "😴",  "detect_boost": 0.0,  "cost": 0},
}