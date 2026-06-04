import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

// ─── API CONFIG ───────────────────────────────────────────────────────────────
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";
async function apiCall(endpoint, method = "GET") {
  try {
    const res = await fetch(`${API_URL}${endpoint}`, { method });
    const data = await res.json();
    return data.data;
  } catch (e) {
    return null;
  }
}

// ─── PIXEL ART STYLES ─────────────────────────────────────────────────────────
const PIXEL_FONT = "'Courier New', monospace";

const COLORS = {
  bg: "#0a0a0f",
  panel: "#12121a",
  border: "#2a2a4a",
  red: "#ff3333",
  redDim: "#8b0000",
  blue: "#3399ff",
  blueDim: "#003380",
  green: "#00ff88",
  greenDim: "#006644",
  yellow: "#ffdd00",
  purple: "#cc44ff",
  orange: "#ff8800",
  white: "#ffffff",
  gray: "#666688",
  scanline: "rgba(0,255,136,0.03)",
};

// ─── ATTACK & DEFENSE DATA ────────────────────────────────────────────────────
const ATTACKS = [
  { id: 0,  name: "Phishing",           emoji: "🎣", color: "#ff8800" },
  { id: 1,  name: "Exploit CVE",        emoji: "💉", color: "#ff3333" },
  { id: 2,  name: "DDoS",              emoji: "💥", color: "#ff0066" },
  { id: 3,  name: "Malware",           emoji: "🦠", color: "#cc44ff" },
  { id: 4,  name: "Ransomware",        emoji: "💰", color: "#ffdd00" },
  { id: 5,  name: "Social Eng.",       emoji: "🎭", color: "#ff8800" },
  { id: 6,  name: "Lateral Move",      emoji: "🔀", color: "#3399ff" },
  { id: 7,  name: "Priv. Escalation",  emoji: "⬆️", color: "#ff3333" },
  { id: 8,  name: "Data Exfil",        emoji: "📤", color: "#cc44ff" },
  { id: 9,  name: "Zero Day",          emoji: "🌟", color: "#ffdd00" },
  { id: 10, name: "Brute Force",       emoji: "🔨", color: "#ff8800" },
  { id: 11, name: "Idle",              emoji: "😴", color: "#666688" },
];

const DEFENSES = [
  { id: 0,  name: "Monitor",           emoji: "👁️",  color: "#3399ff" },
  { id: 1,  name: "Block IP",          emoji: "🚫",  color: "#ff3333" },
  { id: 2,  name: "Patch",             emoji: "🔧",  color: "#00ff88" },
  { id: 3,  name: "Honeypot",          emoji: "🍯",  color: "#ffdd00" },
  { id: 4,  name: "Firewall",          emoji: "🛡️",  color: "#3399ff" },
  { id: 5,  name: "Antivirus",         emoji: "🔍",  color: "#00ff88" },
  { id: 6,  name: "Isolate Node",      emoji: "🔒",  color: "#ff8800" },
  { id: 7,  name: "Reset Creds",       emoji: "🔑",  color: "#cc44ff" },
  { id: 8,  name: "Deploy IDS",        emoji: "🚨",  color: "#ff3333" },
  { id: 9,  name: "Backup",            emoji: "💾",  color: "#00ff88" },
  { id: 10, name: "Threat Intel",      emoji: "🧠",  color: "#cc44ff" },
  { id: 11, name: "Do Nothing",        emoji: "😴",  color: "#666688" },
];

const NODES = [
  { id: "N1", name: "Web Server",   emoji: "🌐", x: 200, y: 150 },
  { id: "N2", name: "DB Server",    emoji: "🗄️", x: 400, y: 150 },
  { id: "N3", name: "User PC",      emoji: "💻", x: 200, y: 300 },
  { id: "N4", name: "Admin Node",   emoji: "👑", x: 600, y: 150 },
  { id: "N5", name: "Email Server", emoji: "📧", x: 300, y: 300 },
  { id: "N6", name: "Firewall",     emoji: "🔥", x: 600, y: 300 },
];

const EDGES = [
  ["N1","N2"],["N1","N3"],["N1","N5"],
  ["N2","N4"],["N3","N5"],["N4","N6"],
];

// ─── MOCK SIMULATION ENGINE ───────────────────────────────────────────────────
function runSimulationStep(gameState) {
  const attacks = ATTACKS;
  const defenses = DEFENSES;

  const attackId = Math.floor(Math.random() * 12);
  const defenseId = Math.floor(Math.random() * 12);
  const attack = attacks[attackId];
  const defense = defenses[defenseId];

  let newState = { ...gameState };
  let logEntry = "";
  let event = null;

  const currentNodeId = newState.attackerPosition;

  // Attacker logic
  if (attackId === 6) {
    // Move
    const neighbors = {
      N1: ["N2","N3","N5"], N2: ["N1","N4"],
      N3: ["N1","N5"], N4: ["N2","N6"],
      N5: ["N1","N3"], N6: ["N4"]
    };
    const reachable = neighbors[currentNodeId] || [];
    if (reachable.length > 0) {
      newState.attackerPosition = reachable[Math.floor(Math.random() * reachable.length)];
      logEntry = `${attack.emoji} Attacker moved to ${newState.attackerPosition}`;
      event = "move";
    }
  } else if (attackId !== 11) {
    const successChance = 0.35 + Math.random() * 0.3;
    if (Math.random() < successChance) {
      const alreadyCompromised = newState.compromised[currentNodeId];
      if (!alreadyCompromised && attackId !== 8) {
        newState.compromised = { ...newState.compromised, [currentNodeId]: true };
        newState.redScore += Math.floor(Math.random() * 30) + 10;
        logEntry = `${attack.emoji} ${attack.name} succeeded on ${currentNodeId}!`;
        event = "compromise";
      } else if (attackId === 4 && alreadyCompromised) {
        newState.redScore += 80;
        logEntry = `💰 RANSOMWARE deployed on ${currentNodeId}!`;
        event = "ransomware";
      } else if (attackId === 8 && alreadyCompromised) {
        newState.redScore += 60;
        logEntry = `📤 Data exfiltrated from ${currentNodeId}!`;
        event = "exfil";
      } else {
        logEntry = `${attack.emoji} ${attack.name} failed`;
        event = "fail";
      }
    } else {
      logEntry = `${attack.emoji} ${attack.name} failed on ${currentNodeId}`;
      event = "fail";
    }
  }

  // Defender logic
  if (defenseId === 2) {
    // Patch
    newState.blueScore += 10;
    logEntry += ` | 🔧 Defender patched a node`;
  } else if (defenseId === 3) {
    newState.honeypots = [...new Set([...newState.honeypots, "N" + (Math.floor(Math.random()*6)+1)])];
    logEntry += ` | 🍯 Honeypot deployed`;
  } else if (defenseId === 8) {
    newState.idsActive = true;
    newState.blueScore += 10;
    logEntry += ` | 🚨 IDS deployed`;
  } else if (defenseId === 1) {
    const comp = Object.keys(newState.compromised).filter(k => newState.compromised[k]);
    if (comp.length > 0) {
      const target = comp[Math.floor(Math.random() * comp.length)];
      newState.blocked = [...new Set([...newState.blocked, target])];
      newState.blueScore += 25;
      logEntry += ` | 🚫 Blocked ${target}`;
      event = event || "block";
    }
  } else if (defenseId !== 11) {
    newState.blueScore += 5;
  }

  // Detection check
  if (event === "compromise" || event === "ransomware") {
    if (Math.random() < newState.detectionScore) {
      newState.blueScore += 15;
      logEntry += " | 👁️ DETECTED!";
      event = "detected";
    }
  }

  newState.detectionScore = Math.min(0.9,
    newState.detectionScore + (defenseId === 0 ? 0.02 : 0));
  newState.step += 1;
  newState.lastAttack = attack;
  newState.lastDefense = defense;
  newState.lastEvent = event;

  if (logEntry) {
    newState.battleLog = [
      `Step ${newState.step}: ${logEntry}`,
      ...newState.battleLog.slice(0, 14)
    ];
  }

  // Reward tracking
  newState.redRewards = [...newState.redRewards, newState.redScore];
  newState.blueRewards = [...newState.blueRewards, newState.blueScore];

  return newState;
}

const initialState = {
  compromised: { N1:false, N2:false, N3:false, N4:false, N5:false, N6:false },
  blocked: [],
  honeypots: [],
  attackerPosition: "N1",
  detectionScore: 0.1,
  idsActive: false,
  redScore: 0,
  blueScore: 0,
  step: 0,
  lastAttack: null,
  lastDefense: null,
  lastEvent: null,
  battleLog: [],
  redRewards: [],
  blueRewards: [],
  isRunning: false,
  speed: 500,
  sessionId: null,      
  startTime: null,
};

// ─── COMPONENTS ───────────────────────────────────────────────────────────────

function PixelBorder({ children, color = COLORS.border, style = {} }) {
  return (
    <div style={{
      border: `2px solid ${color}`,
      boxShadow: `0 0 10px ${color}40, inset 0 0 10px ${color}10`,
      background: COLORS.panel,
      ...style
    }}>
      {children}
    </div>
  );
}

function NetworkGraph({ gameState }) {
  const getNodeColor = (nodeId) => {
    if (gameState.compromised[nodeId]) return COLORS.red;
    if (gameState.blocked.includes(nodeId)) return COLORS.orange;
    if (gameState.honeypots.includes(nodeId)) return COLORS.yellow;
    return COLORS.green;
  };

  const getNodeGlow = (nodeId) => {
    if (gameState.attackerPosition === nodeId) return `0 0 20px ${COLORS.red}, 0 0 40px ${COLORS.red}80`;
    if (gameState.compromised[nodeId]) return `0 0 15px ${COLORS.red}80`;
    return `0 0 10px ${getNodeColor(nodeId)}40`;
  };

  return (
    <svg width="100%" height="100%" viewBox="0 0 800 450"
      style={{ background: "transparent" }}>

      {/* Scanlines effect */}
      {Array.from({length: 20}).map((_, i) => (
        <line key={i} x1="0" y1={i*24} x2="800" y2={i*24}
          stroke={COLORS.scanline} strokeWidth="1"/>
      ))}

      {/* Edges */}
      {EDGES.map(([a, b]) => {
        const na = NODES.find(n => n.id === a);
        const nb = NODES.find(n => n.id === b);
        const isHot = gameState.compromised[a] || gameState.compromised[b];
        return (
          <line key={`${a}-${b}`}
            x1={na.x} y1={na.y} x2={nb.x} y2={nb.y}
            stroke={isHot ? COLORS.redDim : COLORS.border}
            strokeWidth={isHot ? 2 : 1}
            strokeDasharray={isHot ? "5,5" : "none"}
          />
        );
      })}

      {/* Attacker path animation */}
      {gameState.lastEvent === "move" && (
        <motion.circle r="6" fill={COLORS.red} opacity={0.8}
          initial={{ opacity: 1, scale: 1 }}
          animate={{ opacity: 0, scale: 3 }}
          transition={{ duration: 0.5 }}
          cx={NODES.find(n => n.id === gameState.attackerPosition)?.x}
          cy={NODES.find(n => n.id === gameState.attackerPosition)?.y}
        />
      )}

      {/* Nodes */}
      {NODES.map(node => {
        const color = getNodeColor(node.id);
        const isAttackerHere = gameState.attackerPosition === node.id;
        return (
          <g key={node.id}>
            {/* Pulse ring for attacker position */}
            {isAttackerHere && (
              <motion.circle
                cx={node.x} cy={node.y} r="35"
                fill="none" stroke={COLORS.red} strokeWidth="2"
                animate={{ r: [35, 50], opacity: [0.8, 0] }}
                transition={{ duration: 1, repeat: Infinity }}
              />
            )}

            {/* Node box */}
            <rect
              x={node.x - 30} y={node.y - 25}
              width="60" height="50" rx="4"
              fill={COLORS.panel}
              stroke={color}
              strokeWidth={isAttackerHere ? 3 : 2}
              style={{ filter: `drop-shadow(${getNodeGlow(node.id)})` }}
            />

            {/* Emoji */}
            <text x={node.x} y={node.y - 5}
              textAnchor="middle" fontSize="16">{node.emoji}</text>

            {/* Node ID */}
            <text x={node.x} y={node.y + 12}
              textAnchor="middle" fontSize="9"
              fill={color} fontFamily={PIXEL_FONT}
              fontWeight="bold">{node.id}</text>

            {/* Status indicators */}
            {gameState.compromised[node.id] && (
              <text x={node.x + 28} y={node.y - 20}
                fontSize="12">💀</text>
            )}
            {gameState.honeypots.includes(node.id) && (
              <text x={node.x - 38} y={node.y - 20}
                fontSize="12">🍯</text>
            )}

            {/* Attacker skull */}
            {isAttackerHere && (
              <motion.text x={node.x} y={node.y - 40}
                textAnchor="middle" fontSize="18"
                animate={{ y: [node.y - 40, node.y - 45, node.y - 40] }}
                transition={{ duration: 1, repeat: Infinity }}
              >💀</motion.text>
            )}

            {/* Node name below */}
            <text x={node.x} y={node.y + 38}
              textAnchor="middle" fontSize="8"
              fill={COLORS.gray} fontFamily={PIXEL_FONT}>{node.name}</text>
          </g>
        );
      })}
    </svg>
  );
}

function Scoreboard({ redScore, blueScore, step, detectionScore, idsActive }) {
  return (
    <div style={{ display: "flex", gap: "12px", marginBottom: "12px" }}>
      <PixelBorder color={COLORS.red} style={{ flex: 1, padding: "12px", textAlign: "center" }}>
        <div style={{ color: COLORS.red, fontFamily: PIXEL_FONT, fontSize: "11px", letterSpacing: "2px" }}>
          ◄ RED TEAM
        </div>
        <motion.div
          key={redScore}
          animate={{ scale: [1.3, 1] }}
          transition={{ duration: 0.3 }}
          style={{ color: COLORS.red, fontFamily: PIXEL_FONT, fontSize: "32px", fontWeight: "bold" }}
        >
          {redScore}
        </motion.div>
        <div style={{ color: COLORS.gray, fontSize: "9px", fontFamily: PIXEL_FONT }}>ATTACKER POINTS</div>
      </PixelBorder>

      <PixelBorder color={COLORS.yellow} style={{ flex: 1, padding: "12px", textAlign: "center" }}>
        <div style={{ color: COLORS.yellow, fontFamily: PIXEL_FONT, fontSize: "10px" }}>STEP</div>
        <div style={{ color: COLORS.white, fontFamily: PIXEL_FONT, fontSize: "28px" }}>{step}</div>
        <div style={{ color: COLORS.gray, fontSize: "9px", fontFamily: PIXEL_FONT }}>
          DETECT: {Math.round(detectionScore * 100)}% {idsActive ? "🚨IDS" : ""}
        </div>
      </PixelBorder>

      <PixelBorder color={COLORS.blue} style={{ flex: 1, padding: "12px", textAlign: "center" }}>
        <div style={{ color: COLORS.blue, fontFamily: PIXEL_FONT, fontSize: "11px", letterSpacing: "2px" }}>
          BLUE TEAM ►
        </div>
        <motion.div
          key={blueScore}
          animate={{ scale: [1.3, 1] }}
          transition={{ duration: 0.3 }}
          style={{ color: COLORS.blue, fontFamily: PIXEL_FONT, fontSize: "32px", fontWeight: "bold" }}
        >
          {blueScore}
        </motion.div>
        <div style={{ color: COLORS.gray, fontSize: "9px", fontFamily: PIXEL_FONT }}>DEFENDER POINTS</div>
      </PixelBorder>
    </div>
  );
}

function ActionCard({ data, type, isActive }) {
  const color = type === "attack" ? data?.color || COLORS.red : data?.color || COLORS.blue;
  return (
    <motion.div
      animate={isActive ? { scale: [1, 1.05, 1], boxShadow: [`0 0 0px ${color}`, `0 0 20px ${color}`, `0 0 5px ${color}`] } : {}}
      transition={{ duration: 0.4 }}
      style={{
        border: `2px solid ${isActive ? color : COLORS.border}`,
        background: isActive ? `${color}20` : COLORS.panel,
        padding: "8px",
        borderRadius: "4px",
        textAlign: "center",
        minWidth: "80px",
        cursor: "default",
      }}
    >
      <div style={{ fontSize: "20px" }}>{data?.emoji || "❓"}</div>
      <div style={{ color: isActive ? color : COLORS.gray, fontFamily: PIXEL_FONT, fontSize: "8px", marginTop: "4px" }}>
        {data?.name || "???"}
      </div>
    </motion.div>
  );
}


// ─── NEURAL NETWORK VISUALIZER ────────────────────────────────────────────────
function NeuralNetModal({ onClose, lastAttack, lastDefense, step }) {
  const getActivations = (actionId, layerSize, layer) => {
    if (actionId === null || actionId === undefined) return Array(layerSize).fill(0.3);
    const seed = (actionId * 7 + layer * 13 + step) % 100;
    return Array(layerSize).fill(0).map((_, i) => {
      const val = Math.sin(i * 0.5 + seed * 0.3 + actionId * 0.7) * 0.5 + 0.5;
      return Math.max(0.1, val);
    });
  };

  const attackId = ATTACKS.findIndex(a => a.name === lastAttack?.name);
  const defenseId = DEFENSES.findIndex(d => d.name === lastDefense?.name);

  const layers = [
    { name: "INPUT\n(29)", size: 8, color: COLORS.purple },
    { name: "HIDDEN 1\n(128)", size: 10, color: COLORS.blue },
    { name: "HIDDEN 2\n(128)", size: 10, color: COLORS.green },
    { name: "OUTPUT\n(12)", size: 6, color: COLORS.yellow },
  ];

  const W = 500;
  const H = 380;
  const layerX = (i) => 60 + i * (W - 80) / (layers.length - 1);
  const neuronY = (idx, total) => 30 + (idx + 0.5) * (H - 60) / total;

  const renderNetwork = (actionId, color, label) => (
    <div style={{ flex: 1 }}>
      <div style={{
        color, fontFamily: PIXEL_FONT,
        fontSize: "11px", textAlign: "center",
        marginBottom: "8px", letterSpacing: "3px"
      }}>
        {label}
      </div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`}
        style={{
          background: `${color}08`,
          borderRadius: "8px",
          border: `1px solid ${color}30`
        }}>

        {/* Connections */}
        {layers.slice(0, -1).map((layer, li) => {
          const nextLayer = layers[li + 1];
          const acts = getActivations(actionId, layer.size, li);
          const nextActs = getActivations(actionId, nextLayer.size, li + 1);
          return acts.map((actA, ni) =>
            nextActs.map((actB, nj) => {
              const strength = (actA + actB) / 2;
              return (
                <motion.line
                  key={`${li}-${ni}-${nj}`}
                  x1={layerX(li)} y1={neuronY(ni, layer.size)}
                  x2={layerX(li + 1)} y2={neuronY(nj, nextLayer.size)}
                  stroke={color}
                  strokeWidth={strength * 2}
                  animate={{ opacity: [strength * 0.3, strength * 0.6, strength * 0.3] }}
                  transition={{ duration: 1.5 + Math.random(), repeat: Infinity }}
                />
              );
            })
          );
        })}

        {/* Neurons */}
        {layers.map((layer, li) => {
          const acts = getActivations(actionId, layer.size, li);
          return acts.map((activation, ni) => {
            const x = layerX(li);
            const y = neuronY(ni, layer.size);
            const r = 12;
            return (
              <g key={`${li}-${ni}`}>
                {/* Glow ring */}
                {activation > 0.5 && (
                  <motion.circle
                    cx={x} cy={y} r={r + 6}
                    fill="none"
                    stroke={layer.color}
                    strokeWidth="1"
                    animate={{ opacity: [0.6, 0, 0.6], r: [r + 4, r + 10, r + 4] }}
                    transition={{ duration: 1.2, repeat: Infinity }}
                  />
                )}
                {/* Main neuron */}
                <motion.circle
                  cx={x} cy={y} r={r}
                  fill={layer.color}
                  animate={{
                    opacity: [
                      0.2 + activation * 0.8,
                      0.4 + activation * 0.6,
                      0.2 + activation * 0.8
                    ],
                    r: [r, r + (activation > 0.7 ? 2 : 0), r]
                  }}
                  transition={{ duration: 0.8 + activation * 0.8, repeat: Infinity }}
                  style={{
                    filter: activation > 0.6
                      ? `drop-shadow(0 0 8px ${layer.color})`
                      : "none"
                  }}
                />
                {/* Border */}
                <circle cx={x} cy={y} r={r}
                  fill="none" stroke={layer.color}
                  strokeWidth="1.5" opacity={0.8} />
                {/* Activation value */}
                <text x={x} y={y + 4}
                  textAnchor="middle" fontSize="7"
                  fill="white" fontFamily={PIXEL_FONT}
                  opacity={0.9}>
                  {activation.toFixed(1)}
                </text>
              </g>
            );
          });
        })}

        {/* Layer labels */}
        {layers.map((layer, li) => (
          <text key={li}
            x={layerX(li)} y={H - 8}
            textAnchor="middle" fontSize="8"
            fill={layer.color} fontFamily={PIXEL_FONT}>
            {layer.name.split('\n')[0]}
          </text>
        ))}
        {layers.map((layer, li) => (
          <text key={`sub-${li}`}
            x={layerX(li)} y={H + 4}
            textAnchor="middle" fontSize="7"
            fill={COLORS.gray} fontFamily={PIXEL_FONT}>
            {layer.name.split('\n')[1]}
          </text>
        ))}
      </svg>
    </div>
  );

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
      style={{
        position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
        background: "rgba(0,0,0,0.92)",
        display: "flex", alignItems: "center", justifyContent: "center",
        zIndex: 300, cursor: "pointer",
      }}
    >
      <motion.div
        initial={{ scale: 0.3, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.3, opacity: 0 }}
        transition={{ type: "spring", damping: 20, stiffness: 200 }}
        onClick={e => e.stopPropagation()}
        style={{
          background: COLORS.panel,
          border: `2px solid ${COLORS.border}`,
          boxShadow: `0 0 40px ${COLORS.green}30`,
          padding: "24px",
          borderRadius: "8px",
          width: "90vw",
          maxWidth: "1100px",
          cursor: "default",
        }}
      >
        {/* Header */}
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "20px"
        }}>
          <motion.div
            animate={{ textShadow: [`0 0 8px ${COLORS.green}`, `0 0 20px ${COLORS.green}`, `0 0 8px ${COLORS.green}`] }}
            transition={{ duration: 2, repeat: Infinity }}
            style={{ color: COLORS.green, fontFamily: PIXEL_FONT, fontSize: "14px", letterSpacing: "4px" }}
          >
            ► DQN NEURAL NETWORK — LIVE VIEW
          </motion.div>
          <motion.button
            whileHover={{ scale: 1.1, boxShadow: `0 0 12px ${COLORS.red}` }}
            onClick={onClose}
            style={{
              background: "transparent",
              border: `2px solid ${COLORS.red}`,
              color: COLORS.red, fontFamily: PIXEL_FONT,
              fontSize: "12px", padding: "6px 16px",
              cursor: "pointer", letterSpacing: "2px",
            }}
          >✕ CLOSE</motion.button>
        </div>

        {/* Info bar */}
        <div style={{
          display: "flex", gap: "16px",
          marginBottom: "16px", flexWrap: "wrap"
        }}>
          {[
            { label: "ARCHITECTURE", value: "29 → 128 → 128 → 12", color: COLORS.green },
            { label: "ALGORITHM", value: "Deep Q-Network (DQN)", color: COLORS.purple },
            { label: "OPTIMIZER", value: "Adam (lr=0.0005)", color: COLORS.blue },
            { label: "REPLAY BUFFER", value: "10,000 experiences", color: COLORS.yellow },
          ].map(item => (
            <PixelBorder key={item.label} style={{ padding: "6px 12px", flex: 1 }}>
              <div style={{ color: COLORS.gray, fontFamily: PIXEL_FONT, fontSize: "7px" }}>
                {item.label}
              </div>
              <div style={{ color: item.color, fontFamily: PIXEL_FONT, fontSize: "9px", marginTop: "2px" }}>
                {item.value}
              </div>
            </PixelBorder>
          ))}
        </div>

        {/* Networks side by side */}
        <div style={{ display: "flex", gap: "24px", marginBottom: "16px" }}>
          {renderNetwork(attackId, COLORS.red, "◄ RED TEAM BRAIN (ATTACKER DQN)")}
          {renderNetwork(defenseId, COLORS.blue, "BLUE TEAM BRAIN (DEFENDER DQN) ►")}
        </div>

        {/* Current decisions */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
          <PixelBorder color={COLORS.red} style={{ padding: "12px", textAlign: "center" }}>
            <div style={{ color: COLORS.gray, fontFamily: PIXEL_FONT, fontSize: "8px" }}>
              ATTACKER NETWORK OUTPUT
            </div>
            <div style={{ fontSize: "24px", margin: "6px 0" }}>
              {lastAttack?.emoji || "❓"}
            </div>
            <div style={{ color: COLORS.red, fontFamily: PIXEL_FONT, fontSize: "11px" }}>
              {lastAttack?.name || "WAITING FOR FIRST STEP..."}
            </div>
          </PixelBorder>
          <PixelBorder color={COLORS.blue} style={{ padding: "12px", textAlign: "center" }}>
            <div style={{ color: COLORS.gray, fontFamily: PIXEL_FONT, fontSize: "8px" }}>
              DEFENDER NETWORK OUTPUT
            </div>
            <div style={{ fontSize: "24px", margin: "6px 0" }}>
              {lastDefense?.emoji || "❓"}
            </div>
            <div style={{ color: COLORS.blue, fontFamily: PIXEL_FONT, fontSize: "11px" }}>
              {lastDefense?.name || "WAITING FOR FIRST STEP..."}
            </div>
          </PixelBorder>
        </div>

        <div style={{
          textAlign: "center", marginTop: "12px",
          color: COLORS.gray, fontFamily: PIXEL_FONT, fontSize: "8px"
        }}>
          CLICK ANYWHERE OUTSIDE TO CLOSE
        </div>
      </motion.div>
    </motion.div>
  );
}

function NeuralNetVisualizer({ lastAttack, lastDefense, step }) {
  const [showModal, setShowModal] = useState(false);

  const getActivations = (actionId, layerSize, layer) => {
    if (actionId === null || actionId === undefined) return Array(layerSize).fill(0.3);
    const seed = (actionId * 7 + layer * 13 + step) % 100;
    return Array(layerSize).fill(0).map((_, i) => {
      const val = Math.sin(i * 0.5 + seed * 0.3 + actionId * 0.7) * 0.5 + 0.5;
      return Math.max(0.1, val);
    });
  };

  const attackId = ATTACKS.findIndex(a => a.name === lastAttack?.name);
  const defenseId = DEFENSES.findIndex(d => d.name === lastDefense?.name);

  const layers = [
    { name: "INPUT", size: 5, color: COLORS.purple },
    { name: "H1", size: 6, color: COLORS.blue },
    { name: "H2", size: 6, color: COLORS.green },
    { name: "OUT", size: 4, color: COLORS.yellow },
  ];

  const W = 240;
  const H = 180;
  const layerX = (i) => 25 + i * (W - 35) / (layers.length - 1);
  const neuronY = (idx, total) => 15 + (idx + 0.5) * (H - 30) / total;

  return (
    <>
      <motion.div
        whileHover={{ scale: 1.02, boxShadow: `0 0 20px ${COLORS.green}40` }}
        onClick={() => setShowModal(true)}
        style={{ cursor: "pointer" }}
      >
        <PixelBorder style={{ padding: "10px" }}>
          <div style={{
            display: "flex", justifyContent: "space-between",
            alignItems: "center", marginBottom: "6px"
          }}>
            <div style={{
              color: COLORS.gray, fontFamily: PIXEL_FONT,
              fontSize: "9px", letterSpacing: "2px"
            }}>
              ► DQN NEURAL NETWORK — LIVE
            </div>
            <motion.div
              animate={{ opacity: [1, 0.3, 1] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              style={{ color: COLORS.green, fontFamily: PIXEL_FONT, fontSize: "8px" }}
            >
              🔍 CLICK TO EXPAND
            </motion.div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px" }}>
            {[
              { actionId: attackId, color: COLORS.red, label: "RED BRAIN" },
              { actionId: defenseId, color: COLORS.blue, label: "BLUE BRAIN" },
            ].map(({ actionId, color, label }) => (
              <div key={label}>
                <div style={{
                  color, fontFamily: PIXEL_FONT,
                  fontSize: "7px", textAlign: "center", marginBottom: "3px"
                }}>
                  {label}
                </div>
                <svg width="100%" viewBox={`0 0 ${W} ${H}`}
                  style={{ background: `${color}08`, borderRadius: "4px" }}>
                  {layers.slice(0, -1).map((layer, li) => {
                    const nextLayer = layers[li + 1];
                    const acts = getActivations(actionId, layer.size, li);
                    const nextActs = getActivations(actionId, nextLayer.size, li + 1);
                    return acts.map((actA, ni) =>
                      nextActs.map((actB, nj) => {
                        const strength = (actA + actB) / 2;
                        return (
                          <line key={`${li}-${ni}-${nj}`}
                            x1={layerX(li)} y1={neuronY(ni, layer.size)}
                            x2={layerX(li + 1)} y2={neuronY(nj, nextLayer.size)}
                            stroke={color} strokeWidth={strength * 1.2}
                            opacity={strength * 0.35}
                          />
                        );
                      })
                    );
                  })}
                  {layers.map((layer, li) => {
                    const acts = getActivations(actionId, layer.size, li);
                    return acts.map((activation, ni) => {
                      const x = layerX(li);
                      const y = neuronY(ni, layer.size);
                      return (
                        <motion.circle key={`${li}-${ni}`}
                          cx={x} cy={y} r={7}
                          fill={layer.color}
                          animate={{
                            opacity: [0.2 + activation * 0.8, 0.4 + activation * 0.5, 0.2 + activation * 0.8]
                          }}
                          transition={{ duration: 1 + activation, repeat: Infinity }}
                          style={{
                            filter: activation > 0.6
                              ? `drop-shadow(0 0 4px ${layer.color})`
                              : "none"
                          }}
                        />
                      );
                    });
                  })}
                </svg>
              </div>
            ))}
          </div>

          {/* Mini decision display */}
          <div style={{
            display: "flex", justifyContent: "space-around",
            marginTop: "6px"
          }}>
            <span style={{ color: COLORS.red, fontFamily: PIXEL_FONT, fontSize: "8px" }}>
              {lastAttack?.emoji || "❓"} {lastAttack?.name || "WAITING"}
            </span>
            <span style={{ color: COLORS.blue, fontFamily: PIXEL_FONT, fontSize: "8px" }}>
              {lastDefense?.emoji || "❓"} {lastDefense?.name || "WAITING"}
            </span>
          </div>
        </PixelBorder>
      </motion.div>

      {/* Expanded Modal */}
      <AnimatePresence>
        {showModal && (
          <NeuralNetModal
            onClose={() => setShowModal(false)}
            lastAttack={lastAttack}
            lastDefense={lastDefense}
            step={step}
          />
        )}
      </AnimatePresence>
    </>
  );
}


function RewardChart({ redRewards, blueRewards }) {
  const maxVal = Math.max(...redRewards, ...blueRewards, 100);
  const w = 340, h = 80;
  const pts = (arr) => arr.slice(-50).map((v, i, a) =>
    `${(i / Math.max(a.length - 1, 1)) * w},${h - (v / maxVal) * h}`
  ).join(" ");

  return (
    <PixelBorder style={{ padding: "10px" }}>
      <div style={{ color: COLORS.gray, fontFamily: PIXEL_FONT, fontSize: "9px", marginBottom: "6px", letterSpacing: "2px" }}>
        SCORE HISTORY (LAST 50 STEPS)
      </div>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`}>
        <polyline points={pts(redRewards)} fill="none"
          stroke={COLORS.red} strokeWidth="2" />
        <polyline points={pts(blueRewards)} fill="none"
          stroke={COLORS.blue} strokeWidth="2" />
      </svg>
      <div style={{ display: "flex", gap: "16px", marginTop: "4px" }}>
        <span style={{ color: COLORS.red, fontFamily: PIXEL_FONT, fontSize: "8px" }}>● RED</span>
        <span style={{ color: COLORS.blue, fontFamily: PIXEL_FONT, fontSize: "8px" }}>● BLUE</span>
      </div>
    </PixelBorder>
  );
}

function BattleLog({ logs }) {
  return (
    <PixelBorder style={{ padding: "10px", height: "200px", overflowY: "auto" }}>
      <div style={{ color: COLORS.green, fontFamily: PIXEL_FONT, fontSize: "9px", marginBottom: "6px", letterSpacing: "2px" }}>
        ► BATTLE LOG
      </div>
      <AnimatePresence>
        {logs.map((log, i) => (
          <motion.div key={log}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            style={{
              color: i === 0 ? COLORS.white : COLORS.gray,
              fontFamily: PIXEL_FONT,
              fontSize: "9px",
              padding: "2px 0",
              borderBottom: `1px solid ${COLORS.border}`,
            }}
          >
            {log}
          </motion.div>
        ))}
      </AnimatePresence>
    </PixelBorder>
  );
}

function NodeStatusGrid({ compromised, blocked, honeypots }) {
  return (
    <PixelBorder style={{ padding: "10px" }}>
      <div style={{ color: COLORS.gray, fontFamily: PIXEL_FONT, fontSize: "9px", marginBottom: "8px", letterSpacing: "2px" }}>
        NODE STATUS
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "6px" }}>
        {NODES.map(node => {
          const isComp = compromised[node.id];
          const isBlocked = blocked.includes(node.id);
          const isHoney = honeypots.includes(node.id);
          const color = isComp ? COLORS.red : isBlocked ? COLORS.orange : isHoney ? COLORS.yellow : COLORS.green;
          return (
            <div key={node.id} style={{
              border: `1px solid ${color}`,
              background: `${color}10`,
              padding: "4px",
              borderRadius: "2px",
              textAlign: "center"
            }}>
              <div style={{ fontSize: "12px" }}>{node.emoji}</div>
              <div style={{ color, fontFamily: PIXEL_FONT, fontSize: "7px" }}>{node.id}</div>
              <div style={{ color: COLORS.gray, fontFamily: PIXEL_FONT, fontSize: "7px" }}>
                {isComp ? "💀COMP" : isBlocked ? "🚫BLOCK" : isHoney ? "🍯TRAP" : "✅SAFE"}
              </div>
            </div>
          );
        })}
      </div>
    </PixelBorder>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
// ─── SESSION LOG HELPERS ──────────────────────────────────────────────────────
function generateSessionId() {
  return "SESSION-" + Date.now().toString(36).toUpperCase();
}

function formatTime(ts) {
  return new Date(ts).toLocaleString();
}

function LogViewer({ sessions, onClose }) {
  const [selected, setSelected] = useState(sessions.length > 0 ? sessions.length - 1 : null);
  const session = selected !== null ? sessions[selected] : null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      style={{
        position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
        background: "rgba(0,0,0,0.92)", zIndex: 200,
        display: "flex", flexDirection: "column",
        padding: "24px", fontFamily: PIXEL_FONT,
      }}
    >
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <div style={{ color: COLORS.green, fontSize: "18px", letterSpacing: "4px" }}>
          ► SESSION LOGS
        </div>
        <motion.button
          whileHover={{ scale: 1.05 }}
          onClick={onClose}
          style={{
            background: "transparent", border: `2px solid ${COLORS.red}`,
            color: COLORS.red, fontFamily: PIXEL_FONT,
            fontSize: "12px", padding: "6px 16px", cursor: "pointer",
          }}
        >✕ CLOSE</motion.button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: "16px", flex: 1, overflow: "hidden" }}>

        {/* Session list */}
        <PixelBorder style={{ padding: "10px", overflowY: "auto" }}>
          <div style={{ color: COLORS.gray, fontSize: "9px", letterSpacing: "2px", marginBottom: "8px" }}>
            SAVED SESSIONS ({sessions.length})
          </div>
          {sessions.length === 0 && (
            <div style={{ color: COLORS.gray, fontSize: "10px", padding: "8px" }}>
              No sessions saved yet. Play a game first!
            </div>
          )}
          {sessions.map((s, i) => (
            <motion.div
              key={s.sessionId}
              whileHover={{ scale: 1.02 }}
              onClick={() => setSelected(i)}
              style={{
                border: `1px solid ${selected === i ? COLORS.green : COLORS.border}`,
                background: selected === i ? `${COLORS.green}15` : "transparent",
                padding: "10px", marginBottom: "6px",
                cursor: "pointer", borderRadius: "2px",
              }}
            >
              <div style={{ color: COLORS.green, fontSize: "8px" }}>{s.sessionId}</div>
              <div style={{ color: COLORS.gray, fontSize: "8px", marginTop: "2px" }}>{formatTime(s.startTime)}</div>
              <div style={{ display: "flex", gap: "8px", marginTop: "6px" }}>
                <span style={{ color: COLORS.red, fontSize: "9px" }}>RED: {s.finalRedScore}</span>
                <span style={{ color: COLORS.blue, fontSize: "9px" }}>BLUE: {s.finalBlueScore}</span>
              </div>
              <div style={{
                color: s.winner === "RED" ? COLORS.red : COLORS.blue,
                fontSize: "9px", marginTop: "2px", fontWeight: "bold"
              }}>
                {s.winner} TEAM WON
              </div>
            </motion.div>
          ))}
        </PixelBorder>

        {/* Session detail */}
        {session ? (
          <div style={{ display: "flex", flexDirection: "column", gap: "12px", overflow: "hidden" }}>

            {/* Stats */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "8px" }}>
              {[
                { label: "SESSION ID", value: session.sessionId, color: COLORS.green },
                { label: "TOTAL STEPS", value: session.totalSteps, color: COLORS.yellow },
                { label: "RED SCORE", value: session.finalRedScore, color: COLORS.red },
                { label: "BLUE SCORE", value: session.finalBlueScore, color: COLORS.blue },
              ].map(stat => (
                <PixelBorder key={stat.label} style={{ padding: "10px", textAlign: "center" }}>
                  <div style={{ color: COLORS.gray, fontSize: "8px", letterSpacing: "1px" }}>{stat.label}</div>
                  <div style={{ color: stat.color, fontSize: "18px", marginTop: "4px" }}>{stat.value}</div>
                </PixelBorder>
              ))}
            </div>

            {/* Node final state */}
            <PixelBorder style={{ padding: "10px" }}>
              <div style={{ color: COLORS.gray, fontSize: "9px", letterSpacing: "2px", marginBottom: "8px" }}>
                FINAL NODE STATE
              </div>
              <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                {Object.entries(session.finalNodeState).map(([nodeId, wasComp]) => (
                  <div key={nodeId} style={{
                    border: `1px solid ${wasComp ? COLORS.red : COLORS.green}`,
                    background: wasComp ? `${COLORS.red}15` : `${COLORS.green}15`,
                    padding: "6px 12px", borderRadius: "2px",
                  }}>
                    <span style={{ color: wasComp ? COLORS.red : COLORS.green, fontSize: "10px" }}>
                      {nodeId}: {wasComp ? "💀 COMPROMISED" : "✅ SAFE"}
                    </span>
                  </div>
                ))}
              </div>
            </PixelBorder>

            {/* Full battle log */}
            <PixelBorder style={{ padding: "10px", flex: 1, overflowY: "auto" }}>
              <div style={{ color: COLORS.gray, fontSize: "9px", letterSpacing: "2px", marginBottom: "8px" }}>
                FULL BATTLE LOG ({session.fullLog.length} EVENTS)
              </div>
              {session.fullLog.map((entry, i) => (
                <div key={i} style={{
                  color: i === 0 ? COLORS.white : COLORS.gray,
                  fontSize: "9px", padding: "3px 0",
                  borderBottom: `1px solid ${COLORS.border}`,
                  fontFamily: PIXEL_FONT,
                }}>
                  {entry}
                </div>
              ))}
            </PixelBorder>

            {/* Export button */}
            <motion.button
              whileHover={{ scale: 1.02, boxShadow: `0 0 15px ${COLORS.green}` }}
              onClick={() => {
                const content = [
                  `CYBER ARENA RL - SESSION LOG`,
                  `================================`,
                  `Session ID : ${session.sessionId}`,
                  `Date       : ${formatTime(session.startTime)}`,
                  `Winner     : ${session.winner} TEAM`,
                  `Red Score  : ${session.finalRedScore}`,
                  `Blue Score : ${session.finalBlueScore}`,
                  `Steps      : ${session.totalSteps}`,
                  ``,
                  `FINAL NODE STATE`,
                  `----------------`,
                  ...Object.entries(session.finalNodeState).map(
                    ([k, v]) => `${k}: ${v ? "COMPROMISED" : "SAFE"}`
                  ),
                  ``,
                  `BATTLE LOG`,
                  `----------`,
                  ...session.fullLog,
                ].join("\n");

                const blob = new Blob([content], { type: "text/plain" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `${session.sessionId}.txt`;
                a.click();
                URL.revokeObjectURL(url);
              }}
              style={{
                background: "transparent",
                border: `2px solid ${COLORS.green}`,
                color: COLORS.green, fontFamily: PIXEL_FONT,
                fontSize: "12px", padding: "10px",
                cursor: "pointer", letterSpacing: "2px", width: "100%",
              }}
            >
              💾 EXPORT SESSION LOG AS .TXT
            </motion.button>
          </div>
        ) : (
          <PixelBorder style={{ padding: "20px", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <div style={{ color: COLORS.gray, fontSize: "12px" }}>Select a session to view details</div>
          </PixelBorder>
        )}
      </div>
    </motion.div>
  );
}

export default function App() {
  const [gameState, setGameState] = useState({
    ...initialState,
    sessionId: generateSessionId(),
    startTime: Date.now(),
  });
  const [sessions, setSessions] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("cyberArenaLogs") || "[]");
    } catch { return []; }
  });
  const [showLogs, setShowLogs] = useState(false);
  const [fullLog, setFullLog] = useState([]);
  const intervalRef = useRef(null);
  const [useRealAI, setUseRealAI] = useState(false);
  const [apiConnected, setApiConnected] = useState(false);

const saveSession = useCallback((state, log) => {
  const winner = state.redScore > state.blueScore ? "RED" : "BLUE";

  const session = {
    sessionId: state.sessionId,
    startTime: state.startTime,
    finalRedScore: state.redScore,
    finalBlueScore: state.blueScore,
    totalSteps: state.step,
    winner,
    finalNodeState: state.compromised,
    fullLog: log,
  };

  const updated = [...sessions, session];

  setSessions(updated);

  try {
    localStorage.setItem(
      "cyberArenaLogs",
      JSON.stringify(updated)
    );
  } catch (e) {}
}, [sessions]);

  const startGame = async () => {
    if (useRealAI && apiConnected) {
      await apiCall("/reset", "POST");
    }
    setGameState(prev => ({ ...prev, isRunning: true }));
  };

  const pauseGame = () => {
    setGameState(prev => ({ ...prev, isRunning: false }));
  };

  const resetGame = () => {
    clearInterval(intervalRef.current);
    setFullLog([]);
    setGameState({
      ...initialState,
      sessionId: generateSessionId(),
      startTime: Date.now(),
    });
  };

  // Check API connection on load
  useEffect(() => {
    fetch(`${API_URL}/status`)
      .then(r => r.json())
      .then(() => setApiConnected(true))
      .catch(() => setApiConnected(false));
  }, []);

  useEffect(() => {
    if (gameState.isRunning) {
      intervalRef.current = setInterval(async () => {
        setGameState(prev => {
          if (prev.step >= 200) {
            return { ...prev, isRunning: false };
          }
          return prev;
        });

        if (useRealAI && apiConnected) {
          // Use REAL Python DQN
          const data = await apiCall("/step", "POST");
          if (data) {
            setGameState(prev => {
              if (prev.step >= 200) return { ...prev, isRunning: false };
              const s = data.state;
              const attack = ATTACKS[data.att_action] || null;
              const defense = DEFENSES[data.def_action] || null;
              const newLog = s.battle_log || [];
              const newRedScore = s.attacker_score || prev.redScore;
              const newBlueScore = s.defender_score || prev.blueScore;

              return {
                ...prev,
                compromised: s.compromised,
                attackerPosition: s.attacker_position,
                detectionScore: s.detection_score,
                blocked: s.blocked_nodes || [],
                honeypots: s.honeypots || [],
                idsActive: s.ids_active,
                redScore: newRedScore,
                blueScore: newBlueScore,
                step: data.step,
                lastAttack: attack,
                lastDefense: defense,
                lastEvent: null,
                battleLog: newLog,
                redRewards: [...prev.redRewards, newRedScore],
                blueRewards: [...prev.blueRewards, newBlueScore],
                isRunning: !data.done,
              };
            });
          }
        } else {
          // Use frontend simulation
          setGameState(prev => {
            if (prev.step >= 200) return { ...prev, isRunning: false };
            const next = runSimulationStep(prev);
            if (next.battleLog[0] !== prev.battleLog[0]) {
              setFullLog(fl => [next.battleLog[0], ...fl]);
            }
            return next;
          });
        }
      }, gameState.speed);
    } else {
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [gameState.isRunning, gameState.speed, useRealAI, apiConnected]);

// Auto-save when game ends
useEffect(() => {
  if (!gameState.isRunning && gameState.step >= 200) {
    saveSession(gameState, fullLog);
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, [gameState.isRunning, gameState.step]);

  return (
    <div style={{
      background: COLORS.bg,
      minHeight: "100vh",
      padding: "16px",
      fontFamily: PIXEL_FONT,
      color: COLORS.white,
    }}>

      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: "16px" }}>
        <motion.h1
          animate={{ textShadow: [`0 0 10px ${COLORS.green}`, `0 0 30px ${COLORS.green}`, `0 0 10px ${COLORS.green}`] }}
          transition={{ duration: 2, repeat: Infinity }}
          style={{ color: COLORS.green, fontFamily: PIXEL_FONT, fontSize: "24px",
            letterSpacing: "6px", margin: 0 }}
        >
          ◄◄ CYBER ARENA RL ►►
        </motion.h1>
        <div style={{ color: COLORS.gray, fontSize: "10px", letterSpacing: "3px", marginTop: "4px" }}>
          MULTI-AGENT ADVERSARIAL REINFORCEMENT LEARNING
        </div>
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: "8px", justifyContent: "center", marginBottom: "16px", flexWrap: "wrap" }}>
        {[
          { label: gameState.isRunning ? "⏸ PAUSE" : "▶ START", action: gameState.isRunning ? pauseGame : startGame, color: gameState.isRunning ? COLORS.yellow : COLORS.green },
          { label: "↺ RESET", action: resetGame, color: COLORS.orange },
          { label: `📋 LOGS (${sessions.length})`, action: () => setShowLogs(true), color: COLORS.purple },
        ].map(btn => (
          <motion.button key={btn.label}
            whileHover={{ scale: 1.05, boxShadow: `0 0 15px ${btn.color}` }}
            whileTap={{ scale: 0.95 }}
            onClick={btn.action}
            style={{
              background: "transparent",
              border: `2px solid ${btn.color}`,
              color: btn.color,
              fontFamily: PIXEL_FONT,
              fontSize: "12px",
              padding: "8px 20px",
              cursor: "pointer",
              letterSpacing: "2px",
            }}
          >{btn.label}</motion.button>
        ))}

        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ color: COLORS.gray, fontSize: "10px" }}>SPEED:</span>
          {[800, 500, 200, 50].map(spd => (
            <motion.button key={spd}
              whileHover={{ scale: 1.05 }}
              onClick={() => setGameState(prev => ({ ...prev, speed: spd }))}
              style={{
                background: gameState.speed === spd ? COLORS.purple : "transparent",
                border: `1px solid ${COLORS.purple}`,
                color: gameState.speed === spd ? COLORS.white : COLORS.purple,
                fontFamily: PIXEL_FONT, fontSize: "9px",
                padding: "4px 8px", cursor: "pointer",
              }}
            >{spd === 50 ? "TURBO" : spd === 200 ? "FAST" : spd === 500 ? "NORMAL" : "SLOW"}</motion.button>
          ))}
        </div>

        {/* AI Mode Toggle */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ color: COLORS.gray, fontSize: "10px", fontFamily: PIXEL_FONT }}>
            AI MODE:
          </span>
          <motion.button
            whileHover={{ scale: 1.05 }}
            onClick={() => setUseRealAI(false)}
            style={{
              background: !useRealAI ? COLORS.green : "transparent",
              border: `1px solid ${COLORS.green}`,
              color: !useRealAI ? COLORS.bg : COLORS.green,
              fontFamily: PIXEL_FONT, fontSize: "9px",
              padding: "4px 10px", cursor: "pointer",
            }}
          >SIM</motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            onClick={() => setUseRealAI(true)}
            style={{
              background: useRealAI ? COLORS.purple : "transparent",
              border: `1px solid ${apiConnected ? COLORS.purple : COLORS.gray}`,
              color: useRealAI ? COLORS.white : apiConnected ? COLORS.purple : COLORS.gray,
              fontFamily: PIXEL_FONT, fontSize: "9px",
              padding: "4px 10px", cursor: "pointer",
            }}
          >🧠 REAL DQN {apiConnected ? "●" : "○"}</motion.button>
        </div>

      </div>

      {/* Session ID display */}
      <div style={{ textAlign: "center", marginBottom: "12px" }}>
        <span style={{ color: COLORS.gray, fontFamily: PIXEL_FONT, fontSize: "9px", letterSpacing: "2px" }}>
          CURRENT SESSION: </span>
        <span style={{ color: COLORS.green, fontFamily: PIXEL_FONT, fontSize: "9px" }}>
          {gameState.sessionId}
        </span>
      </div>

      {/* Scoreboard */}
      <Scoreboard
        redScore={gameState.redScore}
        blueScore={gameState.blueScore}
        step={gameState.step}
        detectionScore={gameState.detectionScore}
        idsActive={gameState.idsActive}
      />

      {/* Main Layout */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: "12px" }}>

        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          <PixelBorder style={{ padding: "10px", height: "380px" }}>
            <div style={{ color: COLORS.gray, fontFamily: PIXEL_FONT, fontSize: "9px", marginBottom: "6px", letterSpacing: "2px" }}>
              ► NETWORK MAP — ATTACKER @ {gameState.attackerPosition}
            </div>
            <NetworkGraph gameState={gameState} />
          </PixelBorder>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
            <PixelBorder color={COLORS.red} style={{ padding: "10px" }}>
              <div style={{ color: COLORS.red, fontFamily: PIXEL_FONT, fontSize: "9px", marginBottom: "8px", letterSpacing: "2px" }}>
                ◄ RED TEAM LAST ACTION
              </div>
              <div style={{ display: "flex", justifyContent: "center" }}>
                <ActionCard data={gameState.lastAttack} type="attack" isActive={true} />
              </div>
            </PixelBorder>

            <PixelBorder color={COLORS.blue} style={{ padding: "10px" }}>
              <div style={{ color: COLORS.blue, fontFamily: PIXEL_FONT, fontSize: "9px", marginBottom: "8px", letterSpacing: "2px" }}>
                BLUE TEAM LAST ACTION ►
              </div>
              <div style={{ display: "flex", justifyContent: "center" }}>
                <ActionCard data={gameState.lastDefense} type="defense" isActive={true} />
              </div>
            </PixelBorder>
          </div>

          <PixelBorder style={{ padding: "10px" }}>
            <div style={{ color: COLORS.gray, fontFamily: PIXEL_FONT, fontSize: "9px", marginBottom: "8px", letterSpacing: "2px" }}>
              ◄ RED TEAM ARSENAL
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
              {ATTACKS.map(atk => (
                <ActionCard key={atk.id} data={atk} type="attack"
                  isActive={gameState.lastAttack?.name === atk.name} />
              ))}
            </div>
          </PixelBorder>

          <PixelBorder style={{ padding: "10px" }}>
            <div style={{ color: COLORS.gray, fontFamily: PIXEL_FONT, fontSize: "9px", marginBottom: "8px", letterSpacing: "2px" }}>
              BLUE TEAM ARSENAL ►
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
              {DEFENSES.map(def => (
                <ActionCard key={def.id} data={def} type="defense"
                  isActive={gameState.lastDefense?.name === def.name} />
              ))}
            </div>
          </PixelBorder>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          <NodeStatusGrid
            compromised={gameState.compromised}
            blocked={gameState.blocked}
            honeypots={gameState.honeypots}
          />
          <NeuralNetVisualizer
            lastAttack={gameState.lastAttack}
            lastDefense={gameState.lastDefense}
            step={gameState.step}
          />
          <RewardChart
            redRewards={gameState.redRewards}
            blueRewards={gameState.blueRewards}
          />
          <BattleLog logs={gameState.battleLog} />
        </div>
      </div>

      {/* Game Over */}
      <AnimatePresence>
        {gameState.step >= 200 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            style={{
              position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
              background: "rgba(0,0,0,0.85)",
              display: "flex", alignItems: "center", justifyContent: "center",
              zIndex: 100,
            }}
          >
            <PixelBorder color={gameState.redScore > gameState.blueScore ? COLORS.red : COLORS.blue}
              style={{ padding: "40px", textAlign: "center" }}>
              <div style={{ fontSize: "48px", marginBottom: "16px" }}>
                {gameState.redScore > gameState.blueScore ? "💀" : "🛡️"}
              </div>
              <div style={{
                color: gameState.redScore > gameState.blueScore ? COLORS.red : COLORS.blue,
                fontFamily: PIXEL_FONT, fontSize: "24px", letterSpacing: "4px"
              }}>
                {gameState.redScore > gameState.blueScore ? "RED TEAM WINS" : "BLUE TEAM WINS"}
              </div>
              <div style={{ color: COLORS.gray, fontFamily: PIXEL_FONT, fontSize: "12px", margin: "16px 0" }}>
                RED: {gameState.redScore} pts | BLUE: {gameState.blueScore} pts
              </div>
              <div style={{ color: COLORS.green, fontFamily: PIXEL_FONT, fontSize: "10px", marginBottom: "16px" }}>
                ✅ Session auto-saved to logs
              </div>
              <div style={{ display: "flex", gap: "12px", justifyContent: "center" }}>
                <motion.button whileHover={{ scale: 1.05 }} onClick={resetGame}
                  style={{
                    background: "transparent", border: `2px solid ${COLORS.green}`,
                    color: COLORS.green, fontFamily: PIXEL_FONT,
                    fontSize: "14px", padding: "10px 24px",
                    cursor: "pointer", letterSpacing: "2px",
                  }}
                >▶ PLAY AGAIN</motion.button>
                <motion.button whileHover={{ scale: 1.05 }} onClick={() => { resetGame(); setShowLogs(true); }}
                  style={{
                    background: "transparent", border: `2px solid ${COLORS.purple}`,
                    color: COLORS.purple, fontFamily: PIXEL_FONT,
                    fontSize: "14px", padding: "10px 24px",
                    cursor: "pointer", letterSpacing: "2px",
                  }}
                >📋 VIEW LOGS</motion.button>
              </div>
            </PixelBorder>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Footer */}
      <div style={{
        marginTop: "32px",
        borderTop: `1px solid ${COLORS.border}`,
        paddingTop: "16px",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        gap: "24px",
        flexWrap: "wrap",
      }}>
        <div style={{ color: COLORS.gray, fontFamily: PIXEL_FONT, fontSize: "9px", letterSpacing: "2px" }}>
          BUILT BY VANSH KUMAR
        </div>

        <motion.a
          href="https://github.com/vansh-kumar-007"
          target="_blank"
          rel="noopener noreferrer"
          whileHover={{ scale: 1.05, boxShadow: `0 0 12px ${COLORS.white}` }}
          style={{
            display: "flex", alignItems: "center", gap: "8px",
            border: `1px solid ${COLORS.border}`,
            background: COLORS.panel,
            padding: "8px 16px",
            textDecoration: "none",
            borderRadius: "2px",
          }}
        >
          <span style={{ fontSize: "16px" }}>🐙</span>
          <span style={{ color: COLORS.white, fontFamily: PIXEL_FONT, fontSize: "10px", letterSpacing: "2px" }}>
            GITHUB
          </span>
        </motion.a>

        <motion.a
          href="https://www.linkedin.com/in/vanshkumar007/"
          target="_blank"
          rel="noopener noreferrer"
          whileHover={{ scale: 1.05, boxShadow: `0 0 12px #0077b5` }}
          style={{
            display: "flex", alignItems: "center", gap: "8px",
            border: `1px solid #0077b5`,
            background: COLORS.panel,
            padding: "8px 16px",
            textDecoration: "none",
            borderRadius: "2px",
          }}
        >
          <span style={{ fontSize: "16px" }}>💼</span>
          <span style={{ color: "#0077b5", fontFamily: PIXEL_FONT, fontSize: "10px", letterSpacing: "2px" }}>
            LINKEDIN
          </span>
        </motion.a>

        <div style={{ color: COLORS.gray, fontFamily: PIXEL_FONT, fontSize: "9px", letterSpacing: "1px" }}>
          CYBER ARENA RL — MULTI-AGENT RL SIMULATION
        </div>
      </div>
      {/* Log Viewer */}
      <AnimatePresence>
        {showLogs && (
          <LogViewer sessions={sessions} onClose={() => setShowLogs(false)} />
        )}
      </AnimatePresence>
    </div>
  );
}