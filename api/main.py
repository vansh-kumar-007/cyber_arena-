# api/main.py
# FastAPI server — exposes the RL simulation as REST API endpoints
# Run with: uvicorn api.main:app --reload --port 8000

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from api.simulation import SimulationManager

# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CyberArena RL API",
    description="Multi-Agent Adversarial RL Simulation API",
    version="1.0.0"
)

# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize simulation (loads trained models)
sim = SimulationManager()

# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "CyberArena RL API is running",
        "endpoints": ["/reset", "/step", "/weights", "/status"]
    }

@app.post("/reset")
def reset_simulation():
    """Reset environment for a new episode"""
    result = sim.reset()
    return {"success": True, "data": result}

@app.post("/step")
def step_simulation():
    """Run one step — both agents act, environment updates"""
    result = sim.step()
    return {"success": True, "data": result}

@app.get("/weights")
def get_weights():
    """Get neural network weights for visualization"""
    weights = sim.get_network_weights()
    return {"success": True, "data": weights}

@app.get("/status")
def get_status():
    """Get current simulation status"""
    return {
        "success": True,
        "data": {
            "episode": sim.episode_count,
            "step": sim.env.current_step,
            "is_done": sim.is_done,
            "att_epsilon": round(sim.attacker.epsilon, 4),
            "def_epsilon": round(sim.defender.epsilon, 4),
            "att_memory": len(sim.attacker.memory),
            "def_memory": len(sim.defender.memory),
        }
    }

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)