"""HTTP API for CyberArena's simulated multi-agent RL environment."""
from __future__ import annotations

import logging
import os
import secrets
import threading
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.simulation import SimulationManager

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CyberArena RL API",
    description="Educational, simulated multi-agent adversarial reinforcement learning environment",
    version="1.1.0",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CYBERARENA_CORS_ORIGINS",
        "https://cyber-arena-delta.vercel.app,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Admin-Token"],
)

# One simulation per worker. Use one Uvicorn worker for a single coherent live game.
sim = SimulationManager()
simulation_lock = threading.RLock()


class ResetRequest(BaseModel):
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    n_attackers: int | None = Field(default=None, ge=1, le=4)
    n_defenders: int | None = Field(default=None, ge=1, le=4)


class MemoryLessonUpdate(BaseModel):
    lesson: str | None = Field(default=None, min_length=1, max_length=4000)
    lesson_status: Literal["active", "deprecated"] | None = None


def require_admin(token: str | None) -> None:
    """Protect memory mutation endpoints; never ship this token in browser code."""
    expected = os.getenv("CYBERARENA_ADMIN_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Memory administration is disabled; configure CYBERARENA_ADMIN_TOKEN on the server.",
        )
    if token is None or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=403, detail="Valid X-Admin-Token required.")


@app.get("/")
def root():
    return {
        "message": "CyberArena RL API is running",
        "mode": "simulated educational environment",
        "endpoints": ["/health", "/reset", "/step", "/weights", "/status", "/memory", "/memory/search"],
    }


@app.get("/health")
def health():
    with simulation_lock:
        status = sim.status()
    return {
        "status": "ok",
        "models_loaded": status["models_loaded"],
        "state_size": status["state_size"],
        "n_attackers": status["n_attackers"],
        "n_defenders": status["n_defenders"],
        "memory_available": status["memory_write_errors"] == 0,
    }


@app.post("/reset")
def reset_simulation(request: ResetRequest | None = None):
    """Start a seeded episode and optionally select the requested attacker/defender counts."""
    with simulation_lock:
        if request is None:
            result = sim.reset()
        else:
            result = sim.reset(
                seed=request.seed,
                n_attackers=request.n_attackers,
                n_defenders=request.n_defenders,
            )
    return {"success": True, "data": result}


@app.post("/step")
def step_simulation():
    """Run one DQN decision step and persist its observed outcomes."""
    with simulation_lock:
        result = sim.step()
    return {"success": True, "data": result}


@app.get("/weights")
def get_weights():
    """Get model weights for the technical visualizer."""
    with simulation_lock:
        weights = sim.get_network_weights()
    return {"success": True, "data": weights}


@app.get("/status")
def get_status():
    with simulation_lock:
        status = sim.status()
    return {"success": True, "data": status}


@app.get("/memory")
def get_memory(
    limit: int = Query(default=50, ge=1, le=500),
    agent: Literal["attacker", "defender"] | None = None,
):
    """List saved experience records for review; read-only without admin credentials."""
    entries = sim.memory.list(limit=limit, agent=agent)
    return {"success": True, "data": {"summary": sim.memory.summary(), "experiences": entries}}


@app.get("/memory/search")
def search_memory(
    q: str = Query(min_length=1, max_length=500),
    limit: int = Query(default=5, ge=1, le=50),
    agent: Literal["attacker", "defender"] | None = None,
):
    """Retrieve lexical precedents. This is not an embedding-based semantic search."""
    try:
        matches = sim.memory.retrieve(q, agent=agent, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"success": True, "data": {"query": q, "matches": matches}}


@app.patch("/memory/{experience_id}")
def update_memory_lesson(
    experience_id: str,
    request: MemoryLessonUpdate,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
):
    require_admin(x_admin_token)
    try:
        updated = sim.memory.update_lesson(
            experience_id, lesson=request.lesson, lesson_status=request.lesson_status
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Experience not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"success": True, "data": updated}


@app.delete("/memory/{experience_id}")
def delete_memory(
    experience_id: str,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
):
    require_admin(x_admin_token)
    if not sim.memory.delete(experience_id):
        raise HTTPException(status_code=404, detail="Experience not found")
    return {"success": True, "data": {"deleted_id": experience_id}}


@app.delete("/memory")
def reset_memory(
    confirm: bool = Query(default=False),
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
):
    require_admin(x_admin_token)
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass confirm=true to delete all experience records.")
    deleted = sim.memory.reset()
    logger.warning("Agent experience memory reset; deleted %s records", deleted)
    return {"success": True, "data": {"deleted_count": deleted}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
