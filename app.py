"""
FastAPI application for IncidentCommanderEnv.
Exposes /reset, /step, /state, /grade endpoints.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from env import IncidentCommanderEnv, list_scenarios


app = FastAPI(
    title="IncidentCommanderEnv",
    description="IT Incident Response Simulation Environment",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage
sessions: Dict[str, IncidentCommanderEnv] = {}


def get_env(session_id: str = "default") -> IncidentCommanderEnv:
    """Get or create environment for session."""
    if session_id not in sessions:
        sessions[session_id] = IncidentCommanderEnv(seed=42)
    return sessions[session_id]


class ResetRequest(BaseModel):
    task_id: str = Field(default="easy_single_failure")
    session_id: str = Field(default="default")


class StepRequest(BaseModel):
    action_type: str
    target_service: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    session_id: str = Field(default="default")


class SessionRequest(BaseModel):
    session_id: str = Field(default="default")


@app.get("/")
def root():
    """API info."""
    return {
        "name": "IncidentCommanderEnv",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/grade", "/tasks", "/actions"]
    }


@app.get("/health")
def health():
    """Health check."""
    return {"status": "healthy"}


@app.get("/tasks")
def get_tasks():
    """List available tasks."""
    return {"tasks": list_scenarios()}


@app.get("/actions")
def get_actions():
    """Get action space."""
    env = get_env()
    return env.get_action_space()


@app.post("/reset")
def reset(request: ResetRequest):
    """Reset environment."""
    try:
        env = get_env(request.session_id)
        obs = env.reset(request.task_id)
        return {"success": True, "observation": obs}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/step")
def step(request: StepRequest):
    """Take action."""
    try:
        env = get_env(request.session_id)
        action = {
            "action_type": request.action_type,
            "target_service": request.target_service,
            "parameters": request.parameters or {}
        }
        result = env.step(action)
        return {"success": True, **result}
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/state")
def state(request: SessionRequest):
    """Get current state."""
    env = get_env(request.session_id)
    s = env.state()
    if "error" in s:
        raise HTTPException(400, s["error"])
    return {"success": True, "observation": s}


@app.post("/grade")
def grade(request: SessionRequest):
    """Grade trajectory."""
    env = get_env(request.session_id)
    g = env.grade()
    if "error" in g:
        raise HTTPException(400, g["error"])
    return {"success": True, "grade": g}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)