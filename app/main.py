"""
InboxPilot — FastAPI server.

Exposes the InboxPilot environment over HTTP for HF Spaces deployment.

Endpoints:
  GET  /         — Landing page / metadata
  GET  /health   — Health check
  GET  /tasks    — List available tasks
  GET  /state    — Current environment state
  POST /reset    — Reset environment (optionally with task_id)
  POST /step     — Execute an action
"""

from __future__ import annotations

import logging
import traceback

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Any, Optional

from .env import InboxPilotEnv
from .models import Action
from .tasks import list_tasks

logger = logging.getLogger("inboxpilot")

# ---------------------------------------------------------------------------
# App & global env instance
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Auto-reset environment on startup so /state and /step always work."""
    try:
        env.reset()
        logger.info("Environment auto-reset on startup.")
    except Exception as e:
        logger.warning(f"Auto-reset failed (non-fatal): {e}")
    yield


app = FastAPI(
    title="InboxPilot — OpenEnv",
    description="Professional email triage and operations environment for AI agents.",
    version="1.0.0",
    lifespan=lifespan,
)

env = InboxPilotEnv()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    action_type: str = "finish"
    email_id: Optional[str] = None
    payload: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def landing():
    """Minimal HTML landing page."""
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>InboxPilot — OpenEnv</title>
    <style>
      body { font-family: system-ui, sans-serif; max-width: 720px; margin: 60px auto; padding: 0 20px; color: #e0e0e0; background: #0d1117; }
      h1 { color: #58a6ff; } h2 { color: #79c0ff; }
      a { color: #58a6ff; } code { background: #161b22; padding: 2px 6px; border-radius: 4px; }
      .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin: 16px 0; }
    </style>
    </head>
    <body>
      <h1>📬 InboxPilot — OpenEnv</h1>
      <p>Professional email triage and operations environment for AI agents.</p>
      <div class="card">
        <h2>API Endpoints</h2>
        <ul>
          <li><code>POST /reset</code> — Reset environment</li>
          <li><code>POST /step</code> — Execute an action</li>
          <li><code>GET /state</code> — Current state</li>
          <li><code>GET /tasks</code> — List tasks</li>
          <li><code>GET /health</code> — Health check</li>
          <li><code>GET /docs</code> — Interactive API docs</li>
        </ul>
      </div>
      <div class="card">
        <h2>Tasks</h2>
        <ul>
          <li><strong>Easy:</strong> Support Inbox Triage (6 emails)</li>
          <li><strong>Medium:</strong> Customer Resolution Workflow (1 complex case)</li>
          <li><strong>Hard:</strong> Executive Inbox Risk Management (7 high-stakes emails)</li>
        </ul>
      </div>
      <p>Built for the <strong>Meta × Scaler Hackathon</strong> · <a href="/docs">API Docs →</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "environment": "InboxPilot", "version": "1.0.0"}


@app.get("/tasks")
def tasks():
    """List all available tasks."""
    return {"tasks": list_tasks()}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    """Reset the environment to the start of a task."""
    try:
        result = env.reset(task_id=req.task_id or "easy_support_triage")
        return result.model_dump()
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"/reset error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.post("/step")
def step(req: StepRequest):
    """Execute an action in the environment."""
    try:
        action = Action(
            action_type=req.action_type or "finish",
            email_id=req.email_id,
            payload=req.payload if isinstance(req.payload, dict) else {},
        )
        result = env.step(action)
        return result.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"/step error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Step failed: {e}")


@app.get("/state")
def state():
    """Return the current internal environment state."""
    try:
        return env.state().model_dump()
    except Exception as e:
        logger.error(f"/state error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"State retrieval failed: {e}")
