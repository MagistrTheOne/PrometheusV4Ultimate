"""Gateway service main application."""

import json
import os
from pathlib import Path
from typing import Dict, List

import httpx
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from libs.common.config import get_settings
from libs.common.schemas import ModelInfo, ModelStatus, Task, TaskStatus

app = FastAPI(
    title="PrometheusULTIMATE v4 Gateway",
    description="HTTP API Gateway for AGI Platform",
    version="0.9.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()


class TaskCreateRequest(BaseModel):
    """Task creation request."""
    goal: str
    inputs: Dict = {}
    limits: Dict = {}
    project_id: str = "default"


class TaskResponse(BaseModel):
    """Task response."""
    task_id: str
    status: str


class MemorySaveRequest(BaseModel):
    """Memory save request."""
    project_id: str
    kind: str
    payload: Dict


class MemorySearchRequest(BaseModel):
    """Memory search request."""
    q: str
    project_id: str = "default"
    top_k: int = 5
    filters: Dict = {}


class SkillRegisterRequest(BaseModel):
    """Skill registration request."""
    name: str
    spec: str
    permissions: Dict


class SkillRunRequest(BaseModel):
    """Skill execution request."""
    name: str
    args: Dict


class FeedbackRequest(BaseModel):
    """Feedback request."""
    task_id: str
    rating: int
    comment: str = ""


def load_model_registry() -> List[ModelInfo]:
    """Load model registry from filesystem."""
    models = []
    registry_path = Path(settings.model_registry_path)
    
    if not registry_path.exists():
        return models
    
    for model_dir in registry_path.rglob("model.json"):
        try:
            with open(model_dir, "r", encoding="utf-8") as f:
                model_data = json.load(f)
                models.append(ModelInfo(**model_data))
        except Exception as e:
            print(f"Error loading model {model_dir}: {e}")
    
    return models


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "gateway"}


@app.get("/models")
async def get_models():
    """Get available models."""
    models = load_model_registry()
    return {
        "models": [model.dict() for model in models],
        "count": len(models)
    }


@app.post("/task", response_model=TaskResponse)
async def create_task(request: TaskCreateRequest):
    """Create a new task."""
    # TODO: Forward to orchestrator service
    # For now, return a mock response
    task_id = "t_mock_001"
    
    return TaskResponse(
        task_id=task_id,
        status="pending"
    )


@app.get("/task/{task_id}")
async def get_task(task_id: str):
    """Get task status and details."""
    # TODO: Forward to orchestrator service
    # For now, return a mock response
    return {
        "task_id": task_id,
        "status": "running",
        "progress": 0.5,
        "artifacts": [],
        "cost": {"usd": 0.0},
        "decisions": []
    }


@app.post("/memory/save")
async def save_memory(request: MemorySaveRequest):
    """Save item to memory."""
    # TODO: Forward to memory service
    return {"status": "saved", "id": "mem_001"}


@app.get("/memory/search")
async def search_memory(q: str, project_id: str = "default", top_k: int = 5):
    """Search memory."""
    # TODO: Forward to memory service
    return {
        "results": [],
        "count": 0,
        "query": q
    }


@app.post("/skill/register")
async def register_skill(request: SkillRegisterRequest):
    """Register a new skill."""
    # TODO: Forward to skills service
    return {
        "status": "registered",
        "name": request.name,
        "stage": "auto_generation"
    }


@app.post("/skill/run")
async def run_skill(request: SkillRunRequest):
    """Execute a skill."""
    # TODO: Forward to skills service
    return {
        "status": "completed",
        "result": f"[STUB] Skill {request.name} executed with args {request.args}"
    }


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback."""
    # TODO: Store feedback for learning/prioritization
    return {"status": "received", "task_id": request.task_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.gateway_port,
        reload=True
    )
