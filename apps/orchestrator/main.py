"""Orchestrator service main application."""

from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from .orchestrator import orchestrator

app = FastAPI(
    title="PrometheusULTIMATE v4 Orchestrator",
    description="Task lifecycle management service",
    version="0.9.0"
)


class TaskCreateRequest(BaseModel):
    """Task creation request."""
    goal: str
    inputs: Dict[str, Any] = {}
    limits: Dict[str, Any] = {}
    project_id: str = "default"


class TaskResponse(BaseModel):
    """Task response."""
    task_id: str
    status: str


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    healthy = await orchestrator.health_check()
    return {
        "status": "healthy" if healthy else "unhealthy",
        "service": "orchestrator"
    }


@app.post("/task", response_model=TaskResponse)
async def create_task(request: TaskCreateRequest):
    """Create a new task."""
    try:
        task_id = await orchestrator.create_task(
            goal=request.goal,
            inputs=request.inputs,
            limits=request.limits,
            project_id=request.project_id
        )
        
        return TaskResponse(
            task_id=task_id,
            status="pending"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/task/{task_id}")
async def get_task(task_id: str):
    """Get task status and details."""
    task_status = await orchestrator.get_task_status(task_id)
    
    if not task_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    return task_status


@app.post("/task/{task_id}/abort")
async def abort_task(task_id: str):
    """Abort a running task."""
    success = await orchestrator.abort_task(task_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found or not running"
        )
    
    return {"status": "aborted", "task_id": task_id}


@app.get("/tasks")
async def list_tasks(project_id: str = "default", limit: int = 100):
    """List tasks for a project."""
    # TODO: Implement task listing
    return {"tasks": [], "count": 0, "project_id": project_id}


if __name__ == "__main__":
    import uvicorn
    from libs.common.config import get_settings
    
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.orchestrator_port,
        reload=True
    )
