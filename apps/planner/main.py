"""Planner service main application."""

from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from .planner import planner


app = FastAPI(
    title="PrometheusULTIMATE v4 Planner",
    description="Cost/time-aware task planning service",
    version="0.9.0"
)


class PlanRequest(BaseModel):
    """Plan creation request."""
    goal: str
    inputs: Dict[str, Any] = {}
    limits: Dict[str, Any] = {}
    project_id: str = "default"
    policy: str = "auto"


class PlanResponse(BaseModel):
    """Plan response."""
    plan_id: str
    steps: list
    budget: Dict[str, Any]
    estimated_duration_ms: int


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    healthy = await planner.health_check()
    return {
        "status": "healthy" if healthy else "unhealthy",
        "service": "planner"
    }


@app.post("/plan", response_model=PlanResponse)
async def create_plan(request: PlanRequest):
    """Create execution plan for a task."""
    try:
        plan = await planner.create_plan(
            goal=request.goal,
            inputs=request.inputs,
            limits=request.limits,
            project_id=request.project_id,
            policy=request.policy
        )
        
        return PlanResponse(
            plan_id=plan["id"],
            steps=plan["steps"],
            budget=plan["budget"],
            estimated_duration_ms=plan["estimated_duration_ms"]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/skills")
async def get_available_skills():
    """Get available skills and their metrics."""
    return {
        "skills": [
            {
                "name": skill_name,
                "avg_latency_ms": metrics.avg_latency_ms,
                "avg_cost_usd": metrics.avg_cost_usd,
                "success_rate": metrics.success_rate,
                "error_rate": metrics.error_rate
            }
            for skill_name, metrics in planner.skill_metrics.items()
        ]
    }


@app.post("/skills/{skill_name}/metrics")
async def update_skill_metrics(
    skill_name: str,
    latency_ms: int,
    cost_usd: float,
    success: bool
):
    """Update skill performance metrics."""
    try:
        planner.update_skill_metrics(skill_name, latency_ms, cost_usd, success)
        return {"status": "updated", "skill_name": skill_name}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/policies")
async def get_policies():
    """Get available planning policies."""
    return {
        "policies": [
            {
                "name": "tiny",
                "description": "Fast, low-cost execution using RadonSAI-Small",
                "model": "radon/small-0.1b"
            },
            {
                "name": "base",
                "description": "Balanced execution using RadonSAI",
                "model": "radon/base-0.8b"
            },
            {
                "name": "balanced",
                "description": "Main working instructor using RadonSAI-Balanced",
                "model": "radon/balanced-3b"
            },
            {
                "name": "ultra",
                "description": "High-quality execution using RadonSAI-Ultra",
                "model": "radon/ultra-13b"
            },
            {
                "name": "mega",
                "description": "Maximum quality using RadonSAI-Mega",
                "model": "radon/mega-70b"
            },
            {
                "name": "auto",
                "description": "Automatic policy selection based on limits",
                "model": "auto"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    from libs.common.config import get_settings
    
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.planner_port,
        reload=True
    )
