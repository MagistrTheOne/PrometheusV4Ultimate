"""Critic service main application."""

from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from .critic import critic


app = FastAPI(
    title="PrometheusULTIMATE v4 Critic",
    description="Validation and quality control service",
    version="0.9.0"
)


class ReviewRequest(BaseModel):
    """Review request."""
    task_id: str
    content: str
    context: Dict[str, Any] = {}
    project_id: str = "default"


class ReviewResponse(BaseModel):
    """Review response."""
    review_id: str
    overall_passed: bool
    checks: Dict[str, Any]
    issues: List[str]
    suggestions: List[str]
    confidence: float


class FactCheckRequest(BaseModel):
    """Fact check request."""
    content: str
    project_id: str = "default"


class NumberCheckRequest(BaseModel):
    """Number check request."""
    content: str
    context: Dict[str, Any] = {}


class CodeCheckRequest(BaseModel):
    """Code check request."""
    code: str
    language: str = "python"


class PolicyCheckRequest(BaseModel):
    """Policy check request."""
    content: str
    policies: List[str] = []


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    healthy = await critic.health_check()
    return {
        "status": "healthy" if healthy else "unhealthy",
        "service": "critic"
    }


@app.post("/review", response_model=ReviewResponse)
async def review_task(request: ReviewRequest):
    """Comprehensive task review."""
    try:
        result = await critic.review_task(
            task_id=request.task_id,
            content=request.content,
            context=request.context,
            project_id=request.project_id
        )
        
        return ReviewResponse(
            review_id=result["review_id"],
            overall_passed=result["overall_passed"],
            checks=result["checks"],
            issues=result["issues"],
            suggestions=result["suggestions"],
            confidence=result["confidence"]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/check/facts")
async def check_facts(request: FactCheckRequest):
    """Check facts against Memory."""
    try:
        result = await critic.fact_checker.check_facts(
            content=request.content,
            project_id=request.project_id
        )
        
        return {
            "passed": result.passed,
            "issues": result.issues,
            "suggestions": result.suggestions,
            "confidence": result.confidence
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/check/numbers")
async def check_numbers(request: NumberCheckRequest):
    """Check numbers for accuracy."""
    try:
        result = await critic.number_checker.check_numbers(
            content=request.content,
            context=request.context
        )
        
        return {
            "passed": result.passed,
            "issues": result.issues,
            "suggestions": result.suggestions,
            "confidence": result.confidence
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/check/code")
async def check_code(request: CodeCheckRequest):
    """Check code for syntax and tests."""
    try:
        result = await critic.code_checker.check_code(
            code=request.code,
            language=request.language
        )
        
        return {
            "passed": result.passed,
            "issues": result.issues,
            "suggestions": result.suggestions,
            "confidence": result.confidence
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/check/policies")
async def check_policies(request: PolicyCheckRequest):
    """Check content against security policies."""
    try:
        result = await critic.policy_checker.check_policies(
            content=request.content,
            policies=request.policies
        )
        
        return {
            "passed": result.passed,
            "issues": result.issues,
            "suggestions": result.suggestions,
            "confidence": result.confidence
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/checks")
async def get_available_checks():
    """Get available validation checks."""
    return {
        "checks": [
            {
                "name": "facts",
                "description": "Check facts against Memory",
                "endpoint": "/check/facts"
            },
            {
                "name": "numbers",
                "description": "Validate numbers and calculations",
                "endpoint": "/check/numbers"
            },
            {
                "name": "code",
                "description": "Check code syntax and run tests",
                "endpoint": "/check/code"
            },
            {
                "name": "policies",
                "description": "Check security and policy compliance",
                "endpoint": "/check/policies"
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
        port=settings.critic_port,
        reload=True
    )
