"""Agent Registry Service - FastAPI application for managing AGI agents."""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from libs.common.config import settings

from .registry import AgentRegistry, AgentInfo, AgentRole, AgentStatus
from .agent_types import AgentCapabilities, AgentTypeDefinition


# Pydantic models for API
class AgentSpecRequest(BaseModel):
    """Request model for agent registration."""
    name: str = Field(..., description="Agent name")
    role: AgentRole = Field(..., description="Agent role")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    agent_type: str = Field(..., description="Agent type identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentInfoResponse(BaseModel):
    """Response model for agent information."""
    agent_id: str
    name: str
    role: AgentRole
    status: AgentStatus
    capabilities: List[str]
    performance_score: float
    resource_usage: Dict[str, float]
    last_seen: datetime
    swarm_affinity: float
    agent_type: str
    metadata: Dict[str, Any]


class AgentUpdateRequest(BaseModel):
    """Request model for agent updates."""
    status: Optional[AgentStatus] = None
    capabilities: Optional[List[str]] = None
    performance_score: Optional[float] = None
    resource_usage: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentSearchRequest(BaseModel):
    """Request model for agent search."""
    role: Optional[AgentRole] = None
    capabilities: Optional[List[str]] = None
    min_performance: Optional[float] = None
    status: Optional[AgentStatus] = None
    agent_type: Optional[str] = None


class NegotiationRequest(BaseModel):
    """Request model for negotiation invitation."""
    session_id: str
    resource_type: str
    total_available: float
    your_requirements: Dict[str, Any]
    deadline: datetime


class NegotiationResponse(BaseModel):
    """Response model for negotiation."""
    accept: bool
    resource_allocation: Optional[Dict[str, float]] = None
    conditions: Optional[List[str]] = None
    reasoning: str


# Global registry instance
agent_registry = AgentRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await agent_registry.initialize()
    print("Agent Registry service started")

    yield

    # Shutdown
    await agent_registry.shutdown()
    print("Agent Registry service stopped")


# Create FastAPI app
app = FastAPI(
    title="AGI Agent Registry Service",
    description="Service for managing AGI agent lifecycle and capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    agent_count = len(agent_registry.agents)
    active_count = len([
        agent for agent in agent_registry.agents.values()
        if agent.status == AgentStatus.AVAILABLE
    ])

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents": {
            "total": agent_count,
            "active": active_count
        }
    }


@app.post("/agents", response_model=AgentInfoResponse)
async def register_agent(request: AgentSpecRequest) -> AgentInfoResponse:
    """Register a new agent."""

    try:
        # Create agent info
        agent_info = AgentInfo(
            agent_id=str(uuid.uuid4()),
            name=request.name,
            role=request.role,
            status=AgentStatus.AVAILABLE,
            capabilities=set(request.capabilities),
            performance_score=0.5,  # Default starting score
            resource_usage={"cpu": 0.0, "memory": 0.0, "gpu": 0.0},
            last_seen=datetime.now(),
            swarm_affinity=0.7,  # Default affinity
            agent_type=request.agent_type,
            metadata=request.metadata
        )

        # Register agent
        registered_agent = await agent_registry.register_agent(agent_info)

        return AgentInfoResponse(
            agent_id=registered_agent.agent_id,
            name=registered_agent.name,
            role=registered_agent.role,
            status=registered_agent.status,
            capabilities=list(registered_agent.capabilities),
            performance_score=registered_agent.performance_score,
            resource_usage=registered_agent.resource_usage,
            last_seen=registered_agent.last_seen,
            swarm_affinity=registered_agent.swarm_affinity,
            agent_type=registered_agent.agent_type,
            metadata=registered_agent.metadata
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register agent: {str(e)}")


@app.get("/agents", response_model=List[AgentInfoResponse])
async def list_agents(
    role: Optional[AgentRole] = None,
    capability: Optional[str] = None,
    status: Optional[AgentStatus] = None,
    agent_type: Optional[str] = None,
    limit: int = 50
) -> List[AgentInfoResponse]:
    """List agents with optional filtering."""

    try:
        agents = await agent_registry.list_agents(
            role_filter=role,
            capability_filter=capability,
            status_filter=status,
            agent_type_filter=agent_type,
            limit=limit
        )

        return [
            AgentInfoResponse(
                agent_id=agent.agent_id,
                name=agent.name,
                role=agent.role,
                status=agent.status,
                capabilities=list(agent.capabilities),
                performance_score=agent.performance_score,
                resource_usage=agent.resource_usage,
                last_seen=agent.last_seen,
                swarm_affinity=agent.swarm_affinity,
                agent_type=agent.agent_type,
                metadata=agent.metadata
            )
            for agent in agents
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")


@app.get("/agents/{agent_id}", response_model=AgentInfoResponse)
async def get_agent(agent_id: str) -> AgentInfoResponse:
    """Get specific agent information."""

    try:
        agent = await agent_registry.get_agent(agent_id)

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        return AgentInfoResponse(
            agent_id=agent.agent_id,
            name=agent.name,
            role=agent.role,
            status=agent.status,
            capabilities=list(agent.capabilities),
            performance_score=agent.performance_score,
            resource_usage=agent.resource_usage,
            last_seen=agent.last_seen,
            swarm_affinity=agent.swarm_affinity,
            agent_type=agent.agent_type,
            metadata=agent.metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent: {str(e)}")


@app.put("/agents/{agent_id}", response_model=AgentInfoResponse)
async def update_agent(agent_id: str, request: AgentUpdateRequest) -> AgentInfoResponse:
    """Update agent information."""

    try:
        updates = {}
        if request.status is not None:
            updates["status"] = request.status
        if request.capabilities is not None:
            updates["capabilities"] = set(request.capabilities)
        if request.performance_score is not None:
            updates["performance_score"] = request.performance_score
        if request.resource_usage is not None:
            updates["resource_usage"] = request.resource_usage
        if request.metadata is not None:
            updates["metadata"] = request.metadata

        updated_agent = await agent_registry.update_agent(agent_id, updates)

        if not updated_agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        return AgentInfoResponse(
            agent_id=updated_agent.agent_id,
            name=updated_agent.name,
            role=updated_agent.role,
            status=updated_agent.status,
            capabilities=list(updated_agent.capabilities),
            performance_score=updated_agent.performance_score,
            resource_usage=updated_agent.resource_usage,
            last_seen=updated_agent.last_seen,
            swarm_affinity=updated_agent.swarm_affinity,
            agent_type=updated_agent.agent_type,
            metadata=updated_agent.metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update agent: {str(e)}")


@app.delete("/agents/{agent_id}")
async def deregister_agent(agent_id: str) -> Dict[str, str]:
    """Deregister an agent."""

    try:
        success = await agent_registry.deregister_agent(agent_id)

        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")

        return {"message": f"Agent {agent_id} deregistered successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to deregister agent: {str(e)}")


@app.post("/agents/search", response_model=List[AgentInfoResponse])
async def search_agents(request: AgentSearchRequest) -> List[AgentInfoResponse]:
    """Advanced agent search."""

    try:
        agents = await agent_registry.search_agents(
            role=request.role,
            capabilities=set(request.capabilities) if request.capabilities else None,
            min_performance=request.min_performance,
            status=request.status,
            agent_type=request.agent_type
        )

        return [
            AgentInfoResponse(
                agent_id=agent.agent_id,
                name=agent.name,
                role=agent.role,
                status=agent.status,
                capabilities=list(agent.capabilities),
                performance_score=agent.performance_score,
                resource_usage=agent.resource_usage,
                last_seen=agent.last_seen,
                swarm_affinity=agent.swarm_affinity,
                agent_type=agent.agent_type,
                metadata=agent.metadata
            )
            for agent in agents
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search agents: {str(e)}")


@app.post("/agents/{agent_id}/negotiate", response_model=NegotiationResponse)
async def handle_negotiation(agent_id: str, request: NegotiationRequest) -> NegotiationResponse:
    """Handle resource negotiation for an agent."""

    try:
        # Get agent
        agent = await agent_registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Simple negotiation logic (would be more sophisticated in practice)
        current_time = datetime.now()

        if current_time > request.deadline:
            return NegotiationResponse(
                accept=False,
                reasoning="Negotiation deadline exceeded"
            )

        # Check if requirements are reasonable
        cpu_required = request.your_requirements.get("cpu_cores", 0)
        memory_required = request.your_requirements.get("gpu_memory_gb", 0)

        # Accept if requirements are within reasonable bounds
        accept = cpu_required <= 4 and memory_required <= 8

        if accept:
            return NegotiationResponse(
                accept=True,
                resource_allocation={
                    "cpu_cores": min(cpu_required, 4),
                    "gpu_memory_gb": min(memory_required, 8),
                    "duration_hours": request.your_requirements.get("duration_hours", 1)
                },
                conditions=["Resource usage monitoring required", "Graceful shutdown on timeout"],
                reasoning="Requirements within acceptable bounds"
            )
        else:
            return NegotiationResponse(
                accept=False,
                reasoning="Resource requirements exceed available capacity"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Negotiation failed: {str(e)}")


@app.get("/agents/types")
async def get_agent_types() -> Dict[str, Dict[str, Any]]:
    """Get available agent types."""

    try:
        return await agent_registry.get_agent_types()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent types: {str(e)}")


@app.post("/agents/{agent_id}/notify")
async def notify_agent(agent_id: str, notification: Dict[str, Any]) -> Dict[str, str]:
    """Send notification to an agent (for internal use)."""

    # This endpoint is used by other services to notify agents
    # In practice, this would trigger actual agent notification mechanisms

    try:
        agent = await agent_registry.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Update last seen
        await agent_registry.update_agent(agent_id, {"last_seen": datetime.now()})

        return {"message": f"Notification sent to agent {agent_id}"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to notify agent: {str(e)}")


@app.get("/stats")
async def get_registry_stats() -> Dict[str, Any]:
    """Get registry statistics."""

    try:
        return await agent_registry.get_stats()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    port = int(settings.get("AGENT_REGISTRY_PORT", 8006))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.get("DEBUG", False)
    )
