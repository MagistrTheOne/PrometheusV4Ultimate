"""Agent Registry - Core logic for managing AGI agents."""

import asyncio
import json
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path

from libs.common.config import settings


class AgentRole(Enum):
    """Agent specialization roles."""
    RESEARCHER = "researcher"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    LEARNER = "learner"
    COORDINATOR = "coordinator"


class AgentStatus(Enum):
    """Agent operational status."""
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class AgentInfo:
    """Agent metadata and capabilities."""
    agent_id: str
    name: str
    role: AgentRole
    status: AgentStatus
    capabilities: Set[str]
    performance_score: float
    resource_usage: Dict[str, float]
    last_seen: datetime
    swarm_affinity: float  # 0-1, how well it works in teams
    agent_type: str
    metadata: Dict[str, Any]


class AgentRegistry:
    """Registry for managing AGI agents."""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or settings.get("AGENT_REGISTRY_STORAGE", "./data/agent_registry"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.agents: Dict[str, AgentInfo] = {}
        self.agent_types: Dict[str, Dict[str, Any]] = {}

        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize the registry."""
        await self._load_registry()
        await self._initialize_agent_types()

        # Start background tasks
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.health_check_task = asyncio.create_task(self._health_check_loop())

    async def shutdown(self) -> None:
        """Shutdown the registry."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.health_check_task:
            self.health_check_task.cancel()

        await self._save_registry()

    async def register_agent(self, agent_info: AgentInfo) -> AgentInfo:
        """Register a new agent."""

        # Check for duplicate agent_id
        if agent_info.agent_id in self.agents:
            raise ValueError(f"Agent {agent_info.agent_id} already registered")

        # Validate agent type
        if agent_info.agent_type not in self.agent_types:
            raise ValueError(f"Unknown agent type: {agent_info.agent_type}")

        # Add to registry
        self.agents[agent_info.agent_id] = agent_info

        # Save to disk
        await self._save_agent(agent_info)

        print(f"Registered agent: {agent_info.name} ({agent_info.agent_id})")
        return agent_info

    async def deregister_agent(self, agent_id: str) -> bool:
        """Deregister an agent."""

        if agent_id not in self.agents:
            return False

        # Remove from registry
        agent_info = self.agents.pop(agent_id)

        # Remove from disk
        await self._delete_agent(agent_id)

        print(f"Deregistered agent: {agent_info.name} ({agent_id})")
        return True

    async def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent information."""

        return self.agents.get(agent_id)

    async def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> Optional[AgentInfo]:
        """Update agent information."""

        if agent_id not in self.agents:
            return None

        agent = self.agents[agent_id]

        # Apply updates
        for key, value in updates.items():
            if hasattr(agent, key):
                setattr(agent, key, value)

        # Update last seen if not explicitly set
        if "last_seen" not in updates:
            agent.last_seen = datetime.now()

        # Save to disk
        await self._save_agent(agent)

        return agent

    async def list_agents(
        self,
        role_filter: Optional[AgentRole] = None,
        capability_filter: Optional[str] = None,
        status_filter: Optional[AgentStatus] = None,
        agent_type_filter: Optional[str] = None,
        limit: int = 50
    ) -> List[AgentInfo]:
        """List agents with optional filtering."""

        agents = list(self.agents.values())

        # Apply filters
        if role_filter:
            agents = [a for a in agents if a.role == role_filter]

        if capability_filter:
            agents = [a for a in agents if capability_filter in a.capabilities]

        if status_filter:
            agents = [a for a in agents if a.status == status_filter]

        if agent_type_filter:
            agents = [a for a in agents if a.agent_type == agent_type_filter]

        # Sort by performance and last seen
        agents.sort(key=lambda a: (a.performance_score, a.last_seen), reverse=True)

        return agents[:limit]

    async def search_agents(
        self,
        role: Optional[AgentRole] = None,
        capabilities: Optional[Set[str]] = None,
        min_performance: Optional[float] = None,
        status: Optional[AgentStatus] = None,
        agent_type: Optional[str] = None,
        limit: int = 20
    ) -> List[AgentInfo]:
        """Advanced agent search."""

        candidates = list(self.agents.values())

        # Apply filters
        if role:
            candidates = [a for a in candidates if a.role == role]

        if capabilities:
            candidates = [a for a in candidates if capabilities.issubset(a.capabilities)]

        if min_performance:
            candidates = [a for a in candidates if a.performance_score >= min_performance]

        if status:
            candidates = [a for a in candidates if a.status == status]

        if agent_type:
            candidates = [a for a in candidates if a.agent_type == agent_type]

        # Score and rank candidates
        scored_candidates = []
        for agent in candidates:
            score = self._calculate_agent_score(agent, capabilities, min_performance or 0.0)
            scored_candidates.append((agent, score))

        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        return [agent for agent, _ in scored_candidates[:limit]]

    def _calculate_agent_score(
        self,
        agent: AgentInfo,
        required_capabilities: Optional[Set[str]],
        min_performance: float
    ) -> float:
        """Calculate relevance score for an agent."""

        score = 0.0

        # Performance score (0-40 points)
        performance_score = agent.performance_score * 40
        score += performance_score

        # Capability match (0-30 points)
        if required_capabilities:
            capability_match = len(required_capabilities.intersection(agent.capabilities))
            capability_score = (capability_match / len(required_capabilities)) * 30
            score += capability_score

        # Availability bonus (0-10 points)
        if agent.status == AgentStatus.AVAILABLE:
            score += 10

        # Swarm affinity bonus (0-10 points)
        score += agent.swarm_affinity * 10

        # Recency bonus (0-10 points) - prefer recently active agents
        hours_since_seen = (datetime.now() - agent.last_seen).total_seconds() / 3600
        recency_score = max(0, 10 - hours_since_seen)  # Decay over 10 hours
        score += recency_score

        return score

    async def get_agent_types(self) -> Dict[str, Dict[str, Any]]:
        """Get available agent types."""

        return self.agent_types.copy()

    async def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""

        total_agents = len(self.agents)
        active_agents = len([a for a in self.agents.values() if a.status == AgentStatus.AVAILABLE])
        busy_agents = len([a for a in self.agents.values() if a.status == AgentStatus.BUSY])
        offline_agents = len([a for a in self.agents.values() if a.status == AgentStatus.OFFLINE])
        error_agents = len([a for a in self.agents.values() if a.status == AgentStatus.ERROR])

        role_distribution = {}
        for role in AgentRole:
            role_distribution[role.value] = len([
                a for a in self.agents.values() if a.role == role
            ])

        avg_performance = (
            sum(a.performance_score for a in self.agents.values()) / total_agents
            if total_agents > 0 else 0.0
        )

        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "busy_agents": busy_agents,
            "offline_agents": offline_agents,
            "error_agents": error_agents,
            "role_distribution": role_distribution,
            "average_performance": avg_performance,
            "agent_types": list(self.agent_types.keys())
        }

    async def _initialize_agent_types(self) -> None:
        """Initialize predefined agent types."""

        self.agent_types = {
            "researcher_basic": {
                "name": "Basic Researcher",
                "description": "General research and information gathering agent",
                "default_capabilities": ["research", "analysis", "web_search"],
                "resource_requirements": {"cpu": 2, "memory": 4, "gpu": 0},
                "performance_baseline": 0.6
            },
            "executor_basic": {
                "name": "Basic Executor",
                "description": "General task execution agent",
                "default_capabilities": ["execution", "tool_usage", "file_operations"],
                "resource_requirements": {"cpu": 2, "memory": 4, "gpu": 0},
                "performance_baseline": 0.7
            },
            "validator_basic": {
                "name": "Basic Validator",
                "description": "General validation and verification agent",
                "default_capabilities": ["validation", "checking", "quality_assurance"],
                "resource_requirements": {"cpu": 1, "memory": 2, "gpu": 0},
                "performance_baseline": 0.8
            },
            "learner_basic": {
                "name": "Basic Learner",
                "description": "General learning and adaptation agent",
                "default_capabilities": ["learning", "adaptation", "pattern_recognition"],
                "resource_requirements": {"cpu": 3, "memory": 8, "gpu": 1},
                "performance_baseline": 0.5
            },
            "coordinator_basic": {
                "name": "Basic Coordinator",
                "description": "General coordination and management agent",
                "default_capabilities": ["coordination", "planning", "resource_management"],
                "resource_requirements": {"cpu": 2, "memory": 4, "gpu": 0},
                "performance_baseline": 0.7
            }
        }

    async def _load_registry(self) -> None:
        """Load registry from disk."""

        registry_file = self.storage_path / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)

                for agent_data in data.get("agents", []):
                    # Convert string timestamps back to datetime
                    if "last_seen" in agent_data:
                        agent_data["last_seen"] = datetime.fromisoformat(agent_data["last_seen"])

                    # Convert role and status back to enums
                    agent_data["role"] = AgentRole(agent_data["role"])
                    agent_data["status"] = AgentStatus(agent_data["status"])

                    # Convert capabilities back to set
                    agent_data["capabilities"] = set(agent_data["capabilities"])

                    agent = AgentInfo(**agent_data)
                    self.agents[agent.agent_id] = agent

                print(f"Loaded {len(self.agents)} agents from registry")

            except Exception as e:
                print(f"Failed to load registry: {e}")

    async def _save_registry(self) -> None:
        """Save registry to disk."""

        try:
            data = {
                "agents": [
                    {
                        **asdict(agent),
                        "role": agent.role.value,
                        "status": agent.status.value,
                        "capabilities": list(agent.capabilities),
                        "last_seen": agent.last_seen.isoformat()
                    }
                    for agent in self.agents.values()
                ],
                "last_updated": datetime.now().isoformat()
            }

            registry_file = self.storage_path / "registry.json"
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            print(f"Failed to save registry: {e}")

    async def _save_agent(self, agent: AgentInfo) -> None:
        """Save individual agent to disk."""

        try:
            agent_file = self.storage_path / f"agent_{agent.agent_id}.json"
            data = {
                **asdict(agent),
                "role": agent.role.value,
                "status": agent.status.value,
                "capabilities": list(agent.capabilities),
                "last_seen": agent.last_seen.isoformat()
            }

            with open(agent_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            print(f"Failed to save agent {agent.agent_id}: {e}")

    async def _delete_agent(self, agent_id: str) -> None:
        """Delete agent file from disk."""

        try:
            agent_file = self.storage_path / f"agent_{agent_id}.json"
            if agent_file.exists():
                agent_file.unlink()

        except Exception as e:
            print(f"Failed to delete agent file {agent_id}: {e}")

    async def _cleanup_loop(self) -> None:
        """Background task to clean up stale agents."""

        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                current_time = datetime.now()
                stale_threshold = timedelta(hours=1)  # Mark as offline after 1 hour

                stale_agents = []
                for agent_id, agent in self.agents.items():
                    if (current_time - agent.last_seen) > stale_threshold:
                        if agent.status != AgentStatus.OFFLINE:
                            stale_agents.append(agent_id)

                # Mark stale agents as offline
                for agent_id in stale_agents:
                    await self.update_agent(agent_id, {"status": AgentStatus.OFFLINE})
                    print(f"Marked agent {agent_id} as offline (stale)")

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _health_check_loop(self) -> None:
        """Background task to perform health checks."""

        while True:
            try:
                await asyncio.sleep(60)  # Run every minute

                # Simple health check - just ensure agents are responsive
                # In practice, this would ping agent endpoints

                current_time = datetime.now()
                for agent_id, agent in list(self.agents.items()):
                    # Mark agents that haven't been seen recently as potentially offline
                    time_since_seen = current_time - agent.last_seen

                    if time_since_seen > timedelta(minutes=10) and agent.status == AgentStatus.AVAILABLE:
                        # Could perform actual health check here
                        pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in health check loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
