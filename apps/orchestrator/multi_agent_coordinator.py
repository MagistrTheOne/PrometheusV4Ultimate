"""Multi-Agent Coordinator for AGI swarm intelligence."""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from libs.common.schemas import Task, TaskStatus
from libs.common.config import settings
import httpx


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
    role: AgentRole
    status: AgentStatus
    capabilities: Set[str]
    performance_score: float
    resource_usage: Dict[str, float]
    last_seen: float
    swarm_affinity: float  # 0-1, how well it works in teams


@dataclass
class SwarmTask:
    """Task for multi-agent execution."""
    task_id: str
    goal: str
    complexity: float
    required_capabilities: Set[str]
    max_agents: int
    deadline: Optional[float]
    priority: int


@dataclass
class TaskAssignment:
    """Assignment of agents to task."""
    task_id: str
    agent_assignments: Dict[str, str]  # agent_id -> role
    expected_completion: float
    resource_requirements: Dict[str, float]
    risk_assessment: float


class MultiAgentCoordinator:
    """Coordinates swarm intelligence for AGI tasks."""

    def __init__(self):
        self.agent_registry: Dict[str, AgentInfo] = {}
        self.active_swarms: Dict[str, List[str]] = {}  # task_id -> agent_ids
        self.task_assignments: Dict[str, TaskAssignment] = {}
        self.coordination_lock = asyncio.Lock()

    async def coordinate_task_execution(
        self,
        task: Task,
        available_agents: List[AgentInfo]
    ) -> TaskAssignment:
        """Coordinate multi-agent execution of a task."""

        async with self.coordination_lock:
            # Analyze task requirements
            swarm_task = await self._analyze_task_requirements(task)

            # Select optimal agent team
            selected_agents = await self._select_agent_team(
                swarm_task, available_agents
            )

            # Assign roles and responsibilities
            assignment = await self._create_task_assignment(
                swarm_task, selected_agents
            )

            # Initialize swarm coordination
            await self._initialize_swarm_coordination(assignment)

            # Store assignment for tracking
            self.task_assignments[task.id] = assignment

            return assignment

    async def _analyze_task_requirements(self, task: Task) -> SwarmTask:
        """Analyze task to determine swarm requirements."""

        # Estimate complexity based on goal length and structure
        complexity = min(1.0, len(task.goal) / 1000.0)

        # Extract required capabilities from task description
        required_capabilities = await self._extract_capabilities(task.goal)

        # Determine optimal team size
        max_agents = min(8, max(2, int(complexity * 6) + 1))

        return SwarmTask(
            task_id=task.id,
            goal=task.goal,
            complexity=complexity,
            required_capabilities=required_capabilities,
            max_agents=max_agents,
            deadline=None,  # Could be extracted from task metadata
            priority=1  # Default priority
        )

    async def _extract_capabilities(self, goal: str) -> Set[str]:
        """Extract required capabilities from task goal."""

        capabilities = set()

        # Simple keyword-based capability extraction
        capability_keywords = {
            "research": ["analyze", "investigate", "study", "explore"],
            "execution": ["implement", "build", "create", "execute"],
            "validation": ["verify", "check", "validate", "test"],
            "learning": ["learn", "adapt", "improve", "optimize"],
            "coordination": ["coordinate", "manage", "organize", "plan"]
        }

        goal_lower = goal.lower()
        for capability, keywords in capability_keywords.items():
            if any(keyword in goal_lower for keyword in keywords):
                capabilities.add(capability)

        # Default capabilities if none detected
        if not capabilities:
            capabilities.update(["research", "execution", "validation"])

        return capabilities

    async def _select_agent_team(
        self,
        swarm_task: SwarmTask,
        available_agents: List[AgentInfo]
    ) -> Dict[AgentRole, List[AgentInfo]]:
        """Select optimal team of agents for the task."""

        team = {}

        # Filter agents by capability match
        capable_agents = [
            agent for agent in available_agents
            if self._agent_matches_capabilities(agent, swarm_task.required_capabilities)
        ]

        if not capable_agents:
            # Fallback to any available agents
            capable_agents = available_agents

        # Assign agents to roles based on their specialization
        for role in AgentRole:
            role_agents = [
                agent for agent in capable_agents
                if agent.role == role and agent.status == AgentStatus.AVAILABLE
            ]

            if role_agents:
                # Sort by performance and swarm affinity
                role_agents.sort(
                    key=lambda a: a.performance_score * a.swarm_affinity,
                    reverse=True
                )

                # Take top agents for this role
                max_for_role = min(len(role_agents), swarm_task.max_agents // len(AgentRole))
                team[role] = role_agents[:max_for_role]

        # Ensure minimum team composition
        await self._ensure_minimum_team_composition(team, capable_agents)

        return team

    def _agent_matches_capabilities(
        self,
        agent: AgentInfo,
        required_capabilities: Set[str]
    ) -> bool:
        """Check if agent has required capabilities."""

        # Check for exact capability matches
        exact_matches = required_capabilities.intersection(agent.capabilities)

        # Check for role-based capability inference
        role_capabilities = {
            AgentRole.RESEARCHER: {"research", "analysis"},
            AgentRole.EXECUTOR: {"execution", "implementation"},
            AgentRole.VALIDATOR: {"validation", "verification"},
            AgentRole.LEARNER: {"learning", "adaptation"},
            AgentRole.COORDINATOR: {"coordination", "management"}
        }

        role_matches = required_capabilities.intersection(
            role_capabilities.get(agent.role, set())
        )

        return len(exact_matches) > 0 or len(role_matches) > 0

    async def _ensure_minimum_team_composition(
        self,
        team: Dict[AgentRole, List[AgentInfo]],
        available_agents: List[AgentInfo]
    ) -> None:
        """Ensure team has minimum required composition."""

        min_roles = {AgentRole.RESEARCHER, AgentRole.EXECUTOR, AgentRole.VALIDATOR}

        for required_role in min_roles:
            if required_role not in team or not team[required_role]:
                # Find best available agent for this role
                candidates = [
                    agent for agent in available_agents
                    if agent.status == AgentStatus.AVAILABLE
                ]

                if candidates:
                    # Sort by versatility (agents that can fill multiple roles)
                    candidates.sort(key=lambda a: len(a.capabilities), reverse=True)
                    team[required_role] = [candidates[0]]

    async def _create_task_assignment(
        self,
        swarm_task: SwarmTask,
        agent_team: Dict[AgentRole, List[AgentInfo]]
    ) -> TaskAssignment:
        """Create detailed task assignment for the agent team."""

        agent_assignments = {}

        # Flatten team into agent -> role mapping
        for role, agents in agent_team.items():
            for agent in agents:
                agent_assignments[agent.agent_id] = role.value

        # Calculate resource requirements
        resource_requirements = await self._calculate_resource_requirements(agent_team)

        # Estimate completion time
        expected_completion = await self._estimate_completion_time(
            swarm_task, agent_team
        )

        # Assess execution risk
        risk_assessment = await self._assess_execution_risk(swarm_task, agent_team)

        return TaskAssignment(
            task_id=swarm_task.task_id,
            agent_assignments=agent_assignments,
            expected_completion=expected_completion,
            resource_requirements=resource_requirements,
            risk_assessment=risk_assessment
        )

    async def _calculate_resource_requirements(
        self,
        agent_team: Dict[AgentRole, List[AgentInfo]]
    ) -> Dict[str, float]:
        """Calculate total resource requirements for the team."""

        requirements = {
            "cpu_cores": 0.0,
            "gpu_memory_gb": 0.0,
            "ram_gb": 0.0,
            "network_bandwidth_mbps": 0.0
        }

        # Sum resource usage across all agents
        for role_agents in agent_team.values():
            for agent in role_agents:
                for resource, usage in agent.resource_usage.items():
                    if resource in requirements:
                        requirements[resource] += usage

        # Add coordination overhead
        team_size = sum(len(agents) for agents in agent_team.values())
        coordination_factor = 1.0 + (team_size - 1) * 0.1  # 10% overhead per additional agent

        for resource in requirements:
            requirements[resource] *= coordination_factor

        return requirements

    async def _estimate_completion_time(
        self,
        swarm_task: SwarmTask,
        agent_team: Dict[AgentRole, List[AgentInfo]]
    ) -> float:
        """Estimate task completion time."""

        base_time = swarm_task.complexity * 3600  # Base time in seconds

        # Factor in team performance
        total_performance = sum(
            agent.performance_score
            for agents in agent_team.values()
            for agent in agents
        )

        team_size = sum(len(agents) for agents in agent_team.values())
        avg_performance = total_performance / max(team_size, 1)

        # Better teams work faster, but coordination overhead increases
        coordination_penalty = 1.0 + (team_size - 1) * 0.15  # 15% penalty per additional agent

        estimated_time = (base_time / avg_performance) * coordination_penalty

        return estimated_time

    async def _assess_execution_risk(
        self,
        swarm_task: SwarmTask,
        agent_team: Dict[AgentRole, List[AgentInfo]]
    ) -> float:
        """Assess risk of task execution failure."""

        risk_factors = []

        # Team composition risk
        team_size = sum(len(agents) for agents in agent_team.values())
        if team_size < 3:
            risk_factors.append(0.3)  # Too small team
        elif team_size > 6:
            risk_factors.append(0.2)  # Coordination complexity

        # Capability coverage risk
        covered_capabilities = set()
        for agents in agent_team.values():
            for agent in agents:
                covered_capabilities.update(agent.capabilities)

        coverage_ratio = len(covered_capabilities.intersection(swarm_task.required_capabilities)) / len(swarm_task.required_capabilities)
        risk_factors.append(1.0 - coverage_ratio)

        # Experience risk
        avg_experience = sum(
            agent.performance_score
            for agents in agent_team.values()
            for agent in agents
        ) / max(team_size, 1)

        experience_risk = 1.0 - avg_experience
        risk_factors.append(experience_risk)

        # Overall risk as weighted average
        overall_risk = sum(risk_factors) / len(risk_factors)

        return min(1.0, max(0.0, overall_risk))

    async def _initialize_swarm_coordination(self, assignment: TaskAssignment) -> None:
        """Initialize coordination mechanisms for the agent swarm."""

        # Register active swarm
        self.active_swarms[assignment.task_id] = list(assignment.agent_assignments.keys())

        # Notify agents of their assignments
        await self._notify_agents_of_assignment(assignment)

        # Set up communication channels
        await self._setup_agent_communication(assignment)

        # Initialize progress tracking
        await self._initialize_progress_tracking(assignment)

    async def _notify_agents_of_assignment(self, assignment: TaskAssignment) -> None:
        """Notify agents of their task assignments."""

        for agent_id, role in assignment.agent_assignments.items():
            try:
                # Send assignment notification via agent registry
                await self._send_agent_notification(
                    agent_id,
                    {
                        "type": "task_assignment",
                        "task_id": assignment.task_id,
                        "role": role,
                        "expected_completion": assignment.expected_completion,
                        "resource_requirements": assignment.resource_requirements
                    }
                )
            except Exception as e:
                # Log notification failure but don't fail the assignment
                print(f"Failed to notify agent {agent_id}: {e}")

    async def _setup_agent_communication(self, assignment: TaskAssignment) -> None:
        """Set up communication channels between agents."""

        agent_ids = list(assignment.agent_assignments.keys())

        # Create communication channels for all agent pairs
        for i, agent_a in enumerate(agent_ids):
            for agent_b in agent_ids[i+1:]:
                await self._establish_agent_channel(agent_a, agent_b)

    async def _initialize_progress_tracking(self, assignment: TaskAssignment) -> None:
        """Initialize progress tracking for the swarm."""

        # Set up progress monitoring for each agent
        for agent_id in assignment.agent_assignments.keys():
            await self._setup_agent_progress_tracking(
                agent_id, assignment.task_id
            )

    async def _send_agent_notification(self, agent_id: str, notification: Dict[str, Any]) -> None:
        """Send notification to specific agent via registry."""

        # This would integrate with the agent registry service
        # For now, placeholder for HTTP call to agent registry
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"http://agent-registry:8005/agents/{agent_id}/notify",
                    json=notification,
                    timeout=5.0
                )
                response.raise_for_status()
            except Exception:
                # Agent may be offline, that's ok for now
                pass

    async def _establish_agent_channel(self, agent_a: str, agent_b: str) -> None:
        """Establish communication channel between two agents."""

        # This would set up messaging channels
        # For now, placeholder for future implementation
        pass

    async def _setup_agent_progress_tracking(self, agent_id: str, task_id: str) -> None:
        """Set up progress tracking for an agent."""

        # This would integrate with observability system
        # For now, placeholder for future implementation
        pass

    async def get_swarm_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active swarm."""

        if task_id not in self.active_swarms:
            return None

        agent_ids = self.active_swarms[task_id]
        assignment = self.task_assignments.get(task_id)

        if not assignment:
            return None

        # Gather status from all agents
        agent_statuses = {}
        for agent_id in agent_ids:
            agent_statuses[agent_id] = await self._get_agent_status(agent_id)

        return {
            "task_id": task_id,
            "agent_count": len(agent_ids),
            "agent_statuses": agent_statuses,
            "assignment": assignment,
            "coordination_health": await self._assess_coordination_health(agent_ids)
        }

    async def _get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of a specific agent."""

        # This would query the agent registry
        # For now, return placeholder status
        return {
            "status": "active",
            "progress": 0.5,
            "last_update": asyncio.get_event_loop().time()
        }

    async def _assess_coordination_health(self, agent_ids: List[str]) -> float:
        """Assess health of swarm coordination."""

        # Simple health assessment based on agent responsiveness
        health_scores = []
        for agent_id in agent_ids:
            try:
                status = await self._get_agent_status(agent_id)
                health_scores.append(1.0 if status["status"] == "active" else 0.0)
            except Exception:
                health_scores.append(0.0)

        return sum(health_scores) / len(health_scores) if health_scores else 0.0

    async def terminate_swarm(self, task_id: str) -> bool:
        """Terminate an active swarm."""

        if task_id not in self.active_swarms:
            return False

        agent_ids = self.active_swarms[task_id]

        # Notify agents of termination
        termination_tasks = []
        for agent_id in agent_ids:
            termination_tasks.append(self._notify_agent_termination(agent_id, task_id))

        await asyncio.gather(*termination_tasks, return_exceptions=True)

        # Clean up swarm tracking
        del self.active_swarms[task_id]
        if task_id in self.task_assignments:
            del self.task_assignments[task_id]

        return True

    async def _notify_agent_termination(self, agent_id: str, task_id: str) -> None:
        """Notify agent of swarm termination."""

        try:
            await self._send_agent_notification(
                agent_id,
                {
                    "type": "swarm_termination",
                    "task_id": task_id,
                    "reason": "task_completed"
                }
            )
        except Exception:
            # Agent may be offline, that's ok
            pass


# Global instance
multi_agent_coordinator = MultiAgentCoordinator()
