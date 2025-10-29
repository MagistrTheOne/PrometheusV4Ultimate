"""Dynamic Team Formation for multi-agent task execution."""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from libs.common.config import settings
from .multi_agent_coordinator import AgentInfo, AgentRole, SwarmTask
import httpx


class TeamFormationStrategy(Enum):
    """Strategies for forming agent teams."""
    CAPABILITY_MATCHING = "capability_matching"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    BALANCED_DIVERSITY = "balanced_diversity"
    SPECIALIZATION_FOCUS = "specialization_focus"
    RAPID_ASSEMBLY = "rapid_assembly"


@dataclass
class TeamComposition:
    """Composition of an agent team."""
    team_id: str
    leader_agent: str
    members: Dict[str, AgentRole]  # agent_id -> role
    synergy_score: float  # 0-1, how well team members work together
    capability_coverage: float  # 0-1, coverage of required capabilities
    performance_expectation: float  # expected team performance
    formation_time: datetime
    expected_duration: float  # expected task completion time


@dataclass
class TeamFormationRequest:
    """Request for team formation."""
    request_id: str
    task: SwarmTask
    available_agents: List[AgentInfo]
    strategy: TeamFormationStrategy
    max_team_size: int
    deadline: Optional[datetime]
    constraints: Dict[str, Any]


@dataclass
class TeamFormationResult:
    """Result of team formation."""
    request_id: str
    team: Optional[TeamComposition]
    alternative_teams: List[TeamComposition]
    formation_time_seconds: float
    candidates_evaluated: int
    reason: Optional[str]  # If no team could be formed


class DynamicTeamFormation:
    """Engine for dynamic formation of agent teams."""

    def __init__(self):
        self.formation_history: Dict[str, TeamFormationResult] = {}
        self.team_synergy_cache: Dict[Tuple[str, ...], float] = {}
        self.performance_timeout = 30.0  # seconds

    async def form_team(
        self,
        task: SwarmTask,
        available_agents: List[AgentInfo],
        strategy: TeamFormationStrategy = TeamFormationStrategy.CAPABILITY_MATCHING,
        max_team_size: Optional[int] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> TeamFormationResult:
        """Form an optimal team for the given task."""

        start_time = datetime.now()

        # Create formation request
        request = TeamFormationRequest(
            request_id=str(uuid.uuid4()),
            task=task,
            available_agents=available_agents,
            strategy=strategy,
            max_team_size=max_team_size or task.max_agents,
            deadline=None,
            constraints=constraints or {}
        )

        try:
            # Apply selected formation strategy
            if strategy == TeamFormationStrategy.CAPABILITY_MATCHING:
                team = await self._form_team_capability_matching(request)
            elif strategy == TeamFormationStrategy.PERFORMANCE_OPTIMIZATION:
                team = await self._form_team_performance_optimization(request)
            elif strategy == TeamFormationStrategy.BALANCED_DIVERSITY:
                team = await self._form_team_balanced_diversity(request)
            elif strategy == TeamFormationStrategy.SPECIALIZATION_FOCUS:
                team = await self._form_team_specialization_focus(request)
            elif strategy == TeamFormationStrategy.RAPID_ASSEMBLY:
                team = await self._form_team_rapid_assembly(request)
            else:
                team = await self._form_team_capability_matching(request)  # fallback

            # Generate alternative teams
            alternative_teams = await self._generate_alternative_teams(request, team)

            formation_time = (datetime.now() - start_time).total_seconds()

            result = TeamFormationResult(
                request_id=request.request_id,
                team=team,
                alternative_teams=alternative_teams,
                formation_time_seconds=formation_time,
                candidates_evaluated=len(available_agents),
                reason=None
            )

            # Cache result
            self.formation_history[request.request_id] = result

            return result

        except Exception as e:
            formation_time = (datetime.now() - start_time).total_seconds()

            return TeamFormationResult(
                request_id=request.request_id,
                team=None,
                alternative_teams=[],
                formation_time_seconds=formation_time,
                candidates_evaluated=len(available_agents),
                reason=f"Formation failed: {str(e)}"
            )

    async def _form_team_capability_matching(self, request: TeamFormationRequest) -> Optional[TeamComposition]:
        """Form team based on capability matching."""

        task = request.task
        agents = request.available_agents

        # Find agents that match required capabilities
        capability_matches = {}
        for agent in agents:
            matches = task.required_capabilities.intersection(agent.capabilities)
            if matches:
                capability_matches[agent.agent_id] = {
                    'agent': agent,
                    'matches': len(matches),
                    'capabilities': matches
                }

        if not capability_matches:
            return None

        # Sort by number of matches and performance
        sorted_matches = sorted(
            capability_matches.values(),
            key=lambda x: (x['matches'], x['agent'].performance_score),
            reverse=True
        )

        # Select top agents up to max team size
        selected_agents = sorted_matches[:request.max_team_size]

        # Assign roles based on agent specialization
        team_members = await self._assign_roles_to_agents(
            [match['agent'] for match in selected_agents],
            task
        )

        # Create team composition
        return await self._create_team_composition(
            team_members, task, "capability_matching"
        )

    async def _form_team_performance_optimization(self, request: TeamFormationRequest) -> Optional[TeamComposition]:
        """Form team optimized for performance."""

        task = request.task
        agents = request.available_agents

        # Score agents by performance and capability relevance
        agent_scores = []
        for agent in agents:
            capability_score = len(task.required_capabilities.intersection(agent.capabilities))
            performance_score = agent.performance_score
            availability_score = 1.0 if agent.status.name == 'AVAILABLE' else 0.5

            total_score = (
                capability_score * 0.4 +
                performance_score * 0.4 +
                availability_score * 0.2
            )

            agent_scores.append((agent, total_score))

        # Sort by total score
        agent_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top performers
        selected_agents = [agent for agent, _ in agent_scores[:request.max_team_size]]

        # Assign roles
        team_members = await self._assign_roles_to_agents(selected_agents, task)

        return await self._create_team_composition(
            team_members, task, "performance_optimization"
        )

    async def _form_team_balanced_diversity(self, request: TeamFormationRequest) -> Optional[TeamComposition]:
        """Form team with balanced diversity of capabilities."""

        task = request.task
        agents = request.available_agents

        # Group agents by their primary capabilities
        capability_groups = {}
        for agent in agents:
            for capability in agent.capabilities:
                if capability not in capability_groups:
                    capability_groups[capability] = []
                capability_groups[capability].append(agent)

        # Select diverse agents covering required capabilities
        selected_agents = []
        covered_capabilities = set()

        # First, ensure all required capabilities are covered
        for required_cap in task.required_capabilities:
            if required_cap in capability_groups:
                candidates = [
                    agent for agent in capability_groups[required_cap]
                    if agent not in selected_agents
                ]

                if candidates:
                    # Select best candidate for this capability
                    best_candidate = max(candidates, key=lambda a: a.performance_score)
                    selected_agents.append(best_candidate)
                    covered_capabilities.update(best_candidate.capabilities)

        # Fill remaining slots with diverse agents
        remaining_slots = request.max_team_size - len(selected_agents)
        if remaining_slots > 0:
            available_agents = [
                agent for agent in agents
                if agent not in selected_agents and agent.status.name == 'AVAILABLE'
            ]

            # Select agents that add new capabilities
            for agent in available_agents:
                if len(selected_agents) >= request.max_team_size:
                    break

                new_capabilities = agent.capabilities - covered_capabilities
                if new_capabilities:
                    selected_agents.append(agent)
                    covered_capabilities.update(agent.capabilities)

        if len(selected_agents) < 2:  # Minimum team size
            return None

        # Assign roles
        team_members = await self._assign_roles_to_agents(selected_agents, task)

        return await self._create_team_composition(
            team_members, task, "balanced_diversity"
        )

    async def _form_team_specialization_focus(self, request: TeamFormationRequest) -> Optional[TeamComposition]:
        """Form team focused on specific specializations."""

        task = request.task
        agents = request.available_agents

        # Find the most critical capability for the task
        critical_capability = await self._identify_critical_capability(task)

        # Select agents specialized in critical capability
        specialized_agents = [
            agent for agent in agents
            if critical_capability in agent.capabilities and
            agent.status.name == 'AVAILABLE'
        ]

        if not specialized_agents:
            # Fallback to capability matching
            return await self._form_team_capability_matching(request)

        # Sort by specialization strength
        specialized_agents.sort(key=lambda a: a.performance_score, reverse=True)

        # Take top specialized agents
        selected_agents = specialized_agents[:request.max_team_size]

        # Assign roles
        team_members = await self._assign_roles_to_agents(selected_agents, task)

        return await self._create_team_composition(
            team_members, task, "specialization_focus"
        )

    async def _form_team_rapid_assembly(self, request: TeamFormationRequest) -> Optional[TeamComposition]:
        """Form team using rapid assembly (fastest method)."""

        task = request.task
        agents = request.available_agents

        # Simple greedy selection
        available_agents = [
            agent for agent in agents
            if agent.status.name == 'AVAILABLE'
        ]

        if len(available_agents) < 2:
            return None

        # Take first N available agents
        selected_agents = available_agents[:request.max_team_size]

        # Quick role assignment
        team_members = await self._quick_assign_roles(selected_agents, task)

        return await self._create_team_composition(
            team_members, task, "rapid_assembly"
        )

    async def _identify_critical_capability(self, task: SwarmTask) -> str:
        """Identify the most critical capability for a task."""

        # Simple heuristic: most frequently mentioned capability
        # In practice, this would use ML to analyze task requirements
        capability_importance = {}
        for capability in task.required_capabilities:
            # Assign importance based on task complexity and capability
            importance = task.complexity
            if capability in ['coordination', 'planning']:
                importance *= 1.5  # More important for complex tasks
            capability_importance[capability] = importance

        return max(capability_importance.keys(), key=lambda k: capability_importance[k])

    async def _assign_roles_to_agents(
        self,
        agents: List[AgentInfo],
        task: SwarmTask
    ) -> Dict[str, AgentRole]:
        """Assign appropriate roles to selected agents."""

        team_members = {}

        # Sort agents by their role preferences and capabilities
        role_assignments = []

        for agent in agents:
            # Determine best role for this agent
            best_role = await self._determine_best_role_for_agent(agent, task)
            role_assignments.append((agent.agent_id, best_role))

        # Ensure role diversity
        role_assignments = await self._ensure_role_diversity(role_assignments, task)

        team_members = dict(role_assignments)

        return team_members

    async def _quick_assign_roles(
        self,
        agents: List[AgentInfo],
        task: SwarmTask
    ) -> Dict[str, AgentRole]:
        """Quick role assignment for rapid assembly."""

        team_members = {}
        available_roles = list(AgentRole)

        for i, agent in enumerate(agents):
            # Cycle through roles
            role = available_roles[i % len(available_roles)]
            team_members[agent.agent_id] = role

        return team_members

    async def _determine_best_role_for_agent(
        self,
        agent: AgentInfo,
        task: SwarmTask
    ) -> AgentRole:
        """Determine the best role for an agent given the task."""

        # Role scoring based on agent capabilities and task requirements
        role_scores = {}

        for role in AgentRole:
            score = 0.0

            # Base score from agent role
            if agent.role == role:
                score += 1.0

            # Capability matching
            role_capabilities = self._get_role_capabilities(role)
            capability_matches = len(role_capabilities.intersection(agent.capabilities))
            score += capability_matches * 0.5

            # Performance bonus
            score += agent.performance_score * 0.3

            role_scores[role] = score

        return max(role_scores.keys(), key=lambda r: role_scores[r])

    def _get_role_capabilities(self, role: AgentRole) -> Set[str]:
        """Get capabilities associated with a role."""

        role_capability_map = {
            AgentRole.RESEARCHER: {"research", "analysis", "investigation"},
            AgentRole.EXECUTOR: {"execution", "implementation", "action"},
            AgentRole.VALIDATOR: {"validation", "verification", "checking"},
            AgentRole.LEARNER: {"learning", "adaptation", "improvement"},
            AgentRole.COORDINATOR: {"coordination", "management", "planning"}
        }

        return role_capability_map.get(role, set())

    async def _ensure_role_diversity(
        self,
        role_assignments: List[Tuple[str, AgentRole]],
        task: SwarmTask
    ) -> List[Tuple[str, AgentRole]]:
        """Ensure the team has diverse roles."""

        assignments = dict(role_assignments)
        role_counts = {}

        for agent_id, role in assignments.items():
            role_counts[role] = role_counts.get(role, 0) + 1

        # Ensure minimum roles are present
        required_roles = {AgentRole.RESEARCHER, AgentRole.EXECUTOR, AgentRole.VALIDATOR}

        for required_role in required_roles:
            if role_counts.get(required_role, 0) == 0:
                # Reassign an agent to this role
                best_agent = None
                best_score = -1

                for agent_id, current_role in assignments.items():
                    if current_role != required_role:
                        # Find agent that can perform this role
                        agent = next((a for a in [] if a.agent_id == agent_id), None)  # Would need agent list
                        if agent:
                            score = len(self._get_role_capabilities(required_role).intersection(agent.capabilities))
                            if score > best_score:
                                best_score = score
                                best_agent = agent_id

                if best_agent:
                    assignments[best_agent] = required_role

        return list(assignments.items())

    async def _create_team_composition(
        self,
        team_members: Dict[str, AgentRole],
        task: SwarmTask,
        formation_method: str
    ) -> TeamComposition:
        """Create a team composition object."""

        if not team_members:
            return None

        # Select team leader (coordinator or highest performing agent)
        leader_agent = await self._select_team_leader(team_members)

        # Calculate team metrics
        synergy_score = await self._calculate_team_synergy(team_members)
        capability_coverage = await self._calculate_capability_coverage(team_members, task)
        performance_expectation = await self._calculate_performance_expectation(team_members)
        expected_duration = await self._estimate_team_duration(team_members, task)

        return TeamComposition(
            team_id=str(uuid.uuid4()),
            leader_agent=leader_agent,
            members=team_members,
            synergy_score=synergy_score,
            capability_coverage=capability_coverage,
            performance_expectation=performance_expectation,
            formation_time=datetime.now(),
            expected_duration=expected_duration
        )

    async def _select_team_leader(self, team_members: Dict[str, AgentRole]) -> str:
        """Select the team leader."""

        # Prefer coordinator role, then highest performing agent
        coordinator = None
        best_agent = None
        best_score = -1

        for agent_id, role in team_members.items():
            if role == AgentRole.COORDINATOR:
                return agent_id

            # Would need to get agent info here
            # For now, return first agent
            if best_agent is None:
                best_agent = agent_id

        return best_agent

    async def _calculate_team_synergy(self, team_members: Dict[str, AgentRole]) -> float:
        """Calculate how well team members work together."""

        # Simple synergy calculation based on role diversity
        roles = set(team_members.values())
        role_diversity = len(roles) / len(AgentRole)

        # Size efficiency (teams of 3-5 are most efficient)
        team_size = len(team_members)
        size_efficiency = 1.0 - abs(team_size - 4) * 0.2
        size_efficiency = max(0.0, min(1.0, size_efficiency))

        return (role_diversity * 0.6 + size_efficiency * 0.4)

    async def _calculate_capability_coverage(
        self,
        team_members: Dict[str, AgentRole],
        task: SwarmTask
    ) -> float:
        """Calculate capability coverage for the task."""

        # Would need agent capability data
        # For now, assume good coverage based on role assignment
        role_capabilities = set()
        for role in team_members.values():
            role_capabilities.update(self._get_role_capabilities(role))

        covered_capabilities = task.required_capabilities.intersection(role_capabilities)

        return len(covered_capabilities) / len(task.required_capabilities)

    async def _calculate_performance_expectation(self, team_members: Dict[str, AgentRole]) -> float:
        """Calculate expected team performance."""

        # Would need individual agent performance scores
        # For now, return synergy score as proxy
        synergy = await self._calculate_team_synergy(team_members)
        return synergy

    async def _estimate_team_duration(
        self,
        team_members: Dict[str, AgentRole],
        task: SwarmTask
    ) -> float:
        """Estimate team task completion duration."""

        base_duration = task.complexity * 3600  # hours to seconds

        # Team efficiency factors
        synergy_bonus = await self._calculate_team_synergy(team_members)
        size_penalty = 1.0 + (len(team_members) - 1) * 0.1  # coordination overhead

        return base_duration / synergy_bonus * size_penalty

    async def _generate_alternative_teams(
        self,
        request: TeamFormationRequest,
        primary_team: Optional[TeamComposition]
    ) -> List[TeamComposition]:
        """Generate alternative team compositions."""

        alternatives = []

        # Try different strategies
        strategies_to_try = [
            strategy for strategy in TeamFormationStrategy
            if strategy != request.strategy
        ][:2]  # Try 2 alternative strategies

        for strategy in strategies_to_try:
            try:
                alt_request = request.__dataclass_replace__(strategy=strategy)
                alternative_team = await self.form_team(
                    alt_request.task,
                    alt_request.available_agents,
                    strategy,
                    alt_request.max_team_size,
                    alt_request.constraints
                )

                if alternative_team.team:
                    alternatives.append(alternative_team.team)

            except Exception:
                continue

        return alternatives

    async def get_team_performance_history(self, team_id: str) -> Optional[Dict[str, Any]]:
        """Get performance history for a team."""

        # This would integrate with observability system
        # For now, return placeholder
        return {
            "team_id": team_id,
            "tasks_completed": 5,
            "average_performance": 0.85,
            "success_rate": 0.9
        }

    async def health_check(self) -> bool:
        """Check team formation engine health."""

        # Check if we can form basic teams
        try:
            # Simple test formation
            test_task = SwarmTask(
                task_id="health_check",
                goal="test",
                complexity=0.1,
                required_capabilities={"test"},
                max_agents=2,
                deadline=None,
                priority=1
            )

            test_agents = [
                AgentInfo(
                    agent_id="test_agent_1",
                    role=AgentRole.RESEARCHER,
                    status=AgentStatus.AVAILABLE,
                    capabilities={"research", "analysis"},
                    performance_score=0.8,
                    resource_usage={"cpu": 1.0, "memory": 2.0},
                    last_seen=datetime.now().timestamp(),
                    swarm_affinity=0.9
                )
            ]

            result = await self.form_team(test_task, test_agents)
            return result.team is not None

        except Exception:
            return False


# Global instance
dynamic_team_formation = DynamicTeamFormation()
