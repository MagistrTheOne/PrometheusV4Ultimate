"""Negotiation Engine for multi-agent resource and priority coordination."""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from libs.common.config import settings
import httpx


class NegotiationStatus(Enum):
    """Status of a negotiation session."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class NegotiationOutcome(Enum):
    """Possible outcomes of a negotiation."""
    AGREEMENT = "agreement"
    COMPROMISE = "compromise"
    DEADLOCK = "deadlock"
    WITHDRAWAL = "withdrawal"


@dataclass
class ResourceRequirements:
    """Resource requirements for negotiation."""
    agent_id: str
    cpu_cores: float
    gpu_memory_gb: float
    ram_gb: float
    network_bandwidth_mbps: float
    priority: int  # 1-10
    duration_hours: float
    flexibility: float  # 0-1, how flexible are the requirements


@dataclass
class NegotiationProposal:
    """Proposal in a negotiation."""
    proposal_id: str
    proposer_id: str
    resource_allocation: Dict[str, float]
    conditions: List[str]
    timestamp: datetime
    expires_at: datetime


@dataclass
class NegotiationSession:
    """A negotiation session between agents."""
    session_id: str
    participants: List[str]
    resource_type: str
    total_available: float
    status: NegotiationStatus
    proposals: List[NegotiationProposal]
    agreements: Dict[str, Dict[str, float]]
    created_at: datetime
    deadline: datetime


@dataclass
class NegotiationResult:
    """Result of a negotiation session."""
    session_id: str
    outcome: NegotiationOutcome
    final_allocation: Dict[str, Dict[str, float]]
    participants: List[str]
    rounds_conducted: int
    time_taken_seconds: float
    consensus_reached: bool


class NegotiationEngine:
    """Engine for coordinating resource allocation through agent negotiation."""

    def __init__(self):
        self.active_sessions: Dict[str, NegotiationSession] = {}
        self.negotiation_timeout = 300  # 5 minutes default
        self.max_negotiation_rounds = 10
        self.consensus_threshold = 0.8  # 80% agreement required

    async def negotiate_resources(
        self,
        requirements: List[ResourceRequirements],
        resource_type: str,
        total_available: float
    ) -> NegotiationResult:
        """Conduct negotiation for resource allocation."""

        # Create negotiation session
        session = await self._create_negotiation_session(
            requirements, resource_type, total_available
        )

        try:
            # Conduct negotiation rounds
            result = await self._conduct_negotiation(session)

            # Clean up session
            await self._cleanup_session(session.session_id)

            return result

        except Exception as e:
            # Handle negotiation failure
            await self._handle_negotiation_failure(session.session_id, str(e))
            return NegotiationResult(
                session_id=session.session_id,
                outcome=NegotiationOutcome.DEADLOCK,
                final_allocation={},
                participants=session.participants,
                rounds_conducted=0,
                time_taken_seconds=0.0,
                consensus_reached=False
            )

    async def _create_negotiation_session(
        self,
        requirements: List[ResourceRequirements],
        resource_type: str,
        total_available: float
    ) -> NegotiationSession:
        """Create a new negotiation session."""

        session_id = str(uuid.uuid4())
        participants = [req.agent_id for req in requirements]

        session = NegotiationSession(
            session_id=session_id,
            participants=participants,
            resource_type=resource_type,
            total_available=total_available,
            status=NegotiationStatus.PENDING,
            proposals=[],
            agreements={},
            created_at=datetime.now(),
            deadline=datetime.now() + timedelta(seconds=self.negotiation_timeout)
        )

        self.active_sessions[session_id] = session

        # Notify participants
        await self._notify_participants_of_negotiation(session, requirements)

        return session

    async def _notify_participants_of_negotiation(
        self,
        session: NegotiationSession,
        requirements: List[ResourceRequirements]
    ) -> None:
        """Notify all participants about the negotiation."""

        notification_tasks = []

        for requirement in requirements:
            notification = {
                "type": "negotiation_invitation",
                "session_id": session.session_id,
                "resource_type": session.resource_type,
                "total_available": session.total_available,
                "your_requirements": {
                    "cpu_cores": requirement.cpu_cores,
                    "gpu_memory_gb": requirement.gpu_memory_gb,
                    "ram_gb": requirement.ram_gb,
                    "network_bandwidth_mbps": requirement.network_bandwidth_mbps,
                    "priority": requirement.priority,
                    "duration_hours": requirement.duration_hours,
                    "flexibility": requirement.flexibility
                },
                "deadline": session.deadline.isoformat()
            }

            notification_tasks.append(
                self._send_agent_notification(requirement.agent_id, notification)
            )

        await asyncio.gather(*notification_tasks, return_exceptions=True)

    async def _conduct_negotiation(self, session: NegotiationSession) -> NegotiationResult:
        """Conduct the negotiation process."""

        session.status = NegotiationStatus.ACTIVE
        start_time = datetime.now()

        for round_num in range(self.max_negotiation_rounds):
            # Check timeout
            if datetime.now() > session.deadline:
                session.status = NegotiationStatus.TIMEOUT
                break

            # Conduct negotiation round
            round_complete = await self._conduct_negotiation_round(session, round_num)

            if round_complete:
                # Check for consensus
                consensus_reached = await self._check_consensus(session)

                if consensus_reached:
                    session.status = NegotiationStatus.COMPLETED
                    break

            # Small delay between rounds
            await asyncio.sleep(1.0)

        # Determine final outcome
        outcome = await self._determine_negotiation_outcome(session)
        final_allocation = await self._create_final_allocation(session, outcome)

        time_taken = (datetime.now() - start_time).total_seconds()

        return NegotiationResult(
            session_id=session.session_id,
            outcome=outcome,
            final_allocation=final_allocation,
            participants=session.participants,
            rounds_conducted=len(session.proposals) // len(session.participants),  # Rough estimate
            time_taken_seconds=time_taken,
            consensus_reached=(outcome == NegotiationOutcome.AGREEMENT)
        )

    async def _conduct_negotiation_round(
        self,
        session: NegotiationSession,
        round_num: int
    ) -> bool:
        """Conduct a single round of negotiation."""

        # Collect proposals from all participants
        proposal_tasks = []
        for participant in session.participants:
            proposal_tasks.append(
                self._collect_proposal_from_agent(participant, session, round_num)
            )

        proposals = await asyncio.gather(*proposal_tasks, return_exceptions=True)

        # Filter out exceptions and None proposals
        valid_proposals = [
            proposal for proposal in proposals
            if proposal is not None and not isinstance(proposal, Exception)
        ]

        # Add proposals to session
        session.proposals.extend(valid_proposals)

        # Evaluate proposals and update agreements
        await self._evaluate_proposals(session, valid_proposals)

        return len(valid_proposals) == len(session.participants)

    async def _collect_proposal_from_agent(
        self,
        agent_id: str,
        session: NegotiationSession,
        round_num: int
    ) -> Optional[NegotiationProposal]:
        """Collect a proposal from a specific agent."""

        try:
            # This would be an HTTP call to the agent's negotiation endpoint
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"http://agent-registry:8005/agents/{agent_id}/negotiate",
                    json={
                        "session_id": session.session_id,
                        "resource_type": session.resource_type,
                        "total_available": session.total_available,
                        "round_num": round_num,
                        "current_proposals": [
                            {
                                "proposer_id": p.proposer_id,
                                "resource_allocation": p.resource_allocation,
                                "conditions": p.conditions
                            }
                            for p in session.proposals[-len(session.participants):]  # Last round proposals
                        ]
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    return NegotiationProposal(
                        proposal_id=str(uuid.uuid4()),
                        proposer_id=agent_id,
                        resource_allocation=data["resource_allocation"],
                        conditions=data.get("conditions", []),
                        timestamp=datetime.now(),
                        expires_at=datetime.now() + timedelta(seconds=30)
                    )

        except Exception as e:
            # Agent unavailable or error
            print(f"Failed to get proposal from agent {agent_id}: {e}")

        return None

    async def _evaluate_proposals(
        self,
        session: NegotiationSession,
        proposals: List[NegotiationProposal]
    ) -> None:
        """Evaluate proposals and update participant agreements."""

        # Simple evaluation: check if proposals are feasible
        for proposal in proposals:
            is_feasible = self._check_proposal_feasibility(
                proposal, session.total_available
            )

            if is_feasible:
                # Update agreements for this proposer
                session.agreements[proposal.proposer_id] = proposal.resource_allocation

    def _check_proposal_feasibility(
        self,
        proposal: NegotiationProposal,
        total_available: float
    ) -> bool:
        """Check if a proposal is feasible given resource constraints."""

        total_requested = sum(proposal.resource_allocation.values())

        # Allow some over-subscription for negotiation flexibility
        return total_requested <= total_available * 1.2

    async def _check_consensus(self, session: NegotiationSession) -> bool:
        """Check if consensus has been reached among participants."""

        if len(session.agreements) < len(session.participants):
            return False

        # Check agreement similarity
        agreement_vectors = list(session.agreements.values())

        if len(agreement_vectors) < 2:
            return False

        # Calculate pairwise agreement scores
        agreement_scores = []
        for i, vec_a in enumerate(agreement_vectors):
            for vec_b in agreement_vectors[i+1:]:
                score = self._calculate_agreement_score(vec_a, vec_b)
                agreement_scores.append(score)

        # Consensus reached if average agreement > threshold
        avg_agreement = sum(agreement_scores) / len(agreement_scores)

        return avg_agreement >= self.consensus_threshold

    def _calculate_agreement_score(
        self,
        allocation_a: Dict[str, float],
        allocation_b: Dict[str, float]
    ) -> float:
        """Calculate agreement score between two resource allocations."""

        all_resources = set(allocation_a.keys()) | set(allocation_b.keys())

        differences = []
        for resource in all_resources:
            val_a = allocation_a.get(resource, 0.0)
            val_b = allocation_b.get(resource, 0.0)

            if val_a + val_b > 0:  # Avoid division by zero
                # Normalized difference
                diff = abs(val_a - val_b) / (val_a + val_b)
                differences.append(diff)

        if not differences:
            return 1.0  # Perfect agreement if no resources

        # Return 1 - average normalized difference
        return 1.0 - (sum(differences) / len(differences))

    async def _determine_negotiation_outcome(self, session: NegotiationSession) -> NegotiationOutcome:
        """Determine the outcome of the negotiation."""

        if session.status == NegotiationStatus.COMPLETED:
            return NegotiationOutcome.AGREEMENT
        elif session.status == NegotiationStatus.TIMEOUT:
            return NegotiationOutcome.DEADLOCK
        elif len(session.agreements) == 0:
            return NegotiationOutcome.WITHDRAWAL
        else:
            # Some agreements but no consensus
            return NegotiationOutcome.COMPROMISE

    async def _create_final_allocation(
        self,
        session: NegotiationSession,
        outcome: NegotiationOutcome
    ) -> Dict[str, Dict[str, float]]:
        """Create final resource allocation based on negotiation outcome."""

        if outcome == NegotiationOutcome.AGREEMENT:
            # Use agreed allocations
            return session.agreements
        elif outcome == NegotiationOutcome.COMPROMISE:
            # Create compromise allocation
            return await self._create_compromise_allocation(session)
        else:
            # No allocation for deadlock/withdrawal
            return {}

    async def _create_compromise_allocation(self, session: NegotiationSession) -> Dict[str, Dict[str, float]]:
        """Create a compromise allocation when consensus isn't perfect."""

        # Simple compromise: average of all agreements
        all_resources = set()
        for allocation in session.agreements.values():
            all_resources.update(allocation.keys())

        compromise = {}
        for participant in session.participants:
            if participant in session.agreements:
                compromise[participant] = session.agreements[participant].copy()
            else:
                # Assign minimal allocation for non-agreeing participants
                compromise[participant] = {resource: 0.0 for resource in all_resources}

        return compromise

    async def _send_agent_notification(self, agent_id: str, notification: Dict[str, Any]) -> None:
        """Send notification to an agent."""

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"http://agent-registry:8005/agents/{agent_id}/notify",
                    json=notification
                )
                response.raise_for_status()
        except Exception:
            # Agent may be offline
            pass

    async def _cleanup_session(self, session_id: str) -> None:
        """Clean up a completed negotiation session."""

        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

    async def _handle_negotiation_failure(self, session_id: str, error: str) -> None:
        """Handle negotiation failure."""

        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.status = NegotiationStatus.FAILED

            # Notify participants of failure
            failure_notification = {
                "type": "negotiation_failed",
                "session_id": session_id,
                "error": error
            }

            notification_tasks = [
                self._send_agent_notification(agent_id, failure_notification)
                for agent_id in session.participants
            ]

            await asyncio.gather(*notification_tasks, return_exceptions=True)

            # Clean up
            await self._cleanup_session(session_id)

    async def get_negotiation_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a negotiation session."""

        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]

        return {
            "session_id": session.session_id,
            "status": session.status.value,
            "participants": session.participants,
            "resource_type": session.resource_type,
            "proposals_count": len(session.proposals),
            "agreements_count": len(session.agreements),
            "created_at": session.created_at.isoformat(),
            "deadline": session.deadline.isoformat(),
            "time_remaining_seconds": max(0, (session.deadline - datetime.now()).total_seconds())
        }

    async def force_negotiation_conclusion(self, session_id: str) -> Optional[NegotiationResult]:
        """Force conclusion of a negotiation session."""

        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]

        # Force compromise outcome
        outcome = NegotiationOutcome.COMPROMISE
        final_allocation = await self._create_compromise_allocation(session)

        time_taken = (datetime.now() - session.created_at).total_seconds()

        result = NegotiationResult(
            session_id=session_id,
            outcome=outcome,
            final_allocation=final_allocation,
            participants=session.participants,
            rounds_conducted=len(session.proposals) // max(len(session.participants), 1),
            time_taken_seconds=time_taken,
            consensus_reached=False  # Forced conclusion
        )

        # Clean up
        await self._cleanup_session(session_id)

        return result

    async def health_check(self) -> bool:
        """Check negotiation engine health."""

        # Check for stuck negotiations
        current_time = datetime.now()
        stuck_sessions = [
            session_id for session_id, session in self.active_sessions.items()
            if current_time > session.deadline
        ]

        # Clean up stuck sessions
        for session_id in stuck_sessions:
            await self._handle_negotiation_failure(session_id, "timeout")

        return True


# Global instance
negotiation_engine = NegotiationEngine()
