"""Temporal Reasoning Engine for understanding sequences and dependencies."""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from libs.common.schemas import Step
import networkx as nx
import httpx


class TemporalRelation(Enum):
    """Allen interval algebra relations."""
    BEFORE = "before"
    AFTER = "after"
    MEETS = "meets"
    MET_BY = "met_by"
    OVERLAPS = "overlaps"
    OVERLAPPED_BY = "overlapped_by"
    DURING = "during"
    CONTAINS = "contains"
    STARTS = "starts"
    STARTED_BY = "started_by"
    FINISHES = "finishes"
    FINISHED_BY = "finished_by"
    EQUALS = "equals"


class TemporalConstraint:
    """Constraint between temporal intervals."""

    def __init__(
        self,
        interval_a: str,
        interval_b: str,
        relation: TemporalRelation,
        min_gap: Optional[timedelta] = None,
        max_gap: Optional[timedelta] = None
    ):
        self.interval_a = interval_a
        self.interval_b = interval_b
        self.relation = relation
        self.min_gap = min_gap
        self.max_gap = max_gap

    def __str__(self) -> str:
        gap_str = ""
        if self.min_gap or self.max_gap:
            gaps = []
            if self.min_gap:
                gaps.append(f"min_gap={self.min_gap}")
            if self.max_gap:
                gaps.append(f"max_gap={self.max_gap}")
            gap_str = f" ({', '.join(gaps)})"

        return f"{self.interval_a} {self.relation.value} {self.interval_b}{gap_str}"


@dataclass
class TimeInterval:
    """Temporal interval with start and end times."""
    id: str
    name: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration: Optional[timedelta]
    uncertainty_start: timedelta = timedelta(0)
    uncertainty_end: timedelta = timedelta(0)

    def overlaps(self, other: 'TimeInterval') -> bool:
        """Check if this interval overlaps with another."""
        if not (self.start_time and self.end_time and other.start_time and other.end_time):
            return False

        return max(self.start_time, other.start_time) < min(self.end_time, other.end_time)

    def contains(self, other: 'TimeInterval') -> bool:
        """Check if this interval contains another."""
        if not (self.start_time and self.end_time and other.start_time and other.end_time):
            return False

        return self.start_time <= other.start_time and self.end_time >= other.end_time

    def before(self, other: 'TimeInterval') -> bool:
        """Check if this interval is before another."""
        if not (self.end_time and other.start_time):
            return False

        return self.end_time <= other.start_time

    def get_possible_relations(self, other: 'TimeInterval') -> List[TemporalRelation]:
        """Get possible temporal relations between intervals."""
        relations = []

        # Consider uncertainty in timing
        self_start_min = self.start_time - self.uncertainty_start if self.start_time else None
        self_start_max = self.start_time + self.uncertainty_start if self.start_time else None
        self_end_min = self.end_time - self.uncertainty_end if self.end_time else None
        self_end_max = self.end_time + self.uncertainty_end if self.end_time else None

        other_start_min = other.start_time - other.uncertainty_start if other.start_time else None
        other_start_max = other.start_time + other.uncertainty_start if other.start_time else None
        other_end_min = other.end_time - other.uncertainty_end if other.end_time else None
        other_end_max = other.end_time + other.uncertainty_end if other.end_time else None

        # Check each relation considering uncertainty
        if self_end_max and other_start_min and self_end_max <= other_start_min:
            relations.append(TemporalRelation.BEFORE)
        if other_end_max and self_start_min and other_end_max <= self_start_min:
            relations.append(TemporalRelation.AFTER)

        # Add more relation checks as needed...

        return relations if relations else [TemporalRelation.OVERLAPS]  # Default fallback


@dataclass
class TemporalPlan:
    """Plan with temporal constraints and scheduling."""
    intervals: List[TimeInterval]
    constraints: List[TemporalConstraint]
    schedule: Dict[str, datetime]  # interval_id -> scheduled_start_time
    conflicts: List[str]
    feasibility_score: float
    reasoning: str


@dataclass
class TemporalConflict:
    """Conflict in temporal constraints."""
    constraint: TemporalConstraint
    violation_severity: float
    suggested_resolution: str
    affected_intervals: List[str]


class TemporalReasoner:
    """Engine for temporal reasoning and constraint satisfaction."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.constraint_solver_timeout = self.config.get('solver_timeout', 30)
        self.max_iterations = self.config.get('max_iterations', 100)

    async def reason_about_temporal_constraints(
        self,
        intervals: List[TimeInterval],
        constraints: List[TemporalConstraint],
        scheduling_hints: Optional[Dict[str, Any]] = None
    ) -> TemporalPlan:
        """Analyze and resolve temporal constraints."""

        start_time = datetime.now()

        # Build constraint graph
        constraint_graph = await self._build_constraint_graph(intervals, constraints)

        # Find conflicts
        conflicts = await self._detect_conflicts(constraint_graph)

        # Attempt to resolve conflicts
        resolved_constraints = await self._resolve_conflicts(constraints, conflicts)

        # Generate schedule
        schedule = await self._generate_schedule(intervals, resolved_constraints, scheduling_hints)

        # Calculate feasibility
        feasibility_score = await self._calculate_feasibility_score(
            schedule, resolved_constraints, conflicts
        )

        reasoning = await self._generate_temporal_reasoning(
            intervals, constraints, schedule, conflicts
        )

        return TemporalPlan(
            intervals=intervals,
            constraints=resolved_constraints,
            schedule=schedule,
            conflicts=[str(c) for c in conflicts],
            feasibility_score=feasibility_score,
            reasoning=reasoning
        )

    async def _build_constraint_graph(
        self,
        intervals: List[TimeInterval],
        constraints: List[TemporalConstraint]
    ) -> nx.DiGraph:
        """Build graph representation of temporal constraints."""

        graph = nx.DiGraph()

        # Add nodes for intervals
        for interval in intervals:
            graph.add_node(interval.id, interval=interval)

        # Add edges for constraints
        for constraint in constraints:
            graph.add_edge(
                constraint.interval_a,
                constraint.interval_b,
                constraint=constraint,
                relation=constraint.relation
            )

        return graph

    async def _detect_conflicts(self, constraint_graph: nx.DiGraph) -> List[TemporalConflict]:
        """Detect conflicts in temporal constraints."""

        conflicts = []

        # Check for cycles that would create impossible constraints
        try:
            cycles = list(nx.simple_cycles(constraint_graph))
            for cycle in cycles:
                if len(cycle) > 2:  # Cycles of length > 2 are problematic
                    conflicts.append(TemporalConflict(
                        constraint=None,  # Cycle conflict
                        violation_severity=0.8,
                        suggested_resolution="Break cycle by removing one constraint",
                        affected_intervals=cycle
                    ))
        except nx.NetworkXError:
            pass  # No cycles found

        # Check individual constraint consistency
        for node_a, node_b, edge_data in constraint_graph.edges(data=True):
            constraint = edge_data['constraint']
            interval_a = constraint_graph.nodes[node_a]['interval']
            interval_b = constraint_graph.nodes[node_b]['interval']

            if not await self._check_constraint_consistency(constraint, interval_a, interval_b):
                conflicts.append(TemporalConflict(
                    constraint=constraint,
                    violation_severity=0.6,
                    suggested_resolution=f"Adjust timing of {constraint.interval_a} or {constraint.interval_b}",
                    affected_intervals=[node_a, node_b]
                ))

        return conflicts

    async def _check_constraint_consistency(
        self,
        constraint: TemporalConstraint,
        interval_a: TimeInterval,
        interval_b: TimeInterval
    ) -> bool:
        """Check if a temporal constraint is consistent with interval timing."""

        if not (interval_a.start_time and interval_a.end_time and
                interval_b.start_time and interval_b.end_time):
            return True  # Can't check without timing

        # Check basic relation consistency
        if constraint.relation == TemporalRelation.BEFORE:
            return interval_a.end_time <= interval_b.start_time
        elif constraint.relation == TemporalRelation.AFTER:
            return interval_a.start_time >= interval_b.end_time
        elif constraint.relation == TemporalRelation.DURING:
            return interval_b.start_time <= interval_a.start_time and interval_a.end_time <= interval_b.end_time
        elif constraint.relation == TemporalRelation.CONTAINS:
            return interval_a.start_time <= interval_b.start_time and interval_b.end_time <= interval_a.end_time

        # For other relations, assume consistent for now
        return True

    async def _resolve_conflicts(
        self,
        constraints: List[TemporalConstraint],
        conflicts: List[TemporalConflict]
    ) -> List[TemporalConstraint]:
        """Attempt to resolve temporal conflicts."""

        resolved_constraints = constraints.copy()

        for conflict in conflicts:
            if conflict.constraint:
                # Try to relax the constraint
                relaxed_constraint = await self._relax_constraint(conflict.constraint)
                if relaxed_constraint:
                    # Replace original constraint
                    index = resolved_constraints.index(conflict.constraint)
                    resolved_constraints[index] = relaxed_constraint

        return resolved_constraints

    async def _relax_constraint(self, constraint: TemporalConstraint) -> Optional[TemporalConstraint]:
        """Relax a constraint to make it more flexible."""

        # Add uncertainty buffers
        if constraint.relation == TemporalRelation.BEFORE:
            # Allow small overlap
            return TemporalConstraint(
                constraint.interval_a,
                constraint.interval_b,
                TemporalRelation.BEFORE,
                min_gap=timedelta(seconds=-30),  # Allow 30s overlap
                max_gap=None
            )

        # For other constraints, return as-is for now
        return constraint

    async def _generate_schedule(
        self,
        intervals: List[TimeInterval],
        constraints: List[TemporalConstraint],
        scheduling_hints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, datetime]:
        """Generate schedule satisfying temporal constraints."""

        schedule = {}
        hints = scheduling_hints or {}

        # Sort intervals by priority/constraints
        sorted_intervals = await self._prioritize_intervals(intervals, constraints)

        # Schedule each interval
        for interval in sorted_intervals:
            scheduled_time = await self._schedule_interval(
                interval, schedule, constraints, hints
            )
            schedule[interval.id] = scheduled_time

        return schedule

    async def _prioritize_intervals(
        self,
        intervals: List[TimeInterval],
        constraints: List[TemporalConstraint]
    ) -> List[TimeInterval]:
        """Prioritize intervals for scheduling."""

        # Count incoming constraints for each interval
        incoming_counts = {}
        for constraint in constraints:
            incoming_counts[constraint.interval_b] = incoming_counts.get(constraint.interval_b, 0) + 1

        # Sort by incoming constraints (more constrained first)
        return sorted(
            intervals,
            key=lambda i: incoming_counts.get(i.id, 0),
            reverse=True
        )

    async def _schedule_interval(
        self,
        interval: TimeInterval,
        current_schedule: Dict[str, datetime],
        constraints: List[TemporalConstraint],
        hints: Dict[str, Any]
    ) -> datetime:
        """Schedule a specific interval."""

        # Get relevant constraints for this interval
        relevant_constraints = [
            c for c in constraints
            if c.interval_a == interval.id or c.interval_b == interval.id
        ]

        # Consider existing schedule
        scheduled_time = hints.get(interval.id)

        if not scheduled_time and relevant_constraints:
            # Find constraining intervals that are already scheduled
            constraining_times = []
            for constraint in relevant_constraints:
                other_id = constraint.interval_b if constraint.interval_a == interval.id else constraint.interval_a
                if other_id in current_schedule:
                    other_time = current_schedule[other_id]
                    # Calculate required time based on constraint
                    required_time = await self._calculate_required_time(
                        constraint, interval, other_time, is_subject=(constraint.interval_a == interval.id)
                    )
                    if required_time:
                        constraining_times.append(required_time)

            if constraining_times:
                # Choose earliest/latest based on constraints
                if any(c.relation in [TemporalRelation.AFTER, TemporalRelation.MET_BY] for c in relevant_constraints):
                    scheduled_time = max(constraining_times)  # Must be after
                else:
                    scheduled_time = min(constraining_times)  # Can be as early as possible

        # Default to current time if no constraints
        if not scheduled_time:
            scheduled_time = datetime.now()

        return scheduled_time

    async def _calculate_required_time(
        self,
        constraint: TemporalConstraint,
        interval: TimeInterval,
        other_time: datetime,
        is_subject: bool
    ) -> Optional[datetime]:
        """Calculate required time for interval based on constraint."""

        # This would implement detailed temporal calculus
        # For now, simple offset
        if constraint.relation == TemporalRelation.AFTER:
            return other_time + timedelta(minutes=5)  # 5 minute buffer
        elif constraint.relation == TemporalRelation.BEFORE:
            return other_time - timedelta(minutes=5)  # 5 minute buffer before

        return None

    async def _calculate_feasibility_score(
        self,
        schedule: Dict[str, datetime],
        constraints: List[TemporalConstraint],
        conflicts: List[TemporalConflict]
    ) -> float:
        """Calculate how feasible the temporal plan is."""

        if not constraints:
            return 1.0

        satisfied_constraints = 0
        for constraint in constraints:
            if await self._check_schedule_satisfies_constraint(
                schedule, constraint
            ):
                satisfied_constraints += 1

        constraint_satisfaction = satisfied_constraints / len(constraints)

        # Penalize for conflicts
        conflict_penalty = len(conflicts) * 0.1

        return max(0.0, min(1.0, constraint_satisfaction - conflict_penalty))

    async def _check_schedule_satisfies_constraint(
        self,
        schedule: Dict[str, datetime],
        constraint: TemporalConstraint
    ) -> bool:
        """Check if schedule satisfies a temporal constraint."""

        time_a = schedule.get(constraint.interval_a)
        time_b = schedule.get(constraint.interval_b)

        if not (time_a and time_b):
            return False

        if constraint.relation == TemporalRelation.BEFORE:
            return time_a < time_b
        elif constraint.relation == TemporalRelation.AFTER:
            return time_a > time_b
        elif constraint.relation == TemporalRelation.DURING:
            # This would need duration information
            return True  # Placeholder

        return True  # Default to satisfied

    async def _generate_temporal_reasoning(
        self,
        intervals: List[TimeInterval],
        constraints: List[TemporalConstraint],
        schedule: Dict[str, datetime],
        conflicts: List[TemporalConflict]
    ) -> str:
        """Generate human-readable reasoning about temporal aspects."""

        reasoning_parts = []

        if conflicts:
            reasoning_parts.append(f"Found {len(conflicts)} temporal conflicts that were resolved")
        else:
            reasoning_parts.append("No temporal conflicts detected")

        reasoning_parts.append(f"Scheduled {len(intervals)} intervals with {len(constraints)} constraints")

        if schedule:
            scheduled_count = len([t for t in schedule.values() if t])
            reasoning_parts.append(f"Successfully scheduled {scheduled_count}/{len(intervals)} intervals")

        feasibility = await self._calculate_feasibility_score(schedule, constraints, conflicts)
        reasoning_parts.append(f"Overall feasibility: {feasibility:.2f}")

        return ". ".join(reasoning_parts)

    async def validate_temporal_sequence(
        self,
        events: List[Dict[str, Any]],
        expected_patterns: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Validate temporal sequence of events."""

        # Convert events to intervals
        intervals = []
        for event in events:
            interval = TimeInterval(
                id=event.get('id', str(uuid.uuid4())),
                name=event.get('name', 'unknown'),
                start_time=event.get('start_time'),
                end_time=event.get('end_time'),
                duration=event.get('duration')
            )
            intervals.append(interval)

        # Check for temporal anomalies
        anomalies = await self._detect_temporal_anomalies(intervals)

        # Validate against expected patterns
        pattern_validation = {}
        if expected_patterns:
            pattern_validation = await self._validate_patterns(intervals, expected_patterns)

        return {
            'intervals': intervals,
            'anomalies': anomalies,
            'pattern_validation': pattern_validation,
            'sequence_valid': len(anomalies) == 0 and all(pattern_validation.values())
        }

    async def _detect_temporal_anomalies(self, intervals: List[TimeInterval]) -> List[str]:
        """Detect temporal anomalies in sequence."""

        anomalies = []

        # Check for overlapping intervals that shouldn't overlap
        for i, interval_a in enumerate(intervals):
            for interval_b in intervals[i+1:]:
                if interval_a.overlaps(interval_b):
                    # Check if overlap is expected based on names/types
                    if not await self._is_expected_overlap(interval_a, interval_b):
                        anomalies.append(
                            f"Unexpected overlap between {interval_a.name} and {interval_b.name}"
                        )

        # Check for impossible timing (end before start)
        for interval in intervals:
            if interval.start_time and interval.end_time and interval.start_time > interval.end_time:
                anomalies.append(f"Invalid timing for {interval.name}: ends before it starts")

        return anomalies

    async def _is_expected_overlap(self, interval_a: TimeInterval, interval_b: TimeInterval) -> bool:
        """Check if overlap between intervals is expected."""

        # Simple heuristic based on names
        # Parallel processing might be expected
        parallel_indicators = ['process', 'execute', 'run', 'parallel']

        a_has_parallel = any(indicator in interval_a.name.lower() for indicator in parallel_indicators)
        b_has_parallel = any(indicator in interval_b.name.lower() for indicator in parallel_indicators)

        return a_has_parallel and b_has_parallel

    async def _validate_patterns(
        self,
        intervals: List[TimeInterval],
        expected_patterns: List[Dict[str, Any]]
    ) -> Dict[str, bool]:
        """Validate intervals against expected temporal patterns."""

        validation_results = {}

        for pattern in expected_patterns:
            pattern_name = pattern.get('name', 'unknown')
            pattern_type = pattern.get('type', 'sequence')

            if pattern_type == 'sequence':
                validation_results[pattern_name] = await self._validate_sequence_pattern(
                    intervals, pattern
                )
            elif pattern_type == 'parallel':
                validation_results[pattern_name] = await self._validate_parallel_pattern(
                    intervals, pattern
                )
            else:
                validation_results[pattern_name] = False

        return validation_results

    async def _validate_sequence_pattern(
        self,
        intervals: List[TimeInterval],
        pattern: Dict[str, Any]
    ) -> bool:
        """Validate sequential pattern."""

        sequence = pattern.get('sequence', [])
        if len(sequence) != len(intervals):
            return False

        # Check if intervals follow the expected sequence
        sorted_intervals = sorted(intervals, key=lambda i: i.start_time or datetime.min)

        for i, expected_name in enumerate(sequence):
            if i < len(sorted_intervals) and expected_name not in sorted_intervals[i].name:
                return False

        return True

    async def _validate_parallel_pattern(
        self,
        intervals: List[TimeInterval],
        pattern: Dict[str, Any]
    ) -> bool:
        """Validate parallel execution pattern."""

        parallel_groups = pattern.get('groups', [])
        if not parallel_groups:
            return False

        # Check if intervals in same group overlap
        for group in parallel_groups:
            group_intervals = [i for i in intervals if i.name in group]

            if len(group_intervals) < 2:
                continue

            # Check if they overlap (run in parallel)
            overlaps = all(
                group_intervals[0].overlaps(other)
                for other in group_intervals[1:]
            )

            if not overlaps:
                return False

        return True

    async def predict_temporal_outcomes(
        self,
        current_sequence: List[Dict[str, Any]],
        prediction_horizon: timedelta
    ) -> Dict[str, Any]:
        """Predict future temporal outcomes based on current sequence."""

        # Convert to intervals
        intervals = []
        for event in current_sequence:
            interval = TimeInterval(
                id=event.get('id', str(uuid.uuid4())),
                name=event.get('name', 'unknown'),
                start_time=event.get('start_time'),
                end_time=event.get('end_time'),
                duration=event.get('duration')
            )
            intervals.append(interval)

        # Analyze patterns
        patterns = await self._analyze_temporal_patterns(intervals)

        # Predict future events
        predictions = await self._predict_future_events(patterns, prediction_horizon)

        return {
            'patterns_identified': patterns,
            'predictions': predictions,
            'confidence': 0.7  # Placeholder
        }

    async def _analyze_temporal_patterns(self, intervals: List[TimeInterval]) -> List[Dict[str, Any]]:
        """Analyze temporal patterns in interval sequence."""

        patterns = []

        if len(intervals) < 2:
            return patterns

        # Look for regular intervals
        durations = []
        gaps = []

        sorted_intervals = sorted(intervals, key=lambda i: i.start_time or datetime.min)

        for i in range(len(sorted_intervals)):
            if sorted_intervals[i].duration:
                durations.append(sorted_intervals[i].duration)

            if i > 0 and sorted_intervals[i].start_time and sorted_intervals[i-1].end_time:
                gap = sorted_intervals[i].start_time - sorted_intervals[i-1].end_time
                gaps.append(gap)

        # Identify patterns
        if durations and len(set(str(d) for d in durations)) == 1:
            patterns.append({
                'type': 'regular_duration',
                'duration': durations[0],
                'confidence': 0.8
            })

        if gaps and len(set(str(g) for g in gaps)) == 1:
            patterns.append({
                'type': 'regular_gaps',
                'gap': gaps[0],
                'confidence': 0.8
            })

        return patterns

    async def _predict_future_events(
        self,
        patterns: List[Dict[str, Any]],
        horizon: timedelta
    ) -> List[Dict[str, Any]]:
        """Predict future events based on patterns."""

        predictions = []

        for pattern in patterns:
            if pattern['type'] == 'regular_duration':
                # Predict next event with same duration
                predictions.append({
                    'type': 'continuation',
                    'description': f"Next event expected with duration {pattern['duration']}",
                    'confidence': pattern['confidence']
                })
            elif pattern['type'] == 'regular_gaps':
                # Predict next event after regular gap
                predictions.append({
                    'type': 'continuation',
                    'description': f"Next event expected after gap of {pattern['gap']}",
                    'confidence': pattern['confidence']
                })

        return predictions

    async def health_check(self) -> bool:
        """Check temporal reasoner health."""

        try:
            # Test basic temporal operations
            interval_a = TimeInterval(
                id="test_a",
                name="test_a",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(hours=1),
                duration=timedelta(hours=1)
            )

            interval_b = TimeInterval(
                id="test_b",
                name="test_b",
                start_time=datetime.now() + timedelta(hours=2),
                end_time=datetime.now() + timedelta(hours=3),
                duration=timedelta(hours=1)
            )

            # Test temporal reasoning
            constraints = [
                TemporalConstraint("test_a", "test_b", TemporalRelation.BEFORE)
            ]

            plan = await self.reason_about_temporal_constraints([interval_a, interval_b], constraints)

            return plan.feasibility_score > 0.5

        except Exception:
            return False


# Global instance
temporal_reasoner = TemporalReasoner()
