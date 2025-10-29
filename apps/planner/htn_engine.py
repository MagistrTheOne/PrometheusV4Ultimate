"""Hierarchical Task Networks (HTN) Engine for complex task decomposition."""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from libs.common.schemas import Step
import httpx


class HTNTaskType(Enum):
    """Types of tasks in HTN."""
    PRIMITIVE = "primitive"  # Directly executable
    COMPOUND = "compound"    # Decomposable into subtasks


class HTNStatus(Enum):
    """Status of HTN planning."""
    PLANNING = "planning"
    DECOMPOSING = "decomposing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class HTNCondition:
    """Precondition or effect in HTN."""
    predicate: str
    parameters: Dict[str, Any]
    negated: bool = False

    def __str__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        pred_str = f"{self.predicate}({params_str})"
        return f"¬{pred_str}" if self.negated else pred_str


@dataclass
class HTNTask:
    """Task in Hierarchical Task Network."""
    id: str
    name: str
    type: HTNTaskType
    preconditions: List[HTNCondition]
    effects: List[HTNCondition]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]

    def is_primitive(self) -> bool:
        return self.type == HTNTaskType.PRIMITIVE

    def is_compound(self) -> bool:
        return self.type == HTNTaskType.COMPOUND


@dataclass
class HTNMethod:
    """Method for decomposing compound tasks."""
    id: str
    name: str
    task_name: str  # Which task this method decomposes
    preconditions: List[HTNCondition]
    subtasks: List[HTNTask]
    ordering_constraints: List[Tuple[str, str]]  # (before_task_id, after_task_id)
    variable_constraints: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class HTNState:
    """World state for HTN planning."""
    facts: Set[str]  # Current state facts
    variable_bindings: Dict[str, Any]  # Variable assignments
    timestamp: datetime

    def satisfies(self, condition: HTNCondition) -> bool:
        """Check if state satisfies a condition."""
        # Simple implementation - check if fact exists
        fact_str = str(condition)
        if condition.negated:
            return fact_str not in self.facts
        else:
            return fact_str in self.facts

    def apply_effects(self, effects: List[HTNCondition]) -> 'HTNState':
        """Apply effects to create new state."""
        new_facts = self.facts.copy()

        for effect in effects:
            fact_str = str(effect)
            if effect.negated:
                new_facts.discard(fact_str.replace("¬", ""))
            else:
                new_facts.add(fact_str)

        return HTNState(
            facts=new_facts,
            variable_bindings=self.variable_bindings.copy(),
            timestamp=datetime.now()
        )


@dataclass
class HTNPlan:
    """Generated HTN plan."""
    id: str
    root_task: HTNTask
    tasks: List[HTNTask]
    methods_used: List[HTNMethod]
    ordering_constraints: List[Tuple[str, str]]
    variable_bindings: Dict[str, Any]
    status: HTNStatus
    decomposition_depth: int
    estimated_complexity: float
    created_at: datetime


@dataclass
class DecompositionResult:
    """Result of task decomposition."""
    success: bool
    subtasks: List[HTNTask]
    method_used: Optional[HTNMethod]
    new_constraints: List[Tuple[str, str]]
    reasoning: str
    confidence: float


class HTNDomain:
    """HTN domain definition."""

    def __init__(self, name: str):
        self.name = name
        self.tasks: Dict[str, HTNTask] = {}
        self.methods: Dict[str, List[HTNMethod]] = {}
        self.operators: Dict[str, HTNTask] = {}  # Primitive tasks

    def add_task(self, task: HTNTask) -> None:
        """Add a task to the domain."""
        self.tasks[task.name] = task
        if task.is_primitive():
            self.operators[task.name] = task

    def add_method(self, method: HTNMethod) -> None:
        """Add a method to the domain."""
        if method.task_name not in self.methods:
            self.methods[method.task_name] = []
        self.methods[method.task_name].append(method)

    def get_task(self, name: str) -> Optional[HTNTask]:
        """Get task by name."""
        return self.tasks.get(name)

    def get_methods(self, task_name: str) -> List[HTNMethod]:
        """Get methods for decomposing a task."""
        return self.methods.get(task_name, [])

    def get_applicable_methods(self, task: HTNTask, state: HTNState) -> List[HTNMethod]:
        """Get methods applicable to a task in current state."""
        methods = self.get_methods(task.name)
        applicable = []

        for method in methods:
            if self._method_applicable(method, task, state):
                applicable.append(method)

        return applicable

    def _method_applicable(self, method: HTNMethod, task: HTNTask, state: HTNState) -> bool:
        """Check if method is applicable."""
        # Check method preconditions
        for precondition in method.preconditions:
            if not state.satisfies(precondition):
                return False

        # Check task preconditions are satisfied
        for precondition in task.preconditions:
            if not state.satisfies(precondition):
                return False

        return True


class HTNPlanner:
    """HTN planning engine."""

    def __init__(self, domain: HTNDomain, config: Optional[Dict[str, Any]] = None):
        self.domain = domain
        self.config = config or {}
        self.max_depth = self.config.get('max_depth', 10)
        self.max_tasks = self.config.get('max_tasks', 100)
        self.timeout_seconds = self.config.get('timeout_seconds', 300)

    async def plan(
        self,
        root_task: HTNTask,
        initial_state: HTNState
    ) -> Optional[HTNPlan]:
        """Generate HTN plan for root task."""

        start_time = datetime.now()
        plan_id = str(uuid.uuid4())

        try:
            # Initialize planning context
            context = PlanningContext(
                plan_id=plan_id,
                root_task=root_task,
                initial_state=initial_state,
                current_state=initial_state,
                tasks=[],
                methods_used=[],
                ordering_constraints=[],
                variable_bindings={},
                depth=0
            )

            # Perform planning
            success = await self._plan_task(context, root_task)

            if success:
                # Create final plan
                plan = HTNPlan(
                    id=plan_id,
                    root_task=root_task,
                    tasks=context.tasks,
                    methods_used=context.methods_used,
                    ordering_constraints=context.ordering_constraints,
                    variable_bindings=context.variable_bindings,
                    status=HTNStatus.COMPLETED,
                    decomposition_depth=context.depth,
                    estimated_complexity=self._estimate_complexity(context),
                    created_at=start_time
                )

                # Validate plan
                if await self._validate_plan(plan, initial_state):
                    return plan

            return None

        except Exception as e:
            # Create failed plan
            return HTNPlan(
                id=plan_id,
                root_task=root_task,
                tasks=[],
                methods_used=[],
                ordering_constraints=[],
                variable_bindings={},
                status=HTNStatus.FAILED,
                decomposition_depth=0,
                estimated_complexity=0.0,
                created_at=start_time
            )

    async def _plan_task(self, context: 'PlanningContext', task: HTNTask) -> bool:
        """Plan a single task."""

        # Check depth limit
        if context.depth >= self.max_depth:
            return False

        # Check task limit
        if len(context.tasks) >= self.max_tasks:
            return False

        # Check timeout
        if (datetime.now() - context.root_task_created_at).total_seconds() > self.timeout_seconds:
            return False

        context.depth += 1

        if task.is_primitive():
            # Primitive task - just add it
            context.tasks.append(task)
            context.depth -= 1
            return True

        # Compound task - find applicable method
        applicable_methods = self.domain.get_applicable_methods(task, context.current_state)

        if not applicable_methods:
            context.depth -= 1
            return False

        # Try each method
        for method in applicable_methods:
            if await self._try_method(context, method, task):
                context.depth -= 1
                return True

        context.depth -= 1
        return False

    async def _try_method(
        self,
        context: 'PlanningContext',
        method: HTNMethod,
        task: HTNTask
    ) -> bool:
        """Try to apply a method for task decomposition."""

        # Save context for backtracking
        saved_tasks = context.tasks.copy()
        saved_methods = context.methods_used.copy()
        saved_constraints = context.ordering_constraints.copy()
        saved_state = context.current_state

        try:
            # Apply method
            context.methods_used.append(method)

            # Add ordering constraints from method
            context.ordering_constraints.extend(method.ordering_constraints)

            # Plan each subtask
            for subtask in method.subtasks:
                if not await self._plan_task(context, subtask):
                    # Failed - backtrack
                    context.tasks = saved_tasks
                    context.methods_used = saved_methods
                    context.ordering_constraints = saved_constraints
                    context.current_state = saved_state
                    return False

                # Update state with subtask effects
                context.current_state = context.current_state.apply_effects(subtask.effects)

            return True

        except Exception:
            # Backtrack on any error
            context.tasks = saved_tasks
            context.methods_used = saved_methods
            context.ordering_constraints = saved_constraints
            context.current_state = saved_state
            return False

    async def _validate_plan(self, plan: HTNPlan, initial_state: HTNState) -> bool:
        """Validate that plan achieves the goal."""

        try:
            # Simulate plan execution
            current_state = initial_state

            # Sort tasks by ordering constraints (topological sort)
            ordered_tasks = await self._order_tasks(plan.tasks, plan.ordering_constraints)

            # Execute tasks in order
            for task in ordered_tasks:
                # Check preconditions
                for precondition in task.preconditions:
                    if not current_state.satisfies(precondition):
                        return False

                # Apply effects
                current_state = current_state.apply_effects(task.effects)

            # Check if root task effects are satisfied
            for effect in plan.root_task.effects:
                if not current_state.satisfies(effect):
                    return False

            return True

        except Exception:
            return False

    async def _order_tasks(
        self,
        tasks: List[HTNTask],
        constraints: List[Tuple[str, str]]
    ) -> List[HTNTask]:
        """Order tasks according to constraints using topological sort."""

        # Create task map
        task_map = {task.id: task for task in tasks}

        # Build dependency graph
        graph = {task.id: [] for task in tasks}
        in_degree = {task.id: 0 for task in tasks}

        for before_id, after_id in constraints:
            if before_id in graph and after_id in graph:
                graph[before_id].append(after_id)
                in_degree[after_id] += 1

        # Topological sort using Kahn's algorithm
        result = []
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]

        while queue:
            current = queue.pop(0)
            result.append(task_map[current])

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(result) != len(tasks):
            raise ValueError("Circular dependency in task ordering")

        return result

    def _estimate_complexity(self, context: 'PlanningContext') -> float:
        """Estimate plan complexity."""

        # Factors: depth, number of tasks, number of methods
        depth_factor = context.depth / self.max_depth
        task_factor = len(context.tasks) / self.max_tasks
        method_factor = len(context.methods_used) / 10  # Assume 10 methods is complex

        return (depth_factor + task_factor + method_factor) / 3.0


@dataclass
class PlanningContext:
    """Context for HTN planning."""
    plan_id: str
    root_task: HTNTask
    initial_state: HTNState
    current_state: HTNState
    tasks: List[HTNTask]
    methods_used: List[HTNMethod]
    ordering_constraints: List[Tuple[str, str]]
    variable_bindings: Dict[str, Any]
    depth: int

    @property
    def root_task_created_at(self) -> datetime:
        """Get creation time from root task (placeholder)."""
        return datetime.now()  # Would be stored in actual implementation


class HTNReasoningEngine:
    """Enhanced reasoning for HTN planning."""

    def __init__(self, domain: HTNDomain):
        self.domain = domain

    async def find_optimal_decomposition(
        self,
        task: HTNTask,
        state: HTNState,
        constraints: Dict[str, Any]
    ) -> Optional[DecompositionResult]:
        """Find optimal decomposition for a compound task."""

        applicable_methods = self.domain.get_applicable_methods(task, state)

        if not applicable_methods:
            return DecompositionResult(
                success=False,
                subtasks=[],
                method_used=None,
                new_constraints=[],
                reasoning="No applicable methods found",
                confidence=0.0
            )

        # Evaluate each method
        method_evaluations = []
        for method in applicable_methods:
            evaluation = await self._evaluate_method(method, task, state, constraints)
            method_evaluations.append((method, evaluation))

        # Select best method
        best_method, best_evaluation = max(method_evaluations, key=lambda x: x[1]['score'])

        return DecompositionResult(
            success=True,
            subtasks=best_method.subtasks,
            method_used=best_method,
            new_constraints=best_method.ordering_constraints,
            reasoning=best_evaluation['reasoning'],
            confidence=best_evaluation['confidence']
        )

    async def _evaluate_method(
        self,
        method: HTNMethod,
        task: HTNTask,
        state: HTNState,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a decomposition method."""

        score = 0.0
        reasoning_parts = []

        # Factor 1: Method specificity (how well it matches task)
        specificity = len(method.preconditions) / max(1, len(task.preconditions))
        score += specificity * 0.3
        reasoning_parts.append(f"specificity: {specificity:.2f}")

        # Factor 2: Subtask count (prefer balanced decomposition)
        subtask_count = len(method.subtasks)
        optimal_count = constraints.get('optimal_subtasks', 3)
        count_penalty = abs(subtask_count - optimal_count) / optimal_count
        score += (1 - count_penalty) * 0.3
        reasoning_parts.append(f"subtask_count: {subtask_count} (optimal: {optimal_count})")

        # Factor 3: Constraint complexity
        constraint_complexity = len(method.ordering_constraints) / max(1, subtask_count)
        score += (1 - constraint_complexity) * 0.2  # Prefer fewer constraints
        reasoning_parts.append(f"constraint_complexity: {constraint_complexity:.2f}")

        # Factor 4: Effect coverage (how well method achieves task effects)
        effect_coverage = await self._calculate_effect_coverage(method, task)
        score += effect_coverage * 0.2
        reasoning_parts.append(f"effect_coverage: {effect_coverage:.2f}")

        return {
            'score': score,
            'reasoning': ", ".join(reasoning_parts),
            'confidence': min(1.0, score + 0.2)  # Add small confidence boost
        }

    async def _calculate_effect_coverage(self, method: HTNMethod, task: HTNTask) -> float:
        """Calculate how well method effects cover task effects."""

        if not task.effects:
            return 1.0

        covered_effects = 0

        # Check if method subtasks collectively achieve task effects
        subtask_effects = set()
        for subtask in method.subtasks:
            for effect in subtask.effects:
                subtask_effects.add(str(effect))

        for task_effect in task.effects:
            if str(task_effect) in subtask_effects:
                covered_effects += 1

        return covered_effects / len(task.effects)


class HTNEngine:
    """Main HTN Engine combining planning and reasoning."""

    def __init__(self, domain: HTNDomain, config: Optional[Dict[str, Any]] = None):
        self.domain = domain
        self.planner = HTNPlanner(domain, config)
        self.reasoning_engine = HTNReasoningEngine(domain)
        self.config = config or {}

    async def decompose_task(
        self,
        task: HTNTask,
        state: HTNState,
        constraints: Optional[Dict[str, Any]] = None
    ) -> DecompositionResult:
        """Decompose a compound task using HTN."""

        constraints = constraints or {}

        # Try optimal decomposition first
        optimal_result = await self.reasoning_engine.find_optimal_decomposition(
            task, state, constraints
        )

        if optimal_result.success:
            return optimal_result

        # Fallback: try any applicable method
        applicable_methods = self.domain.get_applicable_methods(task, state)

        if applicable_methods:
            method = applicable_methods[0]  # Take first available
            return DecompositionResult(
                success=True,
                subtasks=method.subtasks,
                method_used=method,
                new_constraints=method.ordering_constraints,
                reasoning="Fallback to first applicable method",
                confidence=0.5
            )

        return DecompositionResult(
            success=False,
            subtasks=[],
            method_used=None,
            new_constraints=[],
            reasoning="No decomposition methods available",
            confidence=0.0
        )

    async def create_plan(
        self,
        root_task: HTNTask,
        initial_state: HTNState
    ) -> Optional[HTNPlan]:
        """Create complete HTN plan."""

        return await self.planner.plan(root_task, initial_state)

    async def validate_plan(self, plan: HTNPlan, initial_state: HTNState) -> bool:
        """Validate HTN plan."""

        return await self.planner._validate_plan(plan, initial_state)

    async def optimize_plan(self, plan: HTNPlan, optimization_criteria: Dict[str, Any]) -> HTNPlan:
        """Optimize existing HTN plan."""

        # This would implement plan optimization strategies
        # For now, return original plan
        return plan

    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics about the HTN domain."""

        return {
            'total_tasks': len(self.domain.tasks),
            'primitive_tasks': len([t for t in self.domain.tasks.values() if t.is_primitive()]),
            'compound_tasks': len([t for t in self.domain.tasks.values() if t.is_compound()]),
            'total_methods': sum(len(methods) for methods in self.domain.methods.values()),
            'avg_methods_per_task': sum(len(methods) for methods in self.domain.methods.values()) / max(1, len(self.domain.tasks))
        }

    async def learn_from_execution(
        self,
        plan: HTNPlan,
        execution_result: Dict[str, Any]
    ) -> None:
        """Learn from plan execution to improve future planning."""

        # This would implement learning mechanisms
        # For now, placeholder for future implementation
        pass

    async def health_check(self) -> bool:
        """Check HTN engine health."""

        try:
            # Test basic domain operations
            stats = self.get_domain_statistics()
            return stats['total_tasks'] >= 0
        except Exception:
            return False


# Global instances (would be configured per domain)
htn_engine = None

def create_htn_engine(domain_name: str = "default") -> HTNEngine:
    """Create HTN engine with domain."""

    # Create default domain
    domain = HTNDomain(domain_name)

    # Add basic tasks (would be loaded from configuration)
    # This is a placeholder - actual domain would be much more comprehensive

    global htn_engine
    htn_engine = HTNEngine(domain)

    return htn_engine
