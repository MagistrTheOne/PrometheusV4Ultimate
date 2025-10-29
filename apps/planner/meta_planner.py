"""Meta-Planner for strategic planning and strategy selection."""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from libs.common.llm_providers import provider_registry, ChatMessage, ChatRole
from libs.common.schemas import Task
import httpx


class PlanningStrategy(Enum):
    """Available planning strategies."""
    DIRECT_EXECUTION = "direct_execution"
    DECOMPOSITION_BASED = "decomposition_based"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    PARALLEL_EXECUTION = "parallel_execution"
    HIERARCHICAL_PLANNING = "hierarchical_planning"
    ADAPTIVE_EXECUTION = "adaptive_execution"


class StrategyEvaluationMetric(Enum):
    """Metrics for evaluating planning strategies."""
    ESTIMATED_DURATION = "estimated_duration"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    SUCCESS_PROBABILITY = "success_probability"
    COST_ESTIMATE = "cost_estimate"
    COMPLEXITY_SCORE = "complexity_score"
    RISK_ASSESSMENT = "risk_assessment"


@dataclass
class StrategyProfile:
    """Profile of a planning strategy."""
    strategy: PlanningStrategy
    name: str
    description: str
    suitable_task_types: List[str]
    expected_performance: Dict[StrategyEvaluationMetric, float]
    resource_requirements: Dict[str, float]
    risk_factors: List[str]


@dataclass
class StrategyEvaluation:
    """Evaluation of a strategy for a specific task."""
    strategy: PlanningStrategy
    task_id: str
    scores: Dict[StrategyEvaluationMetric, float]
    confidence: float
    reasoning: str
    estimated_outcomes: Dict[str, Any]


@dataclass
class StrategySelection:
    """Result of strategy selection."""
    selected_strategy: PlanningStrategy
    task_id: str
    evaluation_results: List[StrategyEvaluation]
    selection_reasoning: str
    expected_performance: Dict[str, float]
    fallback_strategies: List[PlanningStrategy]
    selection_timestamp: datetime


@dataclass
class MetaPlanningContext:
    """Context for meta-planning decisions."""
    task: Task
    available_resources: Dict[str, float]
    time_constraints: Dict[str, Any]
    historical_performance: Dict[PlanningStrategy, List[float]]
    system_state: Dict[str, Any]
    environmental_factors: Dict[str, Any]


class MetaPlanner:
    """Meta-planner that selects optimal planning strategies."""

    def __init__(self):
        self.strategy_profiles = self._initialize_strategy_profiles()
        self.performance_history: Dict[str, List[StrategySelection]] = {}
        self.learning_engine = MetaLearningEngine()

    def _initialize_strategy_profiles(self) -> Dict[PlanningStrategy, StrategyProfile]:
        """Initialize profiles for all planning strategies."""

        return {
            PlanningStrategy.DIRECT_EXECUTION: StrategyProfile(
                strategy=PlanningStrategy.DIRECT_EXECUTION,
                name="Direct Execution",
                description="Execute task directly without complex planning",
                suitable_task_types=["simple", "routine", "well_defined"],
                expected_performance={
                    StrategyEvaluationMetric.ESTIMATED_DURATION: 0.8,
                    StrategyEvaluationMetric.RESOURCE_EFFICIENCY: 0.9,
                    StrategyEvaluationMetric.SUCCESS_PROBABILITY: 0.85,
                    StrategyEvaluationMetric.COST_ESTIMATE: 0.7,
                    StrategyEvaluationMetric.COMPLEXITY_SCORE: 0.3,
                    StrategyEvaluationMetric.RISK_ASSESSMENT: 0.4
                },
                resource_requirements={
                    "cpu_cores": 1,
                    "memory_gb": 2,
                    "planning_time_seconds": 10
                },
                risk_factors=["task_misunderstanding", "resource_shortage"]
            ),

            PlanningStrategy.DECOMPOSITION_BASED: StrategyProfile(
                strategy=PlanningStrategy.DECOMPOSITION_BASED,
                name="Decomposition Based",
                description="Break down complex tasks into manageable subtasks",
                suitable_task_types=["complex", "multi_step", "structured"],
                expected_performance={
                    StrategyEvaluationMetric.ESTIMATED_DURATION: 1.2,
                    StrategyEvaluationMetric.RESOURCE_EFFICIENCY: 0.8,
                    StrategyEvaluationMetric.SUCCESS_PROBABILITY: 0.9,
                    StrategyEvaluationMetric.COST_ESTIMATE: 0.9,
                    StrategyEvaluationMetric.COMPLEXITY_SCORE: 0.6,
                    StrategyEvaluationMetric.RISK_ASSESSMENT: 0.3
                },
                resource_requirements={
                    "cpu_cores": 2,
                    "memory_gb": 4,
                    "planning_time_seconds": 30
                },
                risk_factors=["decomposition_errors", "dependency_conflicts"]
            ),

            PlanningStrategy.ITERATIVE_REFINEMENT: StrategyProfile(
                strategy=PlanningStrategy.ITERATIVE_REFINEMENT,
                name="Iterative Refinement",
                description="Plan, execute, and refine iteratively",
                suitable_task_types=["exploratory", "creative", "uncertain"],
                expected_performance={
                    StrategyEvaluationMetric.ESTIMATED_DURATION: 1.5,
                    StrategyEvaluationMetric.RESOURCE_EFFICIENCY: 0.7,
                    StrategyEvaluationMetric.SUCCESS_PROBABILITY: 0.75,
                    StrategyEvaluationMetric.COST_ESTIMATE: 1.1,
                    StrategyEvaluationMetric.COMPLEXITY_SCORE: 0.8,
                    StrategyEvaluationMetric.RISK_ASSESSMENT: 0.6
                },
                resource_requirements={
                    "cpu_cores": 3,
                    "memory_gb": 6,
                    "planning_time_seconds": 60
                },
                risk_factors=["convergence_failures", "resource_waste"]
            ),

            PlanningStrategy.PARALLEL_EXECUTION: StrategyProfile(
                strategy=PlanningStrategy.PARALLEL_EXECUTION,
                name="Parallel Execution",
                description="Execute independent subtasks in parallel",
                suitable_task_types=["parallelizable", "independent_subtasks"],
                expected_performance={
                    StrategyEvaluationMetric.ESTIMATED_DURATION: 0.6,
                    StrategyEvaluationMetric.RESOURCE_EFFICIENCY: 0.6,
                    StrategyEvaluationMetric.SUCCESS_PROBABILITY: 0.8,
                    StrategyEvaluationMetric.COST_ESTIMATE: 1.2,
                    StrategyEvaluationMetric.COMPLEXITY_SCORE: 0.7,
                    StrategyEvaluationMetric.RISK_ASSESSMENT: 0.5
                },
                resource_requirements={
                    "cpu_cores": 4,
                    "memory_gb": 8,
                    "planning_time_seconds": 45
                },
                risk_factors=["synchronization_issues", "resource_contention"]
            ),

            PlanningStrategy.HIERARCHICAL_PLANNING: StrategyProfile(
                strategy=PlanningStrategy.HIERARCHICAL_PLANNING,
                name="Hierarchical Planning",
                description="Use HTN for complex hierarchical task planning",
                suitable_task_types=["very_complex", "hierarchical", "domain_specific"],
                expected_performance={
                    StrategyEvaluationMetric.ESTIMATED_DURATION: 1.8,
                    StrategyEvaluationMetric.RESOURCE_EFFICIENCY: 0.75,
                    StrategyEvaluationMetric.SUCCESS_PROBABILITY: 0.95,
                    StrategyEvaluationMetric.COST_ESTIMATE: 1.3,
                    StrategyEvaluationMetric.COMPLEXITY_SCORE: 0.9,
                    StrategyEvaluationMetric.RISK_ASSESSMENT: 0.2
                },
                resource_requirements={
                    "cpu_cores": 4,
                    "memory_gb": 12,
                    "planning_time_seconds": 120
                },
                risk_factors=["domain_model_incompleteness", "planning_horizon_limits"]
            ),

            PlanningStrategy.ADAPTIVE_EXECUTION: StrategyProfile(
                strategy=PlanningStrategy.ADAPTIVE_EXECUTION,
                name="Adaptive Execution",
                description="Continuously adapt plan based on execution feedback",
                suitable_task_types=["dynamic", "unpredictable", "real_time"],
                expected_performance={
                    StrategyEvaluationMetric.ESTIMATED_DURATION: 1.3,
                    StrategyEvaluationMetric.RESOURCE_EFFICIENCY: 0.8,
                    StrategyEvaluationMetric.SUCCESS_PROBABILITY: 0.8,
                    StrategyEvaluationMetric.COST_ESTIMATE: 1.0,
                    StrategyEvaluationMetric.COMPLEXITY_SCORE: 0.8,
                    StrategyEvaluationMetric.RISK_ASSESSMENT: 0.4
                },
                resource_requirements={
                    "cpu_cores": 3,
                    "memory_gb": 8,
                    "planning_time_seconds": 90
                },
                risk_factors=["adaptation_overhead", "feedback_delays"]
            )
        }

    async def select_optimal_strategy(
        self,
        task: Task,
        context: MetaPlanningContext
    ) -> StrategySelection:
        """Select the optimal planning strategy for a task."""

        # Evaluate all available strategies
        strategy_evaluations = await asyncio.gather(*[
            self._evaluate_strategy(strategy_profile, task, context)
            for strategy_profile in self.strategy_profiles.values()
        ])

        # Select the best strategy
        selected_strategy, evaluation_results = await self._select_best_strategy(
            strategy_evaluations, task, context
        )

        # Generate selection reasoning
        selection_reasoning = await self._generate_selection_reasoning(
            selected_strategy, evaluation_results, task, context
        )

        # Determine fallback strategies
        fallback_strategies = await self._determine_fallback_strategies(
            selected_strategy, evaluation_results
        )

        # Calculate expected performance
        expected_performance = await self._calculate_expected_performance(
            selected_strategy, evaluation_results
        )

        selection = StrategySelection(
            selected_strategy=selected_strategy,
            task_id=task.id,
            evaluation_results=evaluation_results,
            selection_reasoning=selection_reasoning,
            expected_performance=expected_performance,
            fallback_strategies=fallback_strategies,
            selection_timestamp=datetime.now()
        )

        # Store selection for learning
        if task.id not in self.performance_history:
            self.performance_history[task.id] = []
        self.performance_history[task.id].append(selection)

        return selection

    async def _evaluate_strategy(
        self,
        strategy_profile: StrategyProfile,
        task: Task,
        context: MetaPlanningContext
    ) -> StrategyEvaluation:
        """Evaluate a planning strategy for a specific task."""

        # Assess task-strategy suitability
        suitability_score = await self._assess_task_strategy_suitability(
            strategy_profile, task
        )

        # Evaluate resource compatibility
        resource_compatibility = await self._evaluate_resource_compatibility(
            strategy_profile, context.available_resources
        )

        # Consider time constraints
        time_compatibility = await self._evaluate_time_compatibility(
            strategy_profile, context.time_constraints
        )

        # Factor in historical performance
        historical_performance = await self._consider_historical_performance(
            strategy_profile.strategy, task.id
        )

        # Assess risk factors
        risk_assessment = await self._assess_strategy_risks(
            strategy_profile, task, context
        )

        # Combine all factors
        combined_scores = await self._combine_evaluation_factors(
            suitability_score,
            resource_compatibility,
            time_compatibility,
            historical_performance,
            risk_assessment
        )

        # Generate evaluation reasoning
        reasoning = await self._generate_evaluation_reasoning(
            strategy_profile, combined_scores, task, context
        )

        # Estimate outcomes
        estimated_outcomes = await self._estimate_strategy_outcomes(
            strategy_profile, task, context
        )

        return StrategyEvaluation(
            strategy=strategy_profile.strategy,
            task_id=task.id,
            scores=combined_scores,
            confidence=await self._calculate_evaluation_confidence(combined_scores),
            reasoning=reasoning,
            estimated_outcomes=estimated_outcomes
        )

    async def _assess_task_strategy_suitability(
        self,
        strategy_profile: StrategyProfile,
        task: Task
    ) -> float:
        """Assess how suitable a strategy is for the task type."""

        # Analyze task characteristics
        task_complexity = len(task.goal) / 1000.0  # Simple complexity measure
        task_structure = await self._analyze_task_structure(task)

        # Check if task type matches strategy suitability
        type_match_score = 0.0
        for task_type in strategy_profile.suitable_task_types:
            if task_type in task_structure:
                type_match_score = max(type_match_score, 1.0)
            elif task_type == "complex" and task_complexity > 0.5:
                type_match_score = max(type_match_score, 0.8)
            elif task_type == "simple" and task_complexity < 0.3:
                type_match_score = max(type_match_score, 0.9)

        return type_match_score

    async def _analyze_task_structure(self, task: Task) -> List[str]:
        """Analyze the structure of a task."""

        structure_indicators = []

        goal_lower = task.goal.lower()

        # Check for complexity indicators
        if any(word in goal_lower for word in ["analyze", "investigate", "study"]):
            structure_indicators.append("research")
        if any(word in goal_lower for word in ["implement", "build", "create"]):
            structure_indicators.append("implementation")
        if any(word in goal_lower for word in ["multiple", "several", "various"]):
            structure_indicators.append("multi_step")
        if len(task.goal.split()) > 50:
            structure_indicators.append("complex")
        if len(task.goal.split()) < 20:
            structure_indicators.append("simple")

        return structure_indicators

    async def _evaluate_resource_compatibility(
        self,
        strategy_profile: StrategyProfile,
        available_resources: Dict[str, float]
    ) -> float:
        """Evaluate if available resources can support the strategy."""

        compatibility_score = 1.0

        for resource, required in strategy_profile.resource_requirements.items():
            if resource in available_resources:
                available = available_resources[resource]
                if available < required:
                    # Penalize insufficient resources
                    compatibility_score *= (available / required)
            else:
                # Resource not available
                compatibility_score *= 0.5

        return max(0.0, min(1.0, compatibility_score))

    async def _evaluate_time_compatibility(
        self,
        strategy_profile: StrategyProfile,
        time_constraints: Dict[str, Any]
    ) -> float:
        """Evaluate if strategy fits within time constraints."""

        planning_time = strategy_profile.resource_requirements.get("planning_time_seconds", 0)

        if "max_planning_time" in time_constraints:
            max_time = time_constraints["max_planning_time"]
            if planning_time > max_time:
                return max_time / planning_time  # Penalty for exceeding time
            else:
                return 1.0 - (planning_time / max_time) * 0.2  # Small preference for faster planning

        return 1.0

    async def _consider_historical_performance(
        self,
        strategy: PlanningStrategy,
        task_id: str
    ) -> float:
        """Consider historical performance of the strategy."""

        # Look for similar tasks
        similar_tasks = await self._find_similar_tasks(task_id)

        if not similar_tasks:
            return 0.5  # Neutral score if no history

        # Calculate average performance for this strategy on similar tasks
        strategy_performances = []
        for similar_task_id in similar_tasks:
            if similar_task_id in self.performance_history:
                for selection in self.performance_history[similar_task_id]:
                    if selection.selected_strategy == strategy:
                        # Use success probability as performance measure
                        perf = selection.expected_performance.get("success_probability", 0.5)
                        strategy_performances.append(perf)

        if strategy_performances:
            return sum(strategy_performances) / len(strategy_performances)

        return 0.5

    async def _find_similar_tasks(self, task_id: str) -> List[str]:
        """Find tasks similar to the given task."""

        # This would use task similarity analysis
        # For now, return empty list
        return []

    async def _assess_strategy_risks(
        self,
        strategy_profile: StrategyProfile,
        task: Task,
        context: MetaPlanningContext
    ) -> float:
        """Assess risks associated with using this strategy."""

        risk_score = 0.0

        # Base risk from strategy profile
        base_risk = len(strategy_profile.risk_factors) / 10.0  # Normalize to 0-1

        # Task-specific risk adjustments
        task_risk_factors = await self._analyze_task_risks(task)

        # Context-specific risk adjustments
        context_risk_factors = await self._analyze_context_risks(context)

        risk_score = base_risk + task_risk_factors + context_risk_factors

        return min(1.0, max(0.0, risk_score))

    async def _analyze_task_risks(self, task: Task) -> float:
        """Analyze risks inherent to the task."""

        risk_indicators = 0

        goal_lower = task.goal.lower()

        # High-risk indicators
        if any(word in goal_lower for word in ["critical", "urgent", "emergency"]):
            risk_indicators += 0.3
        if any(word in goal_lower for word in ["complex", "difficult", "challenging"]):
            risk_indicators += 0.2
        if len(task.goal.split()) > 100:  # Long, complex goals
            risk_indicators += 0.1

        return risk_indicators

    async def _analyze_context_risks(self, context: MetaPlanningContext) -> float:
        """Analyze risks from the planning context."""

        risk_indicators = 0

        # Resource scarcity
        if context.available_resources.get("cpu_cores", 0) < 2:
            risk_indicators += 0.2
        if context.available_resources.get("memory_gb", 0) < 4:
            risk_indicators += 0.2

        # Time pressure
        if context.time_constraints.get("urgent", False):
            risk_indicators += 0.3

        return risk_indicators

    async def _combine_evaluation_factors(
        self,
        suitability: float,
        resource_compatibility: float,
        time_compatibility: float,
        historical_performance: float,
        risk_assessment: float
    ) -> Dict[StrategyEvaluationMetric, float]:
        """Combine all evaluation factors into final scores."""

        # Weights for different factors
        weights = {
            'suitability': 0.3,
            'resources': 0.25,
            'time': 0.15,
            'history': 0.15,
            'risk': 0.15
        }

        # Calculate weighted scores for each metric
        scores = {}

        for metric in StrategyEvaluationMetric:
            base_score = 0.5  # Neutral starting point

            if metric == StrategyEvaluationMetric.SUCCESS_PROBABILITY:
                base_score = (
                    suitability * weights['suitability'] +
                    resource_compatibility * weights['resources'] +
                    time_compatibility * weights['time'] +
                    historical_performance * weights['history']
                ) * (1 - risk_assessment * weights['risk'])

            elif metric == StrategyEvaluationMetric.ESTIMATED_DURATION:
                # Strategies with higher complexity take longer
                base_score = suitability * 1.2 if suitability > 0.7 else suitability

            elif metric == StrategyEvaluationMetric.RESOURCE_EFFICIENCY:
                base_score = resource_compatibility

            elif metric == StrategyEvaluationMetric.COST_ESTIMATE:
                base_score = resource_compatibility * 0.8 + time_compatibility * 0.2

            elif metric == StrategyEvaluationMetric.COMPLEXITY_SCORE:
                base_score = suitability

            elif metric == StrategyEvaluationMetric.RISK_ASSESSMENT:
                base_score = risk_assessment

            scores[metric] = max(0.0, min(1.0, base_score))

        return scores

    async def _select_best_strategy(
        self,
        evaluations: List[StrategyEvaluation],
        task: Task,
        context: MetaPlanningContext
    ) -> Tuple[PlanningStrategy, List[StrategyEvaluation]]:
        """Select the best strategy from evaluations."""

        if not evaluations:
            return PlanningStrategy.DIRECT_EXECUTION, []

        # Score each strategy based on evaluation results
        strategy_scores = []
        for evaluation in evaluations:
            # Calculate overall score (weighted combination of metrics)
            overall_score = (
                evaluation.scores[StrategyEvaluationMetric.SUCCESS_PROBABILITY] * 0.4 +
                (1 - evaluation.scores[StrategyEvaluationMetric.ESTIMATED_DURATION]) * 0.2 +
                evaluation.scores[StrategyEvaluationMetric.RESOURCE_EFFICIENCY] * 0.2 +
                (1 - evaluation.scores[StrategyEvaluationMetric.RISK_ASSESSMENT]) * 0.2
            )

            strategy_scores.append((evaluation.strategy, overall_score, evaluation))

        # Select strategy with highest score
        best_strategy, best_score, best_evaluation = max(strategy_scores, key=lambda x: x[1])

        return best_strategy, evaluations

    async def _generate_selection_reasoning(
        self,
        selected_strategy: PlanningStrategy,
        evaluations: List[StrategyEvaluation],
        task: Task,
        context: MetaPlanningContext
    ) -> str:
        """Generate reasoning for strategy selection."""

        strategy_profile = self.strategy_profiles[selected_strategy]

        reasoning_parts = [
            f"Selected {strategy_profile.name} strategy for task '{task.goal[:50]}...'",
            f"Strategy description: {strategy_profile.description}",
            f"Task suitability: {await self._assess_task_strategy_suitability(strategy_profile, task):.2f}",
            f"Resource compatibility: {await self._evaluate_resource_compatibility(strategy_profile, context.available_resources):.2f}"
        ]

        # Add comparison with alternatives
        if len(evaluations) > 1:
            alt_strategies = [e for e in evaluations if e.strategy != selected_strategy]
            if alt_strategies:
                best_alt = max(alt_strategies, key=lambda e: e.scores[StrategyEvaluationMetric.SUCCESS_PROBABILITY])
                reasoning_parts.append(
                    f"Better than alternative {best_alt.strategy.value} "
                    f"(success prob: {best_alt.scores[StrategyEvaluationMetric.SUCCESS_PROBABILITY]:.2f} vs "
                    f"{evaluations[0].scores[StrategyEvaluationMetric.SUCCESS_PROBABILITY]:.2f})"
                )

        return ". ".join(reasoning_parts)

    async def _determine_fallback_strategies(
        self,
        selected_strategy: PlanningStrategy,
        evaluations: List[StrategyEvaluation]
    ) -> List[PlanningStrategy]:
        """Determine fallback strategies in case the primary fails."""

        # Sort evaluations by success probability
        sorted_evaluations = sorted(
            evaluations,
            key=lambda e: e.scores[StrategyEvaluationMetric.SUCCESS_PROBABILITY],
            reverse=True
        )

        # Return top 2 alternatives (excluding the selected one)
        fallback_candidates = [
            e.strategy for e in sorted_evaluations
            if e.strategy != selected_strategy
        ][:2]

        # Always include DIRECT_EXECUTION as a safe fallback
        if PlanningStrategy.DIRECT_EXECUTION not in fallback_candidates:
            fallback_candidates.append(PlanningStrategy.DIRECT_EXECUTION)

        return fallback_candidates[:2]

    async def _calculate_expected_performance(
        self,
        selected_strategy: PlanningStrategy,
        evaluations: List[StrategyEvaluation]
    ) -> Dict[str, float]:
        """Calculate expected performance metrics."""

        selected_evaluation = next(
            (e for e in evaluations if e.strategy == selected_strategy),
            None
        )

        if not selected_evaluation:
            return {}

        return {
            metric.value: score
            for metric, score in selected_evaluation.scores.items()
        }

    async def _calculate_evaluation_confidence(self, scores: Dict[StrategyEvaluationMetric, float]) -> float:
        """Calculate confidence in the evaluation."""

        # Higher confidence when scores are more extreme (closer to 0 or 1)
        score_extremes = [abs(score - 0.5) * 2 for score in scores.values()]
        avg_extreme = sum(score_extremes) / len(score_extremes)

        return min(1.0, max(0.5, avg_extreme))

    async def _generate_evaluation_reasoning(
        self,
        strategy_profile: StrategyProfile,
        scores: Dict[StrategyEvaluationMetric, float],
        task: Task,
        context: MetaPlanningContext
    ) -> str:
        """Generate reasoning for strategy evaluation."""

        key_scores = [
            f"success_probability: {scores[StrategyEvaluationMetric.SUCCESS_PROBABILITY]:.2f}",
            f"estimated_duration: {scores[StrategyEvaluationMetric.ESTIMATED_DURATION]:.2f}",
            f"resource_efficiency: {scores[StrategyEvaluationMetric.RESOURCE_EFFICIENCY]:.2f}",
            f"risk_assessment: {scores[StrategyEvaluationMetric.RISK_ASSESSMENT]:.2f}"
        ]

        return f"Strategy {strategy_profile.name}: {', '.join(key_scores)}"

    async def _estimate_strategy_outcomes(
        self,
        strategy_profile: StrategyProfile,
        task: Task,
        context: MetaPlanningContext
    ) -> Dict[str, Any]:
        """Estimate outcomes of using this strategy."""

        # Use strategy profile expectations as base
        outcomes = dict(strategy_profile.expected_performance)

        # Adjust based on task and context
        task_complexity_factor = len(task.goal) / 500.0
        resource_abundance_factor = sum(context.available_resources.values()) / 100.0

        # Complexity affects duration and success
        outcomes['estimated_duration'] *= (1 + task_complexity_factor * 0.5)
        outcomes['success_probability'] *= (1 - task_complexity_factor * 0.2)

        # Resources affect efficiency and cost
        outcomes['resource_efficiency'] *= min(1.0, resource_abundance_factor)
        outcomes['cost_estimate'] *= (1 + (1 - resource_abundance_factor) * 0.3)

        return outcomes

    async def learn_from_outcome(
        self,
        task_id: str,
        strategy: PlanningStrategy,
        actual_performance: Dict[str, float]
    ) -> None:
        """Learn from actual strategy performance."""

        await self.learning_engine.update_strategy_performance(
            strategy, actual_performance
        )

        # Update historical performance
        if task_id in self.performance_history:
            for selection in self.performance_history[task_id]:
                if selection.selected_strategy == strategy:
                    # Could update expected performance based on actual results
                    pass

    async def get_strategy_recommendations(
        self,
        task_characteristics: Dict[str, Any]
    ) -> List[PlanningStrategy]:
        """Get strategy recommendations for given task characteristics."""

        # This would use learned preferences
        # For now, return all strategies
        return list(PlanningStrategy)

    async def health_check(self) -> bool:
        """Check meta-planner health."""

        try:
            # Test strategy evaluation
            test_task = Task(
                id="health_check",
                goal="test task",
                inputs={},
                limits={},
                project_id="test"
            )

            test_context = MetaPlanningContext(
                task=test_task,
                available_resources={"cpu_cores": 4, "memory_gb": 8},
                time_constraints={},
                historical_performance={},
                system_state={},
                environmental_factors={}
            )

            result = await self.select_optimal_strategy(test_task, test_context)
            return result.selected_strategy is not None

        except Exception:
            return False


# Global instance
meta_planner = MetaPlanner()
