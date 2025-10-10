"""Cost/time-aware planner for PrometheusULTIMATE v4."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from libs.common.llm_providers import provider_registry, ChatMessage, ChatRole
from libs.common.schemas import Step


class PlanningPolicy:
    """Planning policy configuration."""
    
    def __init__(
        self,
        cost_weight: float = 0.5,
        time_weight: float = 0.3,
        quality_weight: float = 0.5,
        max_parallel_steps: int = 3
    ):
        self.cost_weight = cost_weight
        self.time_weight = time_weight
        self.quality_weight = quality_weight
        self.max_parallel_steps = max_parallel_steps


class SkillMetrics:
    """Historical metrics for skill performance."""
    
    def __init__(
        self,
        skill_name: str,
        avg_latency_ms: int = 1000,
        avg_cost_usd: float = 0.001,
        success_rate: float = 0.95,
        error_rate: float = 0.05
    ):
        self.skill_name = skill_name
        self.avg_latency_ms = avg_latency_ms
        self.avg_cost_usd = avg_cost_usd
        self.success_rate = success_rate
        self.error_rate = error_rate


class Planner:
    """Cost/time-aware task planner."""
    
    def __init__(self):
        self.policy = PlanningPolicy()
        self.skill_metrics = self._initialize_skill_metrics()
        self.available_skills = self._get_available_skills()
    
    def _initialize_skill_metrics(self) -> Dict[str, SkillMetrics]:
        """Initialize skill performance metrics."""
        return {
            "csv_join": SkillMetrics("csv_join", 500, 0.0005, 0.98),
            "csv_clean": SkillMetrics("csv_clean", 300, 0.0003, 0.99),
            "http_fetch": SkillMetrics("http_fetch", 2000, 0.001, 0.90),
            "sql_query": SkillMetrics("sql_query", 800, 0.0008, 0.95),
            "plot_basic": SkillMetrics("plot_basic", 1200, 0.0012, 0.92),
            "ocr_stub": SkillMetrics("ocr_stub", 3000, 0.003, 0.85),
            "code_format": SkillMetrics("code_format", 400, 0.0004, 0.99),
            "math_calc": SkillMetrics("math_calc", 200, 0.0002, 0.99),
            "file_zip": SkillMetrics("file_zip", 600, 0.0006, 0.97),
            "email_draft": SkillMetrics("email_draft", 1000, 0.001, 0.94)
        }
    
    def _get_available_skills(self) -> List[str]:
        """Get list of available skills."""
        return list(self.skill_metrics.keys())
    
    async def create_plan(
        self,
        goal: str,
        inputs: Dict[str, Any],
        limits: Dict[str, Any],
        project_id: str,
        policy: str = "auto"
    ) -> Dict[str, Any]:
        """Create execution plan for a task."""
        
        # Determine model policy
        model_policy = self._determine_model_policy(policy, limits)
        
        # Generate plan using LLM
        plan_steps = await self._generate_plan_steps(goal, inputs, model_policy)
        
        # Optimize plan for cost/time/quality
        optimized_steps = self._optimize_plan(plan_steps, limits)
        
        # Create parallel execution groups
        execution_groups = self._create_execution_groups(optimized_steps)
        
        # Calculate budget
        budget = self._calculate_budget(optimized_steps)
        
        plan = {
            "id": str(uuid4()),
            "goal": goal,
            "policy": model_policy,
            "steps": optimized_steps,
            "execution_groups": execution_groups,
            "budget": budget,
            "estimated_duration_ms": self._estimate_duration(optimized_steps),
            "parallel_steps": len(execution_groups)
        }
        
        return plan
    
    def _determine_model_policy(
        self,
        policy: str,
        limits: Dict[str, Any]
    ) -> str:
        """Determine model policy based on constraints."""
        
        if policy != "auto":
            return policy
        
        # Auto policy based on limits
        time_limit = limits.get("time_s", 300)
        cost_limit = limits.get("cost_usd", 1.0)
        
        if time_limit <= 10 and cost_limit <= 0.1:
            return "tiny"  # RadonSAI-Small
        elif time_limit <= 60 and cost_limit <= 0.5:
            return "base"  # RadonSAI
        elif time_limit <= 300 and cost_limit <= 2.0:
            return "balanced"  # RadonSAI-Balanced
        elif time_limit <= 600 and cost_limit <= 5.0:
            return "ultra"  # RadonSAI-Ultra
        else:
            return "mega"  # RadonSAI-Mega
    
    async def _generate_plan_steps(
        self,
        goal: str,
        inputs: Dict[str, Any],
        model_policy: str
    ) -> List[Dict[str, Any]]:
        """Generate plan steps using LLM."""
        
        # Prepare prompt for planning
        messages = [
            ChatMessage(
                role=ChatRole.SYSTEM,
                content="You are a task planner for PrometheusULTIMATE v4. Break down goals into executable steps using available skills."
            ),
            ChatMessage(
                role=ChatRole.USER,
                content=f"""
Goal: {goal}
Inputs: {inputs}
Available skills: {', '.join(self.available_skills)}
Model policy: {model_policy}

Create a step-by-step plan. Each step should specify:
- name: descriptive step name
- skill_name: one of the available skills
- inputs: required inputs for the skill
- description: what this step accomplishes

Return only the plan steps in a structured format.
"""
            )
        ]
        
        # Get LLM response
        response = await provider_registry.route_request(
            messages=messages,
            model=f"radon/{model_policy}",
            temperature=0.3
        )
        
        # Parse response into steps
        steps = self._parse_plan_response(response.text, goal, inputs)
        
        return steps
    
    def _parse_plan_response(
        self,
        response: str,
        goal: str,
        inputs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse LLM response into structured steps."""
        
        # For now, create deterministic steps based on goal analysis
        steps = []
        
        if "csv" in goal.lower() and "report" in goal.lower():
            steps = [
                {
                    "name": "load_csv_data",
                    "skill_name": "csv_clean",
                    "inputs": {"file_path": inputs.get("csv_file", "data.csv")},
                    "description": "Load and clean CSV data"
                },
                {
                    "name": "analyze_data",
                    "skill_name": "sql_query",
                    "inputs": {"query": "SELECT * FROM data", "data": "{{load_csv_data.output}}"},
                    "description": "Analyze the data"
                },
                {
                    "name": "create_visualization",
                    "skill_name": "plot_basic",
                    "inputs": {"data": "{{analyze_data.output}}", "chart_type": "line"},
                    "description": "Create data visualization"
                }
            ]
        elif "code" in goal.lower() and "fix" in goal.lower():
            steps = [
                {
                    "name": "analyze_code",
                    "skill_name": "code_format",
                    "inputs": {"code": inputs.get("code", "")},
                    "description": "Analyze and format code"
                },
                {
                    "name": "run_tests",
                    "skill_name": "math_calc",
                    "inputs": {"expression": "test_code()"},
                    "description": "Run code tests"
                }
            ]
        else:
            # Generic plan
            steps = [
                {
                    "name": "analyze_goal",
                    "skill_name": "math_calc",
                    "inputs": {"expression": f"analyze('{goal}')"},
                    "description": f"Analyze goal: {goal}"
                },
                {
                    "name": "execute_task",
                    "skill_name": "csv_clean",
                    "inputs": {"file_path": "input.txt"},
                    "description": f"Execute task: {goal}"
                }
            ]
        
        return steps
    
    def _optimize_plan(
        self,
        steps: List[Dict[str, Any]],
        limits: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Optimize plan for cost, time, and quality."""
        
        optimized_steps = []
        
        for step in steps:
            skill_name = step["skill_name"]
            
            if skill_name not in self.skill_metrics:
                continue
            
            metrics = self.skill_metrics[skill_name]
            
            # Calculate score
            score = (
                self.policy.cost_weight * metrics.avg_cost_usd +
                self.policy.time_weight * (metrics.avg_latency_ms / 1000) +
                self.policy.quality_weight * (1 - metrics.error_rate)
            )
            
            # Add score to step
            step["score"] = score
            step["estimated_latency_ms"] = metrics.avg_latency_ms
            step["estimated_cost_usd"] = metrics.avg_cost_usd
            step["success_rate"] = metrics.success_rate
            
            optimized_steps.append(step)
        
        # Sort by score (lower is better)
        optimized_steps.sort(key=lambda x: x["score"])
        
        return optimized_steps
    
    def _create_execution_groups(
        self,
        steps: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Create parallel execution groups."""
        
        groups = []
        current_group = []
        
        for step in steps:
            # Check if step can run in parallel with current group
            if self._can_run_parallel(current_group, step):
                current_group.append(step)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [step]
            
            # Limit group size
            if len(current_group) >= self.policy.max_parallel_steps:
                groups.append(current_group)
                current_group = []
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _can_run_parallel(
        self,
        group: List[Dict[str, Any]],
        step: Dict[str, Any]
    ) -> bool:
        """Check if step can run in parallel with group."""
        
        # Simple heuristic: steps with different skills can run in parallel
        group_skills = {s["skill_name"] for s in group}
        return step["skill_name"] not in group_skills
    
    def _calculate_budget(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate execution budget."""
        
        total_cost = sum(step.get("estimated_cost_usd", 0) for step in steps)
        total_time = sum(step.get("estimated_latency_ms", 0) for step in steps)
        
        return {
            "total_cost_usd": total_cost,
            "total_time_ms": total_time,
            "steps_count": len(steps),
            "parallel_groups": len(self._create_execution_groups(steps))
        }
    
    def _estimate_duration(self, steps: List[Dict[str, Any]]) -> int:
        """Estimate total execution duration."""
        
        execution_groups = self._create_execution_groups(steps)
        
        # Duration is sum of group durations (parallel steps in group)
        total_duration = 0
        for group in execution_groups:
            group_duration = max(step.get("estimated_latency_ms", 0) for step in group)
            total_duration += group_duration
        
        return total_duration
    
    def update_skill_metrics(
        self,
        skill_name: str,
        latency_ms: int,
        cost_usd: float,
        success: bool
    ):
        """Update skill performance metrics."""
        
        if skill_name not in self.skill_metrics:
            return
        
        metrics = self.skill_metrics[skill_name]
        
        # Simple exponential moving average
        alpha = 0.1
        metrics.avg_latency_ms = int(alpha * latency_ms + (1 - alpha) * metrics.avg_latency_ms)
        metrics.avg_cost_usd = alpha * cost_usd + (1 - alpha) * metrics.avg_cost_usd
        
        if success:
            metrics.success_rate = min(1.0, metrics.success_rate + 0.01)
            metrics.error_rate = max(0.0, metrics.error_rate - 0.01)
        else:
            metrics.success_rate = max(0.0, metrics.success_rate - 0.01)
            metrics.error_rate = min(1.0, metrics.error_rate + 0.01)
    
    async def health_check(self) -> bool:
        """Check planner health."""
        return True


# Global planner instance
planner = Planner()
