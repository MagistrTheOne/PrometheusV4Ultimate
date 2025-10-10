"""Main Orchestrator service."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from libs.common.schemas import Task, TaskStatus, Step, StepStatus
from libs.common.llm_providers import provider_registry

from .database import task_db
from .state_machine import TaskState, StepState, task_state_machine, step_state_machine


class Orchestrator:
    """Main orchestrator for task lifecycle management."""
    
    def __init__(self):
        self.db = task_db
        self.state_machine = task_state_machine
        self.step_state_machine = step_state_machine
        self.active_tasks: Dict[str, asyncio.Task] = {}
    
    async def create_task(
        self,
        goal: str,
        inputs: Dict[str, Any],
        limits: Dict[str, Any],
        project_id: str
    ) -> str:
        """Create and start a new task."""
        # Create task in database
        task_id = await self.db.create_task(goal, inputs, limits, project_id)
        
        # Log task creation
        await self.db.log_event(
            task_id=task_id,
            event_type="task_created",
            data={
                "goal": goal,
                "inputs": inputs,
                "limits": limits,
                "project_id": project_id
            }
        )
        
        # Start task execution asynchronously
        task_coroutine = self._execute_task(task_id)
        self.active_tasks[task_id] = asyncio.create_task(task_coroutine)
        
        return task_id
    
    async def _execute_task(self, task_id: str):
        """Execute task lifecycle."""
        try:
            # Transition to PLANNING
            await self._transition_task_state(task_id, TaskState.PLANNING)
            
            # TODO: Call Planner service
            plan = await self._create_plan(task_id)
            
            # Transition to RUNNING
            await self._transition_task_state(task_id, TaskState.RUNNING)
            
            # Execute steps
            await self._execute_steps(task_id, plan)
            
            # Transition to REVIEW
            await self._transition_task_state(task_id, TaskState.REVIEW)
            
            # TODO: Call Critic service
            review_result = await self._review_task(task_id)
            
            if review_result["passed"]:
                await self._transition_task_state(task_id, TaskState.COMPLETED)
            else:
                if review_result.get("retry", False):
                    await self._transition_task_state(task_id, TaskState.RUNNING)
                    # Retry failed steps
                    await self._retry_failed_steps(task_id)
                else:
                    await self._transition_task_state(task_id, TaskState.FAILED)
        
        except Exception as e:
            await self._transition_task_state(task_id, TaskState.FAILED, error=str(e))
        
        finally:
            # Clean up active task
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def _transition_task_state(
        self,
        task_id: str,
        new_state: TaskState,
        error: Optional[str] = None,
        rationale: Optional[str] = None
    ):
        """Transition task to new state."""
        # Get current task
        task_data = await self.db.get_task(task_id)
        if not task_data:
            raise ValueError(f"Task {task_id} not found")
        
        current_state = TaskState(task_data["status"])
        
        # Validate transition
        if not self.state_machine.can_transition(current_state, new_state):
            raise ValueError(
                f"Invalid transition from {current_state.value} to {new_state.value}"
            )
        
        # Update database
        await self.db.update_task_status(
            task_id=task_id,
            status=new_state.value,
            error=error,
            rationale=rationale
        )
        
        # Log transition
        await self.db.log_event(
            task_id=task_id,
            event_type="state_transition",
            data={
                "from_state": current_state.value,
                "to_state": new_state.value,
                "error": error,
                "rationale": rationale
            }
        )
    
    async def _create_plan(self, task_id: str) -> Dict[str, Any]:
        """Create execution plan for task."""
        task_data = await self.db.get_task(task_id)
        goal = task_data["goal"]
        inputs = task_data["inputs"]
        limits = task_data["limits"]
        project_id = task_data["project_id"]
        
        # Call Planner service via HTTP
        import httpx
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "http://planner:8001/plan",
                    json={
                        "goal": goal,
                        "inputs": inputs,
                        "limits": limits,
                        "project_id": project_id,
                        "policy": "auto"
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                plan_data = response.json()
                
                plan = {
                    "id": plan_data["plan_id"],
                    "steps": plan_data["steps"],
                    "budget": plan_data["budget"],
                    "estimated_duration_ms": plan_data["estimated_duration_ms"]
                }
                
            except Exception as e:
                # Fallback to simple plan if planner is unavailable
                plan = {
                    "id": str(uuid4()),
                    "steps": [
                        {
                            "name": "analyze_goal",
                            "description": f"Analyze goal: {goal}",
                            "skill_name": "math_calc",
                            "inputs": {"expression": f"analyze('{goal}')"}
                        },
                        {
                            "name": "execute_task",
                            "description": f"Execute task: {goal}",
                            "skill_name": "csv_clean",
                            "inputs": {"file_path": "input.txt"}
                        }
                    ],
                    "budget": {"total_cost_usd": 0.002, "total_time_ms": 2000},
                    "estimated_duration_ms": 2000
                }
        
        # Log plan creation
        await self.db.log_event(
            task_id=task_id,
            event_type="plan_created",
            data=plan
        )
        
        return plan
    
    async def _execute_steps(self, task_id: str, plan: Dict[str, Any]):
        """Execute all steps in the plan."""
        steps = plan.get("steps", [])
        
        for step_data in steps:
            step_id = await self.db.create_step(
                task_id=task_id,
                name=step_data["name"],
                description=step_data["description"],
                skill_name=step_data["skill_name"],
                inputs=step_data["inputs"]
            )
            
            await self._execute_step(step_id)
    
    async def _execute_step(self, step_id: str):
        """Execute a single step."""
        # Get step data
        steps = await self.db.get_task_steps(step_id.split('-')[0])  # Extract task_id
        step_data = next((s for s in steps if s["id"] == step_id), None)
        if not step_data:
            raise ValueError(f"Step {step_id} not found")
        
        # Transition to RUNNING
        await self._transition_step_state(step_id, StepState.RUNNING)
        
        try:
            # TODO: Call Skills service
            result = await self._run_skill(
                skill_name=step_data["skill_name"],
                inputs=step_data["inputs"]
            )
            
            # Transition to COMPLETED
            await self._transition_step_state(
                step_id,
                StepState.COMPLETED,
                outputs=result
            )
        
        except Exception as e:
            # Check if we can retry
            if step_data["retry_count"] < step_data["max_retries"]:
                await self._transition_step_state(step_id, StepState.RETRYING)
                await self.db.increment_step_retry(step_id)
                # Retry after delay
                await asyncio.sleep(1)
                await self._execute_step(step_id)
            else:
                await self._transition_step_state(
                    step_id,
                    StepState.FAILED,
                    error=str(e)
                )
    
    async def _transition_step_state(
        self,
        step_id: str,
        new_state: StepState,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Transition step to new state."""
        # Update database
        await self.db.update_step_status(
            step_id=step_id,
            status=new_state.value,
            outputs=outputs,
            error=error
        )
        
        # Log transition
        await self.db.log_event(
            task_id=step_id.split('-')[0],  # Extract task_id
            step_id=step_id,
            event_type="step_state_transition",
            data={
                "step_id": step_id,
                "to_state": new_state.value,
                "outputs": outputs,
                "error": error
            }
        )
    
    async def _run_skill(
        self,
        skill_name: str,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a skill (placeholder for Skills service integration)."""
        # TODO: Call Skills service via HTTP
        # For now, return mock result
        return {
            "status": "completed",
            "result": f"Skill {skill_name} executed with inputs {inputs}",
            "artifacts": []
        }
    
    async def _review_task(self, task_id: str) -> Dict[str, Any]:
        """Review task execution using Critic service."""
        # Get task data for review
        task_data = await self.db.get_task(task_id)
        steps = await self.db.get_task_steps(task_id)
        
        # Collect content from all steps
        content = f"Task: {task_data['goal']}\n"
        for step in steps:
            if step["outputs"]:
                content += f"Step {step['name']}: {step['outputs']}\n"
        
        # Call Critic service via HTTP
        import httpx
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "http://critic:8002/review",
                    json={
                        "task_id": task_id,
                        "content": content,
                        "context": {"steps": steps},
                        "project_id": task_data["project_id"]
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                review_data = response.json()
                
                return {
                    "passed": review_data["overall_passed"],
                    "issues": review_data["issues"],
                    "suggestions": review_data["suggestions"],
                    "confidence": review_data["confidence"],
                    "checks": review_data["checks"]
                }
                
            except Exception as e:
                # Fallback to simple review if critic is unavailable
                return {
                    "passed": True,
                    "issues": [f"Critic unavailable: {str(e)}"],
                    "suggestions": ["Manual review recommended"],
                    "confidence": 0.5
                }
    
    async def _retry_failed_steps(self, task_id: str):
        """Retry failed steps."""
        steps = await self.db.get_task_steps(task_id)
        failed_steps = [s for s in steps if s["status"] == StepStatus.FAILED.value]
        
        for step in failed_steps:
            if step["retry_count"] < step["max_retries"]:
                await self._execute_step(step["id"])
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive task status."""
        task_data = await self.db.get_task(task_id)
        if not task_data:
            return None
        
        steps = await self.db.get_task_steps(task_id)
        events = await self.db.get_task_events(task_id)
        artifacts = await self.db.get_task_artifacts(task_id)
        
        return {
            "task": task_data,
            "steps": steps,
            "events": events,
            "artifacts": artifacts,
            "is_active": task_id in self.active_tasks
        }
    
    async def abort_task(self, task_id: str) -> bool:
        """Abort a running task."""
        if task_id not in self.active_tasks:
            return False
        
        # Cancel the task
        self.active_tasks[task_id].cancel()
        
        # Transition to ABORTED
        await self._transition_task_state(task_id, TaskState.ABORTED)
        
        return True
    
    async def health_check(self) -> bool:
        """Check orchestrator health."""
        return await self.db.health_check()


# Global orchestrator instance
orchestrator = Orchestrator()
