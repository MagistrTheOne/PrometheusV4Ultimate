"""Task state machine for Orchestrator."""

from enum import Enum
from typing import Dict, List, Optional, Set
from uuid import UUID

from libs.common.schemas import TaskStatus, StepStatus


class TaskState(Enum):
    """Task execution states."""
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    REVIEW = "review"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class StepState(Enum):
    """Step execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class StateTransition:
    """State transition with conditions and actions."""
    
    def __init__(
        self,
        from_state: TaskState,
        to_state: TaskState,
        condition: str,
        action: Optional[str] = None
    ):
        self.from_state = from_state
        self.to_state = to_state
        self.condition = condition
        self.action = action


class TaskStateMachine:
    """Task state machine with transitions and validation."""
    
    def __init__(self):
        self.transitions = self._build_transitions()
        self.final_states = {TaskState.COMPLETED, TaskState.FAILED, TaskState.ABORTED}
    
    def _build_transitions(self) -> Dict[TaskState, List[StateTransition]]:
        """Build state transition rules."""
        return {
            TaskState.PENDING: [
                StateTransition(
                    TaskState.PENDING,
                    TaskState.PLANNING,
                    "planning_started",
                    "start_planning"
                ),
                StateTransition(
                    TaskState.PENDING,
                    TaskState.ABORTED,
                    "task_aborted",
                    "abort_task"
                )
            ],
            TaskState.PLANNING: [
                StateTransition(
                    TaskState.PLANNING,
                    TaskState.RUNNING,
                    "plan_completed",
                    "start_execution"
                ),
                StateTransition(
                    TaskState.PLANNING,
                    TaskState.FAILED,
                    "planning_failed",
                    "fail_task"
                ),
                StateTransition(
                    TaskState.PLANNING,
                    TaskState.ABORTED,
                    "task_aborted",
                    "abort_task"
                )
            ],
            TaskState.RUNNING: [
                StateTransition(
                    TaskState.RUNNING,
                    TaskState.REVIEW,
                    "all_steps_completed",
                    "start_review"
                ),
                StateTransition(
                    TaskState.RUNNING,
                    TaskState.FAILED,
                    "critical_step_failed",
                    "fail_task"
                ),
                StateTransition(
                    TaskState.RUNNING,
                    TaskState.ABORTED,
                    "task_aborted",
                    "abort_task"
                )
            ],
            TaskState.REVIEW: [
                StateTransition(
                    TaskState.REVIEW,
                    TaskState.COMPLETED,
                    "review_passed",
                    "complete_task"
                ),
                StateTransition(
                    TaskState.REVIEW,
                    TaskState.RUNNING,
                    "review_failed_retry",
                    "retry_steps"
                ),
                StateTransition(
                    TaskState.REVIEW,
                    TaskState.FAILED,
                    "review_failed_final",
                    "fail_task"
                ),
                StateTransition(
                    TaskState.REVIEW,
                    TaskState.ABORTED,
                    "task_aborted",
                    "abort_task"
                )
            ]
        }
    
    def can_transition(self, from_state: TaskState, to_state: TaskState) -> bool:
        """Check if transition is allowed."""
        if from_state not in self.transitions:
            return False
        
        return any(
            transition.to_state == to_state
            for transition in self.transitions[from_state]
        )
    
    def get_valid_transitions(self, current_state: TaskState) -> List[TaskState]:
        """Get all valid next states."""
        if current_state not in self.transitions:
            return []
        
        return [transition.to_state for transition in self.transitions[current_state]]
    
    def is_final_state(self, state: TaskState) -> bool:
        """Check if state is final."""
        return state in self.final_states
    
    def get_transition_condition(
        self,
        from_state: TaskState,
        to_state: TaskState
    ) -> Optional[str]:
        """Get condition for specific transition."""
        if from_state not in self.transitions:
            return None
        
        for transition in self.transitions[from_state]:
            if transition.to_state == to_state:
                return transition.condition
        
        return None


class StepStateMachine:
    """Step state machine for individual step execution."""
    
    def __init__(self):
        self.transitions = self._build_step_transitions()
        self.final_states = {StepState.COMPLETED, StepState.FAILED, StepState.SKIPPED}
    
    def _build_step_transitions(self) -> Dict[StepState, List[StateTransition]]:
        """Build step state transition rules."""
        return {
            StepState.PENDING: [
                StateTransition(
                    StepState.PENDING,
                    StepState.RUNNING,
                    "step_started",
                    "start_step"
                ),
                StateTransition(
                    StepState.PENDING,
                    StepState.SKIPPED,
                    "step_skipped",
                    "skip_step"
                )
            ],
            StepState.RUNNING: [
                StateTransition(
                    StepState.RUNNING,
                    StepState.COMPLETED,
                    "step_completed",
                    "complete_step"
                ),
                StateTransition(
                    StepState.RUNNING,
                    StepState.FAILED,
                    "step_failed",
                    "fail_step"
                ),
                StateTransition(
                    StepState.RUNNING,
                    StepState.RETRYING,
                    "step_retry",
                    "retry_step"
                )
            ],
            StepState.RETRYING: [
                StateTransition(
                    StepState.RETRYING,
                    StepState.RUNNING,
                    "retry_started",
                    "restart_step"
                ),
                StateTransition(
                    StepState.RETRYING,
                    StepState.FAILED,
                    "max_retries_exceeded",
                    "fail_step"
                )
            ]
        }
    
    def can_transition(self, from_state: StepState, to_state: StepState) -> bool:
        """Check if step transition is allowed."""
        if from_state not in self.transitions:
            return False
        
        return any(
            transition.to_state == to_state
            for transition in self.transitions[from_state]
        )
    
    def get_valid_transitions(self, current_state: StepState) -> List[StepState]:
        """Get all valid next states for step."""
        if current_state not in self.transitions:
            return []
        
        return [transition.to_state for transition in self.transitions[current_state]]
    
    def is_final_state(self, state: StepState) -> bool:
        """Check if step state is final."""
        return state in self.final_states


# Global state machines
task_state_machine = TaskStateMachine()
step_state_machine = StepStateMachine()
