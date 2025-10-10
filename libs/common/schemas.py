"""Pydantic schemas for PrometheusULTIMATE v4."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Step execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ChatRole(str, Enum):
    """Chat message role."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class TraceEventKind(str, Enum):
    """Trace event types."""
    TASK_START = "task_start"
    TASK_END = "task_end"
    STEP_START = "step_start"
    STEP_END = "step_end"
    CRITIC_REVIEW = "critic_review"
    MEMORY_ACCESS = "memory_access"
    SKILL_EXECUTION = "skill_execution"
    COST_UPDATE = "cost_update"
    ERROR = "error"


class ModelStatus(str, Enum):
    """Model availability status."""
    TRAINING = "training"
    READY = "ready"
    DISABLED = "disabled"
    ERROR = "error"


class ArtifactType(str, Enum):
    """Artifact types."""
    FILE = "file"
    DATA = "data"
    CODE = "code"
    REPORT = "report"
    IMAGE = "image"
    TABLE = "table"


class MemoryType(str, Enum):
    """Memory item types."""
    FACT = "fact"
    ARTIFACT = "artifact"
    NOTE = "note"
    POLICY = "policy"
    EVENT = "event"


class ChatMessage(BaseModel):
    """Chat message."""
    role: ChatRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Step(BaseModel):
    """Execution step."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    skill_name: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metrics: Optional["PerformanceMetrics"] = None


class Task(BaseModel):
    """Task definition and state."""
    id: UUID = Field(default_factory=uuid4)
    goal: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    limits: Dict[str, Union[int, float]] = Field(default_factory=dict)
    project_id: str
    status: TaskStatus = TaskStatus.PENDING
    steps: List[Step] = Field(default_factory=list)
    artifacts: List["Artifact"] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    rationale: Optional[str] = None
    cost_metrics: Optional["CostMetrics"] = None


class TraceEvent(BaseModel):
    """Trace event for observability."""
    trace_id: UUID = Field(default_factory=uuid4)
    task_id: UUID
    step_id: Optional[UUID] = None
    kind: TraceEventKind
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default_factory=dict)
    metrics: Optional["PerformanceMetrics"] = None


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    family: str
    size_b: float
    context: int
    format: str
    runtime: str
    quant: List[str] = Field(default_factory=list)
    status: ModelStatus = ModelStatus.TRAINING
    eos: str = "</s>"


class SkillPermissions(BaseModel):
    """Skill execution permissions."""
    fs: str = "none"  # read|write|none
    net: str = "off"  # on|off
    cpu_ms: int = 800
    ram_mb: int = 512
    time_s: int = 30


class SkillSpec(BaseModel):
    """Skill specification."""
    name: str
    version: str
    description: str
    inputs: Dict[str, str] = Field(default_factory=dict)
    outputs: Dict[str, str] = Field(default_factory=dict)
    permissions: SkillPermissions = Field(default_factory=SkillPermissions)
    limits: Dict[str, int] = Field(default_factory=dict)


class Artifact(BaseModel):
    """Task artifact."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    type: ArtifactType
    path: str
    size_bytes: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryItem(BaseModel):
    """Memory storage item."""
    id: UUID = Field(default_factory=uuid4)
    project_id: str
    type: MemoryType
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CostMetrics(BaseModel):
    """Cost tracking metrics."""
    cpu_ms: int = 0
    ram_mb: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


class PerformanceMetrics(BaseModel):
    """Performance metrics."""
    cpu_ms: int = 0
    ram_mb: int = 0
    latency_ms: int = 0
    success: bool = True
    error_rate: float = 0.0


# Update forward references
Step.model_rebuild()
Task.model_rebuild()
TraceEvent.model_rebuild()
