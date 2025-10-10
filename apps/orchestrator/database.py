"""Database models and operations for Orchestrator."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, String, Text, create_engine, select
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from libs.common.config import get_settings
from libs.common.schemas import TaskStatus, StepStatus

Base = declarative_base()


class TaskModel(Base):
    """Tasks table."""
    __tablename__ = "tasks"
    
    id = Column(String, primary_key=True)
    goal = Column(Text, nullable=False)
    inputs = Column(Text)  # JSON string
    limits = Column(Text)  # JSON string
    project_id = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, default=TaskStatus.PENDING.value)
    rationale = Column(Text)
    error = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    steps = relationship("StepModel", back_populates="task", cascade="all, delete-orphan")
    events = relationship("EventModel", back_populates="task", cascade="all, delete-orphan")


class StepModel(Base):
    """Steps table."""
    __tablename__ = "steps"
    
    id = Column(String, primary_key=True)
    task_id = Column(String, ForeignKey("tasks.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    skill_name = Column(String, nullable=False)
    inputs = Column(Text)  # JSON string
    outputs = Column(Text)  # JSON string
    status = Column(String, nullable=False, default=StepStatus.PENDING.value)
    error = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=2)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    task = relationship("TaskModel", back_populates="steps")
    events = relationship("EventModel", back_populates="step", cascade="all, delete-orphan")


class EventModel(Base):
    """Events table for task/step lifecycle."""
    __tablename__ = "events"
    
    id = Column(String, primary_key=True)
    task_id = Column(String, ForeignKey("tasks.id"), nullable=False, index=True)
    step_id = Column(String, ForeignKey("steps.id"), nullable=True, index=True)
    event_type = Column(String, nullable=False)
    data = Column(Text)  # JSON string
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    task = relationship("TaskModel", back_populates="events")
    step = relationship("StepModel", back_populates="events")


class ArtifactModel(Base):
    """Artifacts table."""
    __tablename__ = "artifacts"
    
    id = Column(String, primary_key=True)
    task_id = Column(String, ForeignKey("tasks.id"), nullable=False, index=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    path = Column(String, nullable=False)
    size_bytes = Column(Integer, default=0)
    artifact_metadata = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)


class TaskDatabase:
    """Database operations for tasks."""
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = create_engine(
            f"sqlite:///{self.settings.sqlite_db_path}",
            echo=False
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def _get_session(self):
        """Get database session."""
        return self.SessionLocal()
    
    async def create_task(
        self,
        goal: str,
        inputs: Dict[str, Any],
        limits: Dict[str, Any],
        project_id: str
    ) -> str:
        """Create a new task."""
        task_id = str(uuid4())
        
        with self._get_session() as session:
            task = TaskModel(
                id=task_id,
                goal=goal,
                inputs=json.dumps(inputs),
                limits=json.dumps(limits),
                project_id=project_id,
                status=TaskStatus.PENDING.value
            )
            session.add(task)
            session.commit()
        
        return task_id
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID."""
        with self._get_session() as session:
            stmt = select(TaskModel).where(TaskModel.id == task_id)
            task = session.execute(stmt).scalar_one_or_none()
            
            if not task:
                return None
            
            return {
                "id": task.id,
                "goal": task.goal,
                "inputs": json.loads(task.inputs or "{}"),
                "limits": json.loads(task.limits or "{}"),
                "project_id": task.project_id,
                "status": task.status,
                "rationale": task.rationale,
                "error": task.error,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at
            }
    
    async def update_task_status(
        self,
        task_id: str,
        status: str,
        error: Optional[str] = None,
        rationale: Optional[str] = None
    ) -> bool:
        """Update task status."""
        with self._get_session() as session:
            stmt = select(TaskModel).where(TaskModel.id == task_id)
            task = session.execute(stmt).scalar_one_or_none()
            
            if not task:
                return False
            
            task.status = status
            if error:
                task.error = error
            if rationale:
                task.rationale = rationale
            
            if status == TaskStatus.RUNNING.value and not task.started_at:
                task.started_at = datetime.utcnow()
            elif status in [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value, TaskStatus.ABORTED.value]:
                task.completed_at = datetime.utcnow()
            
            session.commit()
            return True
    
    async def create_step(
        self,
        task_id: str,
        name: str,
        description: str,
        skill_name: str,
        inputs: Dict[str, Any],
        max_retries: int = 2
    ) -> str:
        """Create a new step."""
        step_id = str(uuid4())
        
        with self._get_session() as session:
            step = StepModel(
                id=step_id,
                task_id=task_id,
                name=name,
                description=description,
                skill_name=skill_name,
                inputs=json.dumps(inputs),
                max_retries=max_retries,
                status=StepStatus.PENDING.value
            )
            session.add(step)
            session.commit()
        
        return step_id
    
    async def get_task_steps(self, task_id: str) -> List[Dict[str, Any]]:
        """Get all steps for a task."""
        with self._get_session() as session:
            stmt = select(StepModel).where(StepModel.task_id == task_id).order_by(StepModel.created_at)
            steps = session.execute(stmt).scalars().all()
            
            return [
                {
                    "id": step.id,
                    "name": step.name,
                    "description": step.description,
                    "skill_name": step.skill_name,
                    "inputs": json.loads(step.inputs or "{}"),
                    "outputs": json.loads(step.outputs or "{}"),
                    "status": step.status,
                    "error": step.error,
                    "retry_count": step.retry_count,
                    "max_retries": step.max_retries,
                    "started_at": step.started_at,
                    "completed_at": step.completed_at
                }
                for step in steps
            ]
    
    async def update_step_status(
        self,
        step_id: str,
        status: str,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> bool:
        """Update step status."""
        with self._get_session() as session:
            stmt = select(StepModel).where(StepModel.id == step_id)
            step = session.execute(stmt).scalar_one_or_none()
            
            if not step:
                return False
            
            step.status = status
            if outputs:
                step.outputs = json.dumps(outputs)
            if error:
                step.error = error
            
            if status == StepStatus.RUNNING.value and not step.started_at:
                step.started_at = datetime.utcnow()
            elif status in [StepStatus.COMPLETED.value, StepStatus.FAILED.value, StepStatus.SKIPPED.value]:
                step.completed_at = datetime.utcnow()
            
            session.commit()
            return True
    
    async def increment_step_retry(self, step_id: str) -> bool:
        """Increment step retry count."""
        with self._get_session() as session:
            stmt = select(StepModel).where(StepModel.id == step_id)
            step = session.execute(stmt).scalar_one_or_none()
            
            if not step:
                return False
            
            step.retry_count += 1
            session.commit()
            return True
    
    async def log_event(
        self,
        task_id: str,
        event_type: str,
        data: Dict[str, Any],
        step_id: Optional[str] = None
    ) -> str:
        """Log an event."""
        event_id = str(uuid4())
        
        with self._get_session() as session:
            event = EventModel(
                id=event_id,
                task_id=task_id,
                step_id=step_id,
                event_type=event_type,
                data=json.dumps(data)
            )
            session.add(event)
            session.commit()
        
        return event_id
    
    async def get_task_events(self, task_id: str) -> List[Dict[str, Any]]:
        """Get all events for a task."""
        with self._get_session() as session:
            stmt = select(EventModel).where(EventModel.task_id == task_id).order_by(EventModel.timestamp)
            events = session.execute(stmt).scalars().all()
            
            return [
                {
                    "id": event.id,
                    "step_id": event.step_id,
                    "event_type": event.event_type,
                    "data": json.loads(event.data or "{}"),
                    "timestamp": event.timestamp
                }
                for event in events
            ]
    
    async def save_artifact(
        self,
        task_id: str,
        name: str,
        artifact_type: str,
        path: str,
        size_bytes: int,
        metadata: Dict[str, Any]
    ) -> str:
        """Save task artifact."""
        artifact_id = str(uuid4())
        
        with self._get_session() as session:
            artifact = ArtifactModel(
                id=artifact_id,
                task_id=task_id,
                name=name,
                type=artifact_type,
                path=path,
                size_bytes=size_bytes,
                artifact_metadata=json.dumps(metadata)
            )
            session.add(artifact)
            session.commit()
        
        return artifact_id
    
    async def get_task_artifacts(self, task_id: str) -> List[Dict[str, Any]]:
        """Get all artifacts for a task."""
        with self._get_session() as session:
            stmt = select(ArtifactModel).where(ArtifactModel.task_id == task_id).order_by(ArtifactModel.created_at)
            artifacts = session.execute(stmt).scalars().all()
            
            return [
                {
                    "id": artifact.id,
                    "name": artifact.name,
                    "type": artifact.type,
                    "path": artifact.path,
                    "size_bytes": artifact.size_bytes,
                    "metadata": json.loads(artifact.artifact_metadata or "{}"),
                    "created_at": artifact.created_at
                }
                for artifact in artifacts
            ]
    
    async def health_check(self) -> bool:
        """Check database health."""
        try:
            with self._get_session() as session:
                session.execute(select(TaskModel).limit(1))
            return True
        except Exception as e:
            print(f"Database health check failed: {e}")
            return False


# Global database instance
task_db = TaskDatabase()
