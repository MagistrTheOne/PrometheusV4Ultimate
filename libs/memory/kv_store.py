"""Key-value storage using SQLite."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import (
    Boolean, Column, DateTime, Integer, String, Text, create_engine, select
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from libs.common.config import get_settings
from libs.common.schemas import MemoryItem, MemoryType

Base = declarative_base()


class FactModel(Base):
    """Facts table."""
    __tablename__ = "facts"
    
    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False)
    metadata = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ArtifactModel(Base):
    """Artifacts table."""
    __tablename__ = "artifacts"
    
    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    path = Column(String, nullable=False)
    size_bytes = Column(Integer, default=0)
    metadata = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)


class PolicyModel(Base):
    """Policies table."""
    __tablename__ = "policies"
    
    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    metadata = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EventModel(Base):
    """Events table."""
    __tablename__ = "events"
    
    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    metadata = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)


class KVStore:
    """SQLite-based key-value storage."""
    
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
    
    async def save_fact(
        self,
        project_id: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> UUID:
        """Save fact."""
        fact_id = UUID()
        
        with self._get_session() as session:
            fact = FactModel(
                id=str(fact_id),
                project_id=project_id,
                content=content,
                metadata=json.dumps(metadata)
            )
            session.add(fact)
            session.commit()
        
        return fact_id
    
    async def get_facts(
        self,
        project_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get facts for project."""
        with self._get_session() as session:
            stmt = select(FactModel).where(
                FactModel.project_id == project_id
            ).order_by(FactModel.created_at.desc()).limit(limit)
            
            facts = session.execute(stmt).scalars().all()
            
            return [
                {
                    "id": fact.id,
                    "content": fact.content,
                    "metadata": json.loads(fact.metadata or "{}"),
                    "created_at": fact.created_at,
                    "updated_at": fact.updated_at
                }
                for fact in facts
            ]
    
    async def save_artifact(
        self,
        project_id: str,
        name: str,
        artifact_type: str,
        path: str,
        size_bytes: int,
        metadata: Dict[str, Any]
    ) -> UUID:
        """Save artifact."""
        artifact_id = UUID()
        
        with self._get_session() as session:
            artifact = ArtifactModel(
                id=str(artifact_id),
                project_id=project_id,
                name=name,
                type=artifact_type,
                path=path,
                size_bytes=size_bytes,
                metadata=json.dumps(metadata)
            )
            session.add(artifact)
            session.commit()
        
        return artifact_id
    
    async def get_artifacts(
        self,
        project_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get artifacts for project."""
        with self._get_session() as session:
            stmt = select(ArtifactModel).where(
                ArtifactModel.project_id == project_id
            ).order_by(ArtifactModel.created_at.desc()).limit(limit)
            
            artifacts = session.execute(stmt).scalars().all()
            
            return [
                {
                    "id": artifact.id,
                    "name": artifact.name,
                    "type": artifact.type,
                    "path": artifact.path,
                    "size_bytes": artifact.size_bytes,
                    "metadata": json.loads(artifact.metadata or "{}"),
                    "created_at": artifact.created_at
                }
                for artifact in artifacts
            ]
    
    async def save_policy(
        self,
        project_id: str,
        name: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> UUID:
        """Save policy."""
        policy_id = UUID()
        
        with self._get_session() as session:
            policy = PolicyModel(
                id=str(policy_id),
                project_id=project_id,
                name=name,
                content=content,
                metadata=json.dumps(metadata)
            )
            session.add(policy)
            session.commit()
        
        return policy_id
    
    async def get_policies(
        self,
        project_id: str,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Get policies for project."""
        with self._get_session() as session:
            stmt = select(PolicyModel).where(
                PolicyModel.project_id == project_id
            )
            
            if active_only:
                stmt = stmt.where(PolicyModel.is_active == True)
            
            stmt = stmt.order_by(PolicyModel.created_at.desc())
            
            policies = session.execute(stmt).scalars().all()
            
            return [
                {
                    "id": policy.id,
                    "name": policy.name,
                    "content": policy.content,
                    "is_active": policy.is_active,
                    "metadata": json.loads(policy.metadata or "{}"),
                    "created_at": policy.created_at,
                    "updated_at": policy.updated_at
                }
                for policy in policies
            ]
    
    async def save_event(
        self,
        project_id: str,
        event_type: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> UUID:
        """Save event."""
        event_id = UUID()
        
        with self._get_session() as session:
            event = EventModel(
                id=str(event_id),
                project_id=project_id,
                event_type=event_type,
                content=content,
                metadata=json.dumps(metadata)
            )
            session.add(event)
            session.commit()
        
        return event_id
    
    async def get_events(
        self,
        project_id: str,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get events for project."""
        with self._get_session() as session:
            stmt = select(EventModel).where(
                EventModel.project_id == project_id
            )
            
            if event_type:
                stmt = stmt.where(EventModel.event_type == event_type)
            
            stmt = stmt.order_by(EventModel.created_at.desc()).limit(limit)
            
            events = session.execute(stmt).scalars().all()
            
            return [
                {
                    "id": event.id,
                    "event_type": event.event_type,
                    "content": event.content,
                    "metadata": json.loads(event.metadata or "{}"),
                    "created_at": event.created_at
                }
                for event in events
            ]
    
    async def search(
        self,
        project_id: str,
        query: str,
        table: str = "facts",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search across tables."""
        results = []
        
        with self._get_session() as session:
            if table == "facts":
                stmt = select(FactModel).where(
                    FactModel.project_id == project_id,
                    FactModel.content.contains(query)
                ).limit(limit)
                
                items = session.execute(stmt).scalars().all()
                
                results = [
                    {
                        "id": item.id,
                        "type": "fact",
                        "content": item.content,
                        "metadata": json.loads(item.metadata or "{}"),
                        "created_at": item.created_at
                    }
                    for item in items
                ]
            
            elif table == "artifacts":
                stmt = select(ArtifactModel).where(
                    ArtifactModel.project_id == project_id,
                    ArtifactModel.name.contains(query)
                ).limit(limit)
                
                items = session.execute(stmt).scalars().all()
                
                results = [
                    {
                        "id": item.id,
                        "type": "artifact",
                        "name": item.name,
                        "content": item.path,
                        "metadata": json.loads(item.metadata or "{}"),
                        "created_at": item.created_at
                    }
                    for item in items
                ]
        
        return results
    
    async def health_check(self) -> bool:
        """Check KV store health."""
        try:
            with self._get_session() as session:
                session.execute(select(FactModel).limit(1))
            return True
        except Exception as e:
            print(f"KV store health check failed: {e}")
            return False


# Global KV store instance
kv_store = KVStore()
