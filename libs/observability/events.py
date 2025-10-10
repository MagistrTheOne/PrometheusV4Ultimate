"""Observability events and tracing."""

import json
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict


class EventType(str, Enum):
    """Types of observability events."""
    PLAN_STARTED = "plan_started"
    STEP_RUN = "step_run"
    CRITIC_FIX = "critic_fix"
    ARTIFACT_SAVED = "artifact_saved"
    COST = "cost"
    TASK_CREATED = "task_created"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    SKILL_REGISTERED = "skill_registered"
    MEMORY_SAVED = "memory_saved"
    MEMORY_SEARCHED = "memory_searched"


@dataclass
class TraceEvent:
    """A single trace event."""
    timestamp: str
    trace_id: str
    event_type: EventType
    task_id: Optional[str] = None
    step_id: Optional[str] = None
    skill_name: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    cost: Optional[Dict[str, float]] = None
    decision: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class EventCollector:
    """Collects and manages trace events."""
    
    def __init__(self, output_dir: str = ".promu/traces"):
        self.output_dir = output_dir
        self.events: List[TraceEvent] = []
        self._ensure_output_dir()
    
    def _ensure_output_dir(self) -> None:
        """Ensure output directory exists."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
    
    def add_event(
        self,
        event_type: EventType,
        trace_id: str,
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
        skill_name: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        cost: Optional[Dict[str, float]] = None,
        decision: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new trace event."""
        event = TraceEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            trace_id=trace_id,
            event_type=event_type,
            task_id=task_id,
            step_id=step_id,
            skill_name=skill_name,
            metrics=metrics,
            cost=cost,
            decision=decision,
            metadata=metadata
        )
        
        self.events.append(event)
        
        # Write to file immediately (for real-time monitoring)
        self._write_event_to_file(event)
    
    def _write_event_to_file(self, event: TraceEvent) -> None:
        """Write event to JSONL file."""
        import os
        filename = f"traces_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(event.to_json() + '\n')
    
    def get_events_by_task(self, task_id: str) -> List[TraceEvent]:
        """Get all events for a specific task."""
        return [event for event in self.events if event.task_id == task_id]
    
    def get_events_by_trace(self, trace_id: str) -> List[TraceEvent]:
        """Get all events for a specific trace."""
        return [event for event in self.events if event.trace_id == trace_id]
    
    def get_events_by_type(self, event_type: EventType) -> List[TraceEvent]:
        """Get all events of a specific type."""
        return [event for event in self.events if event.event_type == event_type]
    
    def export_events(self, filepath: str, task_id: Optional[str] = None) -> None:
        """Export events to JSONL file."""
        events_to_export = self.events
        if task_id:
            events_to_export = self.get_events_by_task(task_id)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for event in events_to_export:
                f.write(event.to_json() + '\n')
    
    def get_metrics_summary(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics summary for events."""
        events = self.events
        if task_id:
            events = self.get_events_by_task(task_id)
        
        # Calculate metrics
        total_events = len(events)
        event_types = {}
        total_cost = 0.0
        total_latency = 0.0
        
        for event in events:
            # Count event types
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            
            # Sum costs
            if event.cost:
                total_cost += event.cost.get('usd', 0.0)
            
            # Sum latency
            if event.metrics:
                total_latency += event.metrics.get('latency_ms', 0.0)
        
        return {
            "total_events": total_events,
            "event_types": event_types,
            "total_cost_usd": total_cost,
            "total_latency_ms": total_latency,
            "average_latency_ms": total_latency / total_events if total_events > 0 else 0
        }


class MetricsCollector:
    """Collects performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def record_metric(self, name: str, value: float) -> None:
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_percentile(self, name: str, percentile: float) -> Optional[float]:
        """Get percentile value for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return None
        
        values = sorted(self.metrics[name])
        index = int(len(values) * percentile / 100)
        return values[min(index, len(values) - 1)]
    
    def get_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = self.metrics[name]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "p50": self.get_percentile(name, 50),
            "p95": self.get_percentile(name, 95),
            "p99": self.get_percentile(name, 99)
        }


# Global instances
event_collector = EventCollector()
metrics_collector = MetricsCollector()
