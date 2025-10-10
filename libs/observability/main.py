"""Observability service main application."""

from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from .events import EventCollector, EventType, TraceEvent, event_collector, metrics_collector

app = FastAPI(title="PrometheusULTIMATE Observability", version="1.0.0")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "observability"}


@app.get("/traces")
async def get_traces(
    task_id: Optional[str] = Query(None, description="Filter by task ID"),
    trace_id: Optional[str] = Query(None, description="Filter by trace ID"),
    event_type: Optional[EventType] = Query(None, description="Filter by event type"),
    limit: int = Query(100, description="Maximum number of events to return")
):
    """Get trace events with optional filtering."""
    events = event_collector.events
    
    # Apply filters
    if task_id:
        events = [e for e in events if e.task_id == task_id]
    if trace_id:
        events = [e for e in events if e.trace_id == trace_id]
    if event_type:
        events = [e for e in events if e.event_type == event_type]
    
    # Limit results
    events = events[-limit:] if limit > 0 else events
    
    return {
        "events": [event.to_dict() for event in events],
        "total": len(events)
    }


@app.get("/traces/{task_id}")
async def get_task_traces(task_id: str):
    """Get all trace events for a specific task."""
    events = event_collector.get_events_by_task(task_id)
    return {
        "task_id": task_id,
        "events": [event.to_dict() for event in events],
        "total": len(events)
    }


@app.get("/metrics")
async def get_metrics(
    task_id: Optional[str] = Query(None, description="Filter by task ID"),
    metric_name: Optional[str] = Query(None, description="Specific metric name")
):
    """Get metrics summary."""
    if task_id:
        summary = event_collector.get_metrics_summary(task_id)
    else:
        summary = event_collector.get_metrics_summary()
    
    if metric_name:
        metric_summary = metrics_collector.get_summary(metric_name)
        summary["metric_details"] = {metric_name: metric_summary}
    
    return summary


@app.get("/metrics/{metric_name}")
async def get_metric_details(metric_name: str):
    """Get detailed metrics for a specific metric name."""
    summary = metrics_collector.get_summary(metric_name)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Metric '{metric_name}' not found")
    
    return {
        "metric_name": metric_name,
        "summary": summary
    }


@app.post("/events")
async def add_event(event_data: dict):
    """Add a new trace event."""
    try:
        event_collector.add_event(
            event_type=EventType(event_data["event_type"]),
            trace_id=event_data["trace_id"],
            task_id=event_data.get("task_id"),
            step_id=event_data.get("step_id"),
            skill_name=event_data.get("skill_name"),
            metrics=event_data.get("metrics"),
            cost=event_data.get("cost"),
            decision=event_data.get("decision"),
            metadata=event_data.get("metadata")
        )
        return {"status": "success", "message": "Event added"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/export")
async def export_traces(
    task_id: Optional[str] = Query(None, description="Export specific task"),
    format: str = Query("jsonl", description="Export format")
):
    """Export trace events."""
    if format != "jsonl":
        raise HTTPException(status_code=400, detail="Only JSONL format supported")
    
    # Generate filename
    from datetime import datetime
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"traces_export_{timestamp}.jsonl"
    if task_id:
        filename = f"traces_task_{task_id}_{timestamp}.jsonl"
    
    # Export to file
    filepath = f".promu/traces/{filename}"
    event_collector.export_events(filepath, task_id)
    
    return {
        "status": "success",
        "filename": filename,
        "filepath": filepath,
        "message": f"Traces exported to {filepath}"
    }


@app.get("/dashboard")
async def get_dashboard_data():
    """Get data for observability dashboard."""
    # Get recent events
    recent_events = event_collector.events[-50:] if event_collector.events else []
    
    # Get metrics summary
    metrics_summary = event_collector.get_metrics_summary()
    
    # Get event type distribution
    event_types = {}
    for event in recent_events:
        event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
    
    # Get recent tasks
    recent_tasks = {}
    for event in recent_events:
        if event.task_id:
            if event.task_id not in recent_tasks:
                recent_tasks[event.task_id] = {
                    "task_id": event.task_id,
                    "first_event": event.timestamp,
                    "last_event": event.timestamp,
                    "event_count": 0,
                    "event_types": set()
                }
            recent_tasks[event.task_id]["last_event"] = event.timestamp
            recent_tasks[event.task_id]["event_count"] += 1
            recent_tasks[event.task_id]["event_types"].add(event.event_type)
    
    # Convert sets to lists for JSON serialization
    for task_data in recent_tasks.values():
        task_data["event_types"] = list(task_data["event_types"])
    
    return {
        "recent_events": [event.to_dict() for event in recent_events],
        "metrics_summary": metrics_summary,
        "event_type_distribution": event_types,
        "recent_tasks": list(recent_tasks.values()),
        "total_events": len(event_collector.events)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8095)
