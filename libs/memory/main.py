"""Memory service main application."""

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .kv_store import kv_store
from .vector_store import vector_store

app = FastAPI(
    title="PrometheusULTIMATE v4 Memory Service",
    description="Memory storage and retrieval service",
    version="0.9.0"
)


class MemorySaveRequest(BaseModel):
    """Memory save request."""
    project_id: str
    kind: str
    content: str
    metadata: Dict = {}


class MemorySearchRequest(BaseModel):
    """Memory search request."""
    project_id: str
    query: str
    top_k: int = 5
    filters: Dict = {}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    vector_healthy = await vector_store.health_check()
    kv_healthy = await kv_store.health_check()
    
    return {
        "status": "healthy" if vector_healthy and kv_healthy else "unhealthy",
        "service": "memory",
        "vector_store": vector_healthy,
        "kv_store": kv_healthy
    }


@app.post("/save")
async def save_memory(request: MemorySaveRequest):
    """Save item to memory."""
    try:
        # For now, save to KV store only
        # TODO: Add vector embedding and save to vector store
        
        if request.kind == "fact":
            item_id = await kv_store.save_fact(
                project_id=request.project_id,
                content=request.content,
                metadata=request.metadata
            )
        elif request.kind == "artifact":
            item_id = await kv_store.save_artifact(
                project_id=request.project_id,
                name=request.metadata.get("name", "unnamed"),
                artifact_type=request.metadata.get("type", "unknown"),
                path=request.metadata.get("path", ""),
                size_bytes=request.metadata.get("size_bytes", 0),
                metadata=request.metadata
            )
        elif request.kind == "policy":
            item_id = await kv_store.save_policy(
                project_id=request.project_id,
                name=request.metadata.get("name", "unnamed"),
                content=request.content,
                metadata=request.metadata
            )
        elif request.kind == "event":
            item_id = await kv_store.save_event(
                project_id=request.project_id,
                event_type=request.metadata.get("event_type", "unknown"),
                content=request.content,
                metadata=request.metadata
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown memory kind: {request.kind}")
        
        return {"status": "saved", "id": str(item_id)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search_memory(
    project_id: str = Query(...),
    q: str = Query(...),
    top_k: int = Query(5),
    table: str = Query("facts")
):
    """Search memory."""
    try:
        results = await kv_store.search(
            project_id=project_id,
            query=q,
            table=table,
            limit=top_k
        )
        
        return {
            "results": results,
            "count": len(results),
            "query": q,
            "project_id": project_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/facts/{project_id}")
async def get_facts(project_id: str, limit: int = Query(100)):
    """Get facts for project."""
    try:
        facts = await kv_store.get_facts(project_id=project_id, limit=limit)
        return {"facts": facts, "count": len(facts)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/artifacts/{project_id}")
async def get_artifacts(project_id: str, limit: int = Query(100)):
    """Get artifacts for project."""
    try:
        artifacts = await kv_store.get_artifacts(project_id=project_id, limit=limit)
        return {"artifacts": artifacts, "count": len(artifacts)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/policies/{project_id}")
async def get_policies(project_id: str, active_only: bool = Query(True)):
    """Get policies for project."""
    try:
        policies = await kv_store.get_policies(
            project_id=project_id,
            active_only=active_only
        )
        return {"policies": policies, "count": len(policies)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/events/{project_id}")
async def get_events(
    project_id: str,
    event_type: Optional[str] = Query(None),
    limit: int = Query(100)
):
    """Get events for project."""
    try:
        events = await kv_store.get_events(
            project_id=project_id,
            event_type=event_type,
            limit=limit
        )
        return {"events": events, "count": len(events)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    from libs.common.config import get_settings
    
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.memory_port,
        reload=True
    )
