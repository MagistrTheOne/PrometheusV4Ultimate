"""Vector storage using Qdrant."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from libs.common.config import get_settings
from libs.common.schemas import MemoryItem, MemoryType


class VectorStore:
    """Qdrant-based vector storage."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key if self.settings.qdrant_api_key else None
        )
        self.collection_prefix = self.settings.memory.vector.get("collection_prefix", "promu_")
        self.top_k = self.settings.memory.vector.get("top_k", 8)
    
    def _get_collection_name(self, project_id: str) -> str:
        """Get collection name for project."""
        return f"{self.collection_prefix}{project_id}"
    
    async def ensure_collection(self, project_id: str, vector_size: int = 384) -> bool:
        """Ensure collection exists."""
        collection_name = self._get_collection_name(project_id)
        
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            existing_names = [col.name for col in collections.collections]
            
            if collection_name not in existing_names:
                # Create collection
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                return True
            return True
        except Exception as e:
            print(f"Error ensuring collection {collection_name}: {e}")
            return False
    
    async def save(
        self,
        project_id: str,
        content: str,
        vector: List[float],
        metadata: Dict[str, Any],
        memory_type: MemoryType = MemoryType.FACT
    ) -> UUID:
        """Save item to vector store."""
        collection_name = self._get_collection_name(project_id)
        
        # Ensure collection exists
        await self.ensure_collection(project_id, len(vector))
        
        # Generate ID
        item_id = uuid4()
        
        # Prepare payload
        payload = {
            "content": content,
            "memory_type": memory_type.value,
            "project_id": project_id,
            **metadata
        }
        
        try:
            # Insert point
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=str(item_id),
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            return item_id
        except Exception as e:
            print(f"Error saving to vector store: {e}")
            raise
    
    async def search(
        self,
        project_id: str,
        query_vector: List[float],
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[UUID, float, Dict[str, Any]]]:
        """Search vector store."""
        collection_name = self._get_collection_name(project_id)
        top_k = top_k or self.top_k
        
        try:
            # Build filter
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
                if conditions:
                    query_filter = models.Filter(must=conditions)
            
            # Search
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter
            )
            
            # Convert results
            search_results = []
            for result in results:
                item_id = UUID(result.id)
                score = result.score
                payload = result.payload
                search_results.append((item_id, score, payload))
            
            return search_results
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
    
    async def delete(self, project_id: str, item_id: UUID) -> bool:
        """Delete item from vector store."""
        collection_name = self._get_collection_name(project_id)
        
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=[str(item_id)])
            )
            return True
        except Exception as e:
            print(f"Error deleting from vector store: {e}")
            return False
    
    async def get_collection_info(self, project_id: str) -> Dict[str, Any]:
        """Get collection information."""
        collection_name = self._get_collection_name(project_id)
        
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """Check vector store health."""
        try:
            collections = self.client.get_collections()
            return True
        except Exception as e:
            print(f"Vector store health check failed: {e}")
            return False


# Global vector store instance
vector_store = VectorStore()
