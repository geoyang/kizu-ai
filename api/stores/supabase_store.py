"""Supabase/pgvector vector store implementation."""

import logging
from typing import List, Optional, Tuple
import numpy as np

from api.core.abstractions import (
    BaseVectorStore,
    VectorSearchResult,
    StoredVector
)

logger = logging.getLogger(__name__)


class SupabaseVectorStore(BaseVectorStore):
    """Supabase with pgvector for vector storage."""

    # Map collection names to table names
    COLLECTION_MAP = {
        "image_embeddings": "image_embeddings",
        "face_embeddings": "face_embeddings",
    }

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        service_role_key: Optional[str] = None
    ):
        self._url = supabase_url
        self._key = supabase_key
        self._service_key = service_role_key
        self._client = None

    @property
    def store_name(self) -> str:
        return "supabase-pgvector"

    async def connect(self) -> None:
        """Connect to Supabase."""
        if self._client is not None:
            return

        from supabase import create_client

        key = self._service_key or self._key
        self._client = create_client(self._url, key)
        logger.info("Connected to Supabase")

    async def disconnect(self) -> None:
        """Disconnect from Supabase."""
        self._client = None
        logger.info("Disconnected from Supabase")

    def _get_table(self, collection: str) -> str:
        """Get table name for collection."""
        return self.COLLECTION_MAP.get(collection, collection)

    def _format_vector(self, embedding: np.ndarray) -> str:
        """Format numpy array as pgvector string: [0.1, 0.2, ...]"""
        # pgvector expects the format: [x1, x2, x3, ...]
        values = ','.join(str(float(v)) for v in embedding)
        return f'[{values}]'

    async def store_embedding(
        self,
        collection: str,
        id: str,
        embedding: np.ndarray,
        metadata: Optional[dict] = None
    ) -> bool:
        """Store a single embedding."""
        await self.connect()

        table = self._get_table(collection)
        # Format embedding as pgvector string
        vector_str = self._format_vector(embedding)
        data = {
            "asset_id": id,
            "embedding": vector_str,
            **(metadata or {})
        }

        # Determine conflict columns based on table
        # face_embeddings: unique on (asset_id, face_index)
        # image_embeddings: unique on (asset_id, model_version)
        if collection == "face_embeddings":
            on_conflict = "asset_id,face_index"
        else:
            on_conflict = "asset_id,model_version"

        try:
            self._client.table(table).upsert(data, on_conflict=on_conflict).execute()
            logger.info(f"Stored embedding for {id}, dimension={len(embedding)}")
            return True
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            return False

    async def store_embeddings_batch(
        self,
        collection: str,
        items: List[Tuple[str, np.ndarray, Optional[dict]]]
    ) -> int:
        """Store multiple embeddings."""
        await self.connect()

        table = self._get_table(collection)
        records = []

        for id, embedding, metadata in items:
            records.append({
                "asset_id": id,
                "embedding": self._format_vector(embedding),
                **(metadata or {})
            })

        # Determine conflict columns based on table
        if collection == "face_embeddings":
            on_conflict = "asset_id,face_index"
        else:
            on_conflict = "asset_id,model_version"

        try:
            self._client.table(table).upsert(records, on_conflict=on_conflict).execute()
            logger.info(f"Stored batch of {len(records)} embeddings")
            return len(records)
        except Exception as e:
            logger.error(f"Failed to store batch: {e}")
            return 0

    async def search(
        self,
        collection: str,
        query_embedding: np.ndarray,
        limit: int = 10,
        threshold: Optional[float] = None,
        filters: Optional[dict] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors using pgvector."""
        await self.connect()

        # Use RPC for vector search - format embedding as pgvector string
        params = {
            "query_embedding": self._format_vector(query_embedding),
            "match_count": limit,
        }

        if threshold:
            params["match_threshold"] = threshold

        if filters and "user_id" in filters:
            params["filter_user_id"] = filters["user_id"]

        rpc_name = f"search_{collection}"
        logger.info(f"Calling RPC {rpc_name} with params: match_count={params['match_count']}, threshold={params.get('match_threshold')}, user_id={params.get('filter_user_id')}")

        try:
            result = self._client.rpc(rpc_name, params).execute()
            logger.info(f"RPC returned {len(result.data)} results")
            if result.data:
                logger.info(f"First result: {result.data[0]}")

            return [
                VectorSearchResult(
                    id=row["asset_id"],
                    score=row["similarity"],
                    metadata=row.get("metadata")
                )
                for row in result.data
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def get_embedding(
        self,
        collection: str,
        id: str
    ) -> Optional[StoredVector]:
        """Retrieve a stored embedding."""
        await self.connect()

        table = self._get_table(collection)

        try:
            result = self._client.table(table).select("*").eq("asset_id", id).single().execute()

            if result.data:
                return StoredVector(
                    id=result.data["asset_id"],
                    embedding=np.array(result.data["embedding"]),
                    metadata={"model_version": result.data.get("model_version")}
                )
        except Exception as e:
            logger.error(f"Get embedding failed: {e}")

        return None

    async def delete_embedding(
        self,
        collection: str,
        id: str
    ) -> bool:
        """Delete an embedding."""
        await self.connect()

        table = self._get_table(collection)

        try:
            self._client.table(table).delete().eq("asset_id", id).execute()
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    async def count(
        self,
        collection: str,
        filters: Optional[dict] = None
    ) -> int:
        """Count vectors in collection."""
        await self.connect()

        table = self._get_table(collection)

        try:
            query = self._client.table(table).select("*", count="exact")
            if filters and "user_id" in filters:
                query = query.eq("user_id", filters["user_id"])

            result = query.execute()
            return result.count or 0
        except Exception as e:
            logger.error(f"Count failed: {e}")
            return 0
