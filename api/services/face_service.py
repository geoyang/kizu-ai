"""Service for face-related operations."""

import logging
from typing import List, Optional
import numpy as np

from api.core import ModelRegistry
from api.core.registry import ModelType
from api.schemas.responses import FaceClusterResponse, SampleFace, MatchedFace
from api.stores import SupabaseVectorStore

logger = logging.getLogger(__name__)


class FaceService:
    """Service for face detection and recognition."""

    def __init__(self, vector_store: SupabaseVectorStore):
        self._store = vector_store

    async def find_face_matches(
        self,
        face_embedding: np.ndarray,
        user_id: str,
        threshold: float = 0.6,
        limit: int = 50
    ) -> List[MatchedFace]:
        """Find faces matching a given embedding."""
        results = await self._store.search(
            collection="face_embeddings",
            query_embedding=face_embedding,
            limit=limit,
            threshold=threshold,
            filters={"user_id": user_id}
        )

        matches = []
        for result in results:
            matches.append(MatchedFace(
                contact_id=result.metadata.get("contact_id"),
                cluster_id=result.metadata.get("cluster_id"),
                confidence=result.score
            ))

        return matches

    async def get_face_embedding_for_contact(
        self,
        contact_id: str,
        user_id: str
    ) -> Optional[np.ndarray]:
        """Get the representative face embedding for a contact."""
        # This would query face_clusters to get representative face
        # Then get that face's embedding
        # Placeholder - would need actual Supabase query
        return None

    async def search_by_contact(
        self,
        contact_id: str,
        user_id: str,
        threshold: float = 0.6,
        limit: int = 50
    ) -> List[dict]:
        """Find all images containing a specific person."""
        embedding = await self.get_face_embedding_for_contact(
            contact_id, user_id
        )

        if embedding is None:
            return []

        results = await self._store.search(
            collection="face_embeddings",
            query_embedding=embedding,
            limit=limit,
            threshold=threshold,
            filters={"user_id": user_id}
        )

        # Group by asset_id (one asset may have multiple face matches)
        asset_matches = {}
        for result in results:
            asset_id = result.metadata.get("asset_id", result.id)
            if asset_id not in asset_matches:
                asset_matches[asset_id] = {
                    "asset_id": asset_id,
                    "best_score": result.score,
                    "face_count": 1
                }
            else:
                asset_matches[asset_id]["face_count"] += 1
                if result.score > asset_matches[asset_id]["best_score"]:
                    asset_matches[asset_id]["best_score"] = result.score

        return list(asset_matches.values())

    async def assign_face_to_contact(
        self,
        face_id: str,
        contact_id: str,
        user_id: str
    ) -> bool:
        """Manually assign a face to a Knox contact."""
        # This would update face_embeddings table
        # Setting contact_id and confidence = 1.0
        # Placeholder for actual implementation
        logger.info(f"Assigning face {face_id} to contact {contact_id}")
        return True

    async def get_unassigned_faces(
        self,
        user_id: str,
        limit: int = 100
    ) -> List[dict]:
        """Get faces that haven't been assigned to a contact."""
        # Query face_embeddings where contact_id is null
        # Group by cluster_id
        # Placeholder
        return []
