"""Search service for natural language image queries."""

import logging
from typing import List, Optional
import time

from api.core import ModelRegistry
from api.core.registry import ModelType
from api.schemas.requests import SearchRequest, SearchFilters
from api.schemas.responses import SearchResult, MatchedFace
from api.stores import SupabaseVectorStore

logger = logging.getLogger(__name__)


class SearchService:
    """Service for semantic image search."""

    def __init__(self, vector_store: SupabaseVectorStore):
        self._store = vector_store

    async def search(
        self,
        request: SearchRequest,
        user_id: str
    ) -> tuple[List[SearchResult], float]:
        """
        Execute a semantic search query.

        Returns:
            Tuple of (results, processing_time_ms)
        """
        start_time = time.time()

        logger.info(f"Search request: query='{request.query}', user_id={user_id}, limit={request.limit}, threshold={request.threshold}")

        # Get embedder
        embedder = ModelRegistry.get(ModelType.EMBEDDER)

        # Embed the query text
        query_embedding = embedder.embed_text(request.query)
        logger.info(f"Query embedding generated, dimension={len(query_embedding.embedding)}")

        # Search vector store
        filters = {"user_id": user_id}
        if request.filters:
            filters.update(self._build_filters(request.filters))

        logger.info(f"Searching with filters: {filters}")

        vector_results = await self._store.search(
            collection="image_embeddings",
            query_embedding=query_embedding.embedding,
            limit=request.limit,
            threshold=request.threshold,
            filters=filters
        )
        logger.info(f"Vector search returned {len(vector_results)} results")

        # Build results with thumbnail URLs
        results = []

        # Fetch web_uri for all matching assets in one query
        asset_ids = [vr.id for vr in vector_results]
        thumbnails = await self._get_thumbnails(asset_ids)

        for vr in vector_results:
            result = SearchResult(
                asset_id=vr.id,
                similarity=vr.score,
                thumbnail_url=thumbnails.get(vr.id),
            )

            # Optionally add descriptions
            if request.include_descriptions:
                result.description = await self._get_description(vr.id)

            # Add matched faces if searching for people
            if request.filters and request.filters.people:
                result.matched_faces = await self._get_matched_faces(
                    vr.id,
                    request.filters.people
                )

            results.append(result)

        elapsed_ms = (time.time() - start_time) * 1000
        return results, elapsed_ms

    def _build_filters(self, filters: SearchFilters) -> dict:
        """Convert search filters to store filters."""
        store_filters = {}

        if filters.date_start:
            store_filters["date_start"] = filters.date_start.isoformat()
        if filters.date_end:
            store_filters["date_end"] = filters.date_end.isoformat()
        if filters.locations:
            store_filters["locations"] = filters.locations
        if filters.media_type:
            store_filters["media_type"] = filters.media_type

        return store_filters

    async def _get_thumbnails(self, asset_ids: List[str]) -> dict:
        """Fetch web_uri for assets from the assets table."""
        if not asset_ids:
            return {}

        try:
            from supabase import create_client
            from api.config import settings

            supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)
            result = supabase.table('assets').select('id, web_uri, thumbnail, path').in_('id', asset_ids).execute()

            thumbnails = {}
            for row in result.data:
                # Prefer web_uri, then thumbnail, then path
                url = row.get('web_uri') or row.get('thumbnail') or row.get('path')
                if url:
                    thumbnails[row['id']] = url

            return thumbnails
        except Exception as e:
            logger.error(f"Failed to fetch thumbnails: {e}")
            return {}

    async def _get_description(self, asset_id: str) -> Optional[str]:
        """Get stored description for an asset."""
        # This would query the image_descriptions table
        # Placeholder for now
        return None

    async def _get_matched_faces(
        self,
        asset_id: str,
        contact_ids: List[str]
    ) -> List[MatchedFace]:
        """Get face matches for an asset."""
        # This would query face_embeddings for matching contacts
        # Placeholder for now
        return []

    async def search_by_face(
        self,
        face_embedding: List[float],
        user_id: str,
        limit: int = 50,
        threshold: float = 0.6
    ) -> List[SearchResult]:
        """Search for images containing a specific face."""
        import numpy as np

        embedding = np.array(face_embedding)

        vector_results = await self._store.search(
            collection="face_embeddings",
            query_embedding=embedding,
            limit=limit,
            threshold=threshold,
            filters={"user_id": user_id}
        )

        return [
            SearchResult(asset_id=vr.id, similarity=vr.score)
            for vr in vector_results
        ]

    async def search_by_object(
        self,
        object_class: str,
        user_id: str,
        min_confidence: float = 0.5,
        limit: int = 50
    ) -> tuple[List[SearchResult], float]:
        """
        Search for images containing a specific detected object.

        Returns:
            Tuple of (results, processing_time_ms)
        """
        start_time = time.time()

        logger.info(f"Object search: class='{object_class}', user_id={user_id}, min_confidence={min_confidence}")

        try:
            from supabase import create_client
            from api.config import settings

            supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

            # Search detected_objects table with case-insensitive match
            result = supabase.table('detected_objects')\
                .select('asset_id, object_class, confidence')\
                .eq('user_id', user_id)\
                .ilike('object_class', f'%{object_class}%')\
                .gte('confidence', min_confidence)\
                .order('confidence', desc=True)\
                .limit(limit)\
                .execute()

            logger.info(f"Object search found {len(result.data)} matches")

            # Get unique asset_ids (one object might be detected multiple times)
            seen_assets = set()
            unique_results = []
            for row in result.data:
                if row['asset_id'] not in seen_assets:
                    seen_assets.add(row['asset_id'])
                    unique_results.append(row)

            # Get thumbnails for matched assets
            asset_ids = [r['asset_id'] for r in unique_results]
            thumbnails = await self._get_thumbnails(asset_ids)

            results = [
                SearchResult(
                    asset_id=r['asset_id'],
                    similarity=r['confidence'],  # Use confidence as similarity
                    thumbnail_url=thumbnails.get(r['asset_id']),
                    matched_objects=[r['object_class']]
                )
                for r in unique_results
            ]

            elapsed_ms = (time.time() - start_time) * 1000
            return results, elapsed_ms

        except Exception as e:
            logger.error(f"Object search failed: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            return [], elapsed_ms
