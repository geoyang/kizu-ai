"""Search service for natural language image queries."""

import logging
from datetime import datetime
from typing import List, Optional
import time

from api.core import ModelRegistry
from api.core.registry import ModelType
from api.schemas.requests import SearchRequest, SearchFilters
from api.schemas.responses import SearchResult, MatchedFace
from api.stores import SupabaseVectorStore
from api.utils.date_parser import parse_query, ParsedQuery

logger = logging.getLogger(__name__)


class SearchService:
    """Service for semantic image search."""

    def __init__(self, vector_store: SupabaseVectorStore):
        self._store = vector_store

    async def search(
        self,
        request: SearchRequest,
        user_id: str
    ) -> tuple[List[SearchResult], float, dict]:
        """
        Execute a semantic search query with NLP date and location parsing.

        Supports queries like:
        - "dogs at the beach in March 2025"
        - "birthday party last summer"
        - "sunset in Paris from 2024"

        Returns:
            Tuple of (results, processing_time_ms, parsed_query_info)
        """
        start_time = time.time()

        logger.info(f"Search request: query='{request.query}', user_id={user_id}, limit={request.limit}, threshold={request.threshold}")

        # Parse dates and locations from natural language query
        parsed = parse_query(request.query)
        logger.info(f"Parsed query: semantic='{parsed.semantic_query}', dates={parsed.date_start} to {parsed.date_end}, locations={parsed.locations}")

        # Get embedder
        embedder = ModelRegistry.get(ModelType.EMBEDDER)

        # Embed the semantic part of the query (without date/location terms)
        query_embedding = embedder.embed_text(parsed.semantic_query if parsed.semantic_query else request.query)
        logger.info(f"Query embedding generated, dimension={len(query_embedding.embedding)}")

        # Determine if we need to fetch extra results for post-filtering
        needs_filtering = parsed.date_start or parsed.date_end or parsed.locations
        fetch_limit = request.limit * 4 if needs_filtering else request.limit

        vector_results = await self._store.search(
            collection="image_embeddings",
            query_embedding=query_embedding.embedding,
            limit=fetch_limit,
            threshold=request.threshold,
            filters={"user_id": user_id}  # Only user_id is supported by RPC
        )
        logger.info(f"Vector search returned {len(vector_results)} results")

        # Apply date filtering if dates were parsed from query
        if parsed.date_start or parsed.date_end:
            vector_results = await self._filter_by_date(
                vector_results, parsed.date_start, parsed.date_end, fetch_limit
            )
            logger.info(f"After date filtering: {len(vector_results)} results")

        # Apply location filtering if locations were parsed from query
        matched_locations = {}
        if parsed.locations:
            vector_results, matched_locations = await self._filter_by_location(
                vector_results, parsed.locations, request.limit
            )
            logger.info(f"After location filtering: {len(vector_results)} results, matched locations: {len(matched_locations)}")

        # Limit results to requested amount
        vector_results = vector_results[:request.limit]

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
                matched_location=matched_locations.get(vr.id),
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

        # Build parsed query info for response
        query_parsed = {
            "semantic_query": parsed.semantic_query,
            "locations": parsed.locations if parsed.locations else None,
            "date_start": parsed.date_start.isoformat() if parsed.date_start else None,
            "date_end": parsed.date_end.isoformat() if parsed.date_end else None,
        }

        return results, elapsed_ms, query_parsed

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

    async def _filter_by_date(
        self,
        results: List,
        date_start: Optional[datetime],
        date_end: Optional[datetime],
        limit: int
    ) -> List:
        """Filter vector search results by asset creation date."""
        if not results:
            return results

        try:
            from supabase import create_client
            from api.config import settings

            supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)
            asset_ids = [r.id for r in results]

            # Get creation dates for all matched assets
            query = supabase.table('assets') \
                .select('id, created_at') \
                .in_('id', asset_ids)

            assets_result = query.execute()

            # Build date lookup
            asset_dates = {}
            for row in assets_result.data:
                created_at = row.get('created_at')
                if created_at:
                    # Parse ISO datetime string
                    if isinstance(created_at, str):
                        try:
                            asset_dates[row['id']] = datetime.fromisoformat(
                                created_at.replace('Z', '+00:00')
                            ).replace(tzinfo=None)
                        except ValueError:
                            pass

            # Filter results by date
            filtered = []
            for result in results:
                asset_date = asset_dates.get(result.id)
                if asset_date is None:
                    continue  # Skip assets without dates

                # Check date range
                if date_start and asset_date < date_start:
                    continue
                if date_end and asset_date > date_end:
                    continue

                filtered.append(result)
                if len(filtered) >= limit:
                    break

            return filtered

        except Exception as e:
            logger.error(f"Date filtering failed: {e}")
            return results[:limit]  # Fall back to unfiltered results

    async def _filter_by_location(
        self,
        results: List,
        locations: List[str],
        limit: int
    ) -> tuple[List, dict]:
        """
        Filter vector search results by asset location metadata.

        Returns:
            Tuple of (filtered_results, location_map) where location_map
            is {asset_id: location_name} for matched assets.
        """
        if not results or not locations:
            return results, {}

        try:
            from supabase import create_client
            from api.config import settings

            supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)
            asset_ids = [r.id for r in results]

            # Get location data for all matched assets
            assets_result = supabase.table('assets') \
                .select('id, location_name, latitude, longitude') \
                .in_('id', asset_ids) \
                .execute()

            # Build location lookup (preserve original case for display)
            asset_locations = {}
            asset_locations_lower = {}
            for row in assets_result.data:
                location_name = row.get('location_name') or ''
                asset_locations[row['id']] = location_name  # Original case
                asset_locations_lower[row['id']] = location_name.lower()

            # Normalize search locations
            search_terms = [loc.lower() for loc in locations]

            # Filter results by location match
            filtered = []
            matched_locations = {}
            for result in results:
                asset_location_lower = asset_locations_lower.get(result.id, '')
                asset_location = asset_locations.get(result.id, '')

                # Check if any search term matches the location
                matches = any(
                    term in asset_location_lower or asset_location_lower in term
                    for term in search_terms
                )

                if matches:
                    filtered.append(result)
                    matched_locations[result.id] = asset_location
                    if len(filtered) >= limit:
                        break

            # If no matches found with location filter, return semantic results
            if not filtered:
                logger.info(f"No location matches found for {locations}, returning semantic results")
                return results[:limit], {}

            return filtered, matched_locations

        except Exception as e:
            logger.error(f"Location filtering failed: {e}")
            return results[:limit], {}  # Fall back to unfiltered results

    async def _get_thumbnails(self, asset_ids: List[str]) -> dict:
        """Fetch correctly-oriented thumbnail URLs for assets.

        Uses album_assets.thumbnail_uri which contains EXIF-corrected thumbnails.
        Falls back to assets table if album_assets doesn't have the asset.
        """
        if not asset_ids:
            return {}

        try:
            from supabase import create_client
            from api.config import settings

            supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)
            thumbnails = {}

            # First try album_assets which has properly oriented thumbnails
            album_result = supabase.table('album_assets') \
                .select('asset_id, thumbnail_uri, asset_uri') \
                .in_('asset_id', asset_ids) \
                .execute()

            for row in album_result.data:
                # Prefer thumbnail_uri (EXIF-corrected), then asset_uri
                url = row.get('thumbnail_uri') or row.get('asset_uri')
                if url and url.startswith('http'):
                    thumbnails[row['asset_id']] = url

            # For any assets not found in album_assets, try assets table
            missing_ids = [aid for aid in asset_ids if aid not in thumbnails]
            if missing_ids:
                assets_result = supabase.table('assets') \
                    .select('id, web_uri, thumbnail, path') \
                    .in_('id', missing_ids) \
                    .execute()

                for row in assets_result.data:
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

    async def get_face_embedding(
        self,
        contact_id: Optional[str] = None,
        cluster_id: Optional[str] = None
    ) -> Optional[List[float]]:
        """Get representative face embedding for a contact or cluster."""
        try:
            from supabase import create_client
            from api.config import settings

            supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

            if cluster_id:
                # Get centroid embedding from the cluster
                result = supabase.table('face_clusters') \
                    .select('centroid_embedding') \
                    .eq('id', cluster_id) \
                    .single() \
                    .execute()

                if result.data and result.data.get('centroid_embedding'):
                    return result.data['centroid_embedding']

                # Fall back to first face in cluster
                face_result = supabase.table('face_embeddings') \
                    .select('embedding') \
                    .eq('cluster_id', cluster_id) \
                    .limit(1) \
                    .execute()

                if face_result.data:
                    return face_result.data[0].get('embedding')

            if contact_id:
                # Get face embedding linked to contact
                result = supabase.table('face_clusters') \
                    .select('centroid_embedding') \
                    .eq('contact_id', contact_id) \
                    .limit(1) \
                    .execute()

                if result.data and result.data[0].get('centroid_embedding'):
                    return result.data[0]['centroid_embedding']

            return None

        except Exception as e:
            logger.error(f"Failed to get face embedding: {e}")
            return None

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

        # Get thumbnails for matched assets
        asset_ids = [vr.id for vr in vector_results]
        thumbnails = await self._get_thumbnails(asset_ids)

        return [
            SearchResult(
                asset_id=vr.id,
                similarity=vr.score,
                thumbnail_url=thumbnails.get(vr.id)
            )
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

    async def search_by_description(
        self,
        query: str,
        user_id: str,
        limit: int = 50
    ) -> tuple[List[SearchResult], float]:
        """
        Search for images by AI-generated description text.

        Uses ILIKE for fuzzy matching against image_descriptions table.

        Returns:
            Tuple of (results, processing_time_ms)
        """
        start_time = time.time()

        logger.info(f"Description search: query='{query}', user_id={user_id}")

        try:
            from supabase import create_client
            from api.config import settings

            supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

            # Search image_descriptions table with case-insensitive match
            result = supabase.table('image_descriptions')\
                .select('asset_id, description')\
                .eq('user_id', user_id)\
                .ilike('description', f'%{query}%')\
                .limit(limit)\
                .execute()

            logger.info(f"Description search found {len(result.data)} matches")

            # Get thumbnails for matched assets
            asset_ids = [r['asset_id'] for r in result.data]
            thumbnails = await self._get_thumbnails(asset_ids)

            results = [
                SearchResult(
                    asset_id=r['asset_id'],
                    similarity=1.0,  # Direct match
                    thumbnail_url=thumbnails.get(r['asset_id']),
                    description=r.get('description')
                )
                for r in result.data
            ]

            elapsed_ms = (time.time() - start_time) * 1000
            return results, elapsed_ms

        except Exception as e:
            logger.error(f"Description search failed: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            return [], elapsed_ms

    async def search_by_text(
        self,
        query: str,
        user_id: str,
        limit: int = 50
    ) -> tuple[List[SearchResult], float]:
        """
        Search for images by OCR-extracted text.

        Uses PostgreSQL full-text search with existing FTS index.

        Returns:
            Tuple of (results, processing_time_ms)
        """
        start_time = time.time()

        logger.info(f"OCR text search: query='{query}', user_id={user_id}")

        try:
            from supabase import create_client
            from api.config import settings

            supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

            # Use textSearch for full-text search on extracted_text column
            # Falls back to ILIKE if FTS index doesn't exist
            result = supabase.table('image_text')\
                .select('asset_id, extracted_text')\
                .eq('user_id', user_id)\
                .ilike('extracted_text', f'%{query}%')\
                .limit(limit)\
                .execute()

            logger.info(f"OCR text search found {len(result.data)} matches")

            # Get thumbnails for matched assets
            asset_ids = [r['asset_id'] for r in result.data]
            thumbnails = await self._get_thumbnails(asset_ids)

            results = [
                SearchResult(
                    asset_id=r['asset_id'],
                    similarity=1.0,  # Direct match
                    thumbnail_url=thumbnails.get(r['asset_id']),
                    extracted_text=r.get('extracted_text')
                )
                for r in result.data
            ]

            elapsed_ms = (time.time() - start_time) * 1000
            return results, elapsed_ms

        except Exception as e:
            logger.error(f"OCR text search failed: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            return [], elapsed_ms

    async def combined_search(
        self,
        user_id: str,
        query: Optional[str] = None,
        filters: Optional['SearchFilters'] = None,
        limit: int = 50,
        threshold: float = 0.2
    ) -> tuple[List[SearchResult], float, dict]:
        """
        Perform a combined search with multiple filters.

        Supports:
        - Semantic query (CLIP embeddings)
        - Date range filter
        - People filter (contact/cluster IDs)
        - Object class filter
        - Description text search
        - OCR text search

        Returns:
            Tuple of (results, processing_time_ms, metadata)
        """
        start_time = time.time()
        from api.schemas.requests import SearchFilters

        logger.info(f"Combined search: query='{query}', filters={filters}")

        if not query and not filters:
            return [], 0, {"error": "Query or filters required"}

        try:
            from supabase import create_client
            from api.config import settings

            supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

            # Start with all asset IDs from various search methods
            candidate_sets = []

            # 1. Semantic search (if query provided)
            if query:
                from api.schemas.requests import SearchRequest
                request = SearchRequest(
                    query=query,
                    limit=limit * 4,  # Over-fetch for filtering
                    threshold=threshold
                )
                semantic_results, _, _ = await self.search(request, user_id)
                if semantic_results:
                    candidate_sets.append({
                        r.asset_id: r.similarity
                        for r in semantic_results
                    })

            # 2. Description search (if description_query in filters)
            if filters and filters.description_query:
                desc_results, _ = await self.search_by_description(
                    filters.description_query, user_id, limit * 2
                )
                if desc_results:
                    candidate_sets.append({
                        r.asset_id: 0.9  # High score for text match
                        for r in desc_results
                    })

            # 3. OCR text search (if text_query in filters)
            if filters and filters.text_query:
                ocr_results, _ = await self.search_by_text(
                    filters.text_query, user_id, limit * 2
                )
                if ocr_results:
                    candidate_sets.append({
                        r.asset_id: 0.9  # High score for text match
                        for r in ocr_results
                    })

            # 4. Object class filter
            if filters and filters.object_classes:
                for obj_class in filters.object_classes:
                    obj_results, _ = await self.search_by_object(
                        obj_class, user_id, min_confidence=0.5, limit=limit * 2
                    )
                    if obj_results:
                        candidate_sets.append({
                            r.asset_id: r.similarity
                            for r in obj_results
                        })

            # 5. People filter (face search)
            if filters and filters.people:
                for person_id in filters.people:
                    face_embedding = await self.get_face_embedding(
                        contact_id=person_id,
                        cluster_id=person_id
                    )
                    if face_embedding:
                        face_results = await self.search_by_face(
                            face_embedding, user_id, limit=limit * 2
                        )
                        if face_results:
                            candidate_sets.append({
                                r.asset_id: r.similarity
                                for r in face_results
                            })

            # Intersect or union candidate sets
            if not candidate_sets:
                return [], (time.time() - start_time) * 1000, {}

            # If multiple sets, find intersection (AND logic)
            if len(candidate_sets) > 1:
                final_candidates = set(candidate_sets[0].keys())
                for cs in candidate_sets[1:]:
                    final_candidates &= set(cs.keys())

                # Average scores for intersected results
                scored_candidates = {}
                for asset_id in final_candidates:
                    scores = [cs.get(asset_id, 0) for cs in candidate_sets]
                    scored_candidates[asset_id] = sum(scores) / len(scores)
            else:
                scored_candidates = candidate_sets[0]

            # Apply date filters if provided
            if filters and (filters.date_start or filters.date_end):
                from datetime import datetime
                asset_ids = list(scored_candidates.keys())
                if asset_ids:
                    assets_result = supabase.table('assets')\
                        .select('id, created_at')\
                        .in_('id', asset_ids)\
                        .execute()

                    for row in assets_result.data:
                        created_at = row.get('created_at')
                        if created_at:
                            try:
                                asset_date = datetime.fromisoformat(
                                    created_at.replace('Z', '+00:00')
                                ).replace(tzinfo=None).date()

                                if filters.date_start and asset_date < filters.date_start:
                                    scored_candidates.pop(row['id'], None)
                                if filters.date_end and asset_date > filters.date_end:
                                    scored_candidates.pop(row['id'], None)
                            except ValueError:
                                pass

            # Sort by score and limit
            sorted_candidates = sorted(
                scored_candidates.items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]

            # Get thumbnails
            asset_ids = [aid for aid, _ in sorted_candidates]
            thumbnails = await self._get_thumbnails(asset_ids)

            results = [
                SearchResult(
                    asset_id=asset_id,
                    similarity=score,
                    thumbnail_url=thumbnails.get(asset_id)
                )
                for asset_id, score in sorted_candidates
            ]

            elapsed_ms = (time.time() - start_time) * 1000
            metadata = {
                "filter_sets": len(candidate_sets),
                "candidates_before_date_filter": len(scored_candidates),
            }

            return results, elapsed_ms, metadata

        except Exception as e:
            logger.error(f"Combined search failed: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            return [], elapsed_ms, {"error": str(e)}
