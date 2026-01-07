"""Search API router."""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

from api.schemas import SearchRequest, SearchResponse, SearchResult
from api.services import SearchService
from api.dependencies import get_search_service, get_current_user_id

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search_images(
    request: SearchRequest,
    search_service: SearchService = Depends(get_search_service),
    user_id: str = Depends(get_current_user_id)
):
    """
    Search images using natural language.

    Examples:
    - "photos of John at the beach"
    - "sunset photos from 2023"
    - "images with dogs"
    - "photos with text or signs"
    """
    try:
        results, elapsed_ms = await search_service.search(request, user_id)

        return SearchResponse(
            status="completed",
            results=results,
            total=len(results),
            processing_time_ms=elapsed_ms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/by-face", response_model=SearchResponse)
async def search_by_face(
    contact_id: Optional[str] = None,
    cluster_id: Optional[str] = None,
    threshold: float = 0.6,
    limit: int = 50,
    search_service: SearchService = Depends(get_search_service),
    user_id: str = Depends(get_current_user_id)
):
    """
    Search images containing a specific person.

    Provide either contact_id (Knox contact) or cluster_id (face cluster).
    """
    if not contact_id and not cluster_id:
        raise HTTPException(
            status_code=400,
            detail="Either contact_id or cluster_id required"
        )

    # Implementation would get face embedding and search
    return SearchResponse(
        status="completed",
        results=[],
        total=0
    )


@router.post("/by-object", response_model=SearchResponse)
async def search_by_object(
    object_class: str,
    min_confidence: float = 0.5,
    limit: int = 50,
    search_service: SearchService = Depends(get_search_service),
    user_id: str = Depends(get_current_user_id)
):
    """
    Search images containing a specific detected object.

    Uses YOLO detection results for exact object matching.
    Example: object_class="wine glass" finds images where a wine glass was detected.
    """
    try:
        results, elapsed_ms = await search_service.search_by_object(
            object_class=object_class,
            user_id=user_id,
            min_confidence=min_confidence,
            limit=limit
        )

        return SearchResponse(
            status="completed",
            results=results,
            total=len(results),
            processing_time_ms=elapsed_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/objects")
async def get_detected_objects(
    user_id: str = Depends(get_current_user_id)
):
    """Get list of all detected object classes for this user."""
    from supabase import create_client
    from api.config import settings

    try:
        supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)
        result = supabase.table('detected_objects').select('object_class').eq('user_id', user_id).execute()

        # Get unique classes with counts
        class_counts = {}
        for row in result.data:
            cls = row['object_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        # Sort by count descending
        sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])

        return {
            "objects": [{"class": cls, "count": cnt} for cls, cnt in sorted_classes]
        }
    except Exception as e:
        return {"objects": [], "error": str(e)}


@router.get("/suggestions")
async def get_search_suggestions(
    query: str,
    limit: int = 10,
    user_id: str = Depends(get_current_user_id)
):
    """Get search suggestions based on partial query."""
    # Would return common objects, people names, locations
    return {
        "suggestions": [],
        "query": query
    }
