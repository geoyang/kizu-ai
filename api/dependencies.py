"""FastAPI dependency injection."""

from fastapi import Depends, HTTPException, Header
from functools import lru_cache
from typing import Optional

from api.config import settings
from api.stores import SupabaseVectorStore
from api.services import (
    SearchService,
    ProcessService,
    FaceService,
    ClusteringService,
)


# Vector store singleton
@lru_cache()
def get_vector_store() -> SupabaseVectorStore:
    """Get the vector store instance."""
    return SupabaseVectorStore(
        supabase_url=settings.supabase_url,
        supabase_key=settings.supabase_anon_key,
        service_role_key=settings.supabase_service_role_key
    )


# Service dependencies
def get_search_service(
    store: SupabaseVectorStore = Depends(get_vector_store)
) -> SearchService:
    """Get search service instance."""
    return SearchService(store)


def get_process_service(
    store: SupabaseVectorStore = Depends(get_vector_store)
) -> ProcessService:
    """Get process service instance."""
    return ProcessService(store)


def get_face_service(
    store: SupabaseVectorStore = Depends(get_vector_store)
) -> FaceService:
    """Get face service instance."""
    return FaceService(store)


def get_clustering_service(
    store: SupabaseVectorStore = Depends(get_vector_store)
) -> ClusteringService:
    """Get clustering service instance."""
    return ClusteringService(store)


# Auth dependency
async def get_current_user_id(
    authorization: Optional[str] = Header(None)
) -> str:
    """
    Extract user ID from authorization header.

    In production, this validates the Supabase JWT.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format"
        )

    token = authorization[7:]

    # In production, validate JWT and extract user_id
    # For now, using a simple approach
    try:
        from supabase import create_client

        supabase = create_client(
            settings.supabase_url,
            settings.supabase_anon_key
        )

        # Verify token
        user = supabase.auth.get_user(token)
        if user and user.user:
            return user.user.id

        raise HTTPException(status_code=401, detail="Invalid token")

    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


async def get_optional_user_id(
    authorization: Optional[str] = Header(None)
) -> Optional[str]:
    """Optional auth - returns None if not authenticated."""
    if not authorization:
        return None

    try:
        return await get_current_user_id(authorization)
    except HTTPException:
        return None
