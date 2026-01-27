"""Request schemas for the API."""

from typing import List, Optional
from datetime import date
from pydantic import BaseModel, Field
from enum import Enum


class ProcessingOperation(str, Enum):
    """Available processing operations."""
    EMBEDDING = "embedding"
    FACES = "faces"
    OBJECTS = "objects"
    OCR = "ocr"
    DESCRIBE = "describe"
    ALL = "all"


class SearchFilters(BaseModel):
    """Filters for search queries."""
    date_start: Optional[date] = None
    date_end: Optional[date] = None
    locations: Optional[List[str]] = None
    people: Optional[List[str]] = None  # Contact IDs or cluster IDs
    media_type: Optional[str] = None  # 'photo' or 'video'
    has_faces: Optional[bool] = None
    has_text: Optional[bool] = None
    # New filters for enhanced search
    description_query: Optional[str] = None  # Search AI-generated descriptions
    text_query: Optional[str] = None  # Search OCR-extracted text
    object_classes: Optional[List[str]] = None  # Filter by detected objects


class SearchRequest(BaseModel):
    """Natural language search request."""
    query: str = Field(..., min_length=1, max_length=500)
    filters: Optional[SearchFilters] = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)
    threshold: float = Field(default=0.2, ge=0.0, le=1.0)  # CLIP text-image similarity is typically 0.15-0.35
    include_descriptions: bool = Field(default=True)


class ProcessImageRequest(BaseModel):
    """Request to process a single image."""
    asset_id: str = Field(default="test")
    operations: List[ProcessingOperation] = Field(
        default=[ProcessingOperation.ALL]
    )
    image_url: Optional[str] = None  # URL to fetch image from
    image_base64: Optional[str] = None  # Base64-encoded image data
    store_results: bool = Field(default=False)  # Store to DB (requires valid asset_id)
    force_reprocess: bool = Field(default=False, description="Clear existing AI data before reprocessing")


class BatchProcessRequest(BaseModel):
    """Request to process multiple images."""
    asset_ids: Optional[List[str]] = None  # If None, process all unprocessed
    operations: List[ProcessingOperation] = Field(
        default=[ProcessingOperation.ALL]
    )
    limit: Optional[int] = Field(default=None, ge=1, le=1000)
    skip_processed: bool = Field(default=False, description="Skip assets that have already been AI processed")
    force_reprocess: bool = Field(default=False, description="Clear existing AI data before reprocessing")


class ClusterFacesRequest(BaseModel):
    """Request to cluster faces."""
    threshold: float = Field(default=0.6, ge=0.3, le=0.9)
    min_cluster_size: int = Field(default=2, ge=1)
    recompute: bool = Field(default=False)


class AssignClusterRequest(BaseModel):
    """Request to assign a cluster to a contact."""
    knox_contact_id: str
    name: Optional[str] = None
    exclude_face_ids: Optional[List[str]] = None  # Face IDs to exclude from assignment


class MergeClustersRequest(BaseModel):
    """Request to merge multiple clusters into one."""
    cluster_ids: List[str] = Field(..., min_items=2)


class FaceSearchRequest(BaseModel):
    """Request to find images with a specific face."""
    contact_id: Optional[str] = None
    cluster_id: Optional[str] = None
    face_embedding: Optional[List[float]] = None  # Direct embedding search
    threshold: float = Field(default=0.6, ge=0.3, le=0.9)
    limit: int = Field(default=50, ge=1, le=200)


class DescriptionSearchRequest(BaseModel):
    """Request to search in AI-generated descriptions."""
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=50, ge=1, le=200)


class TextSearchRequest(BaseModel):
    """Request to search OCR-extracted text."""
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=50, ge=1, le=200)


class CombinedSearchRequest(BaseModel):
    """Combined multi-filter search request."""
    query: Optional[str] = None  # Semantic query (optional if using filters)
    filters: Optional[SearchFilters] = None
    limit: int = Field(default=50, ge=1, le=200)
    threshold: float = Field(default=0.2, ge=0.0, le=1.0)


class TagSyncPreviewRequest(BaseModel):
    """Request to preview manual tag to AI detection matching."""
    asset_ids: Optional[List[str]] = None  # Specific assets, or all if None
    iou_threshold: float = Field(default=0.3, ge=0.1, le=0.9)
    limit: int = Field(default=50, ge=1, le=200)


class ApplyTagSyncRequest(BaseModel):
    """Request to apply tag sync matches to clustering."""
    matches: List[dict]  # List of {manual_tag_id, ai_face_id} pairs to apply
    action: str = Field(default="link")  # 'link' or 'merge'
