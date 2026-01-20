"""Response schemas for the API."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from enum import Enum


class JobStatus(str, Enum):
    """Processing job statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class MatchedFace(BaseModel):
    """A matched face in search results."""
    contact_id: Optional[str] = None
    cluster_id: Optional[str] = None
    name: Optional[str] = None
    confidence: float


class SearchResult(BaseModel):
    """A single search result."""
    asset_id: str
    similarity: float  # Renamed from score for frontend compatibility
    description: Optional[str] = None
    matched_faces: Optional[List[MatchedFace]] = None
    matched_objects: Optional[List[str]] = None
    matched_location: Optional[str] = None  # Location name from asset metadata
    thumbnail_url: Optional[str] = None


class SearchResponse(BaseModel):
    """Response from a search query."""
    status: str  # 'completed' or 'processing'
    job_id: Optional[str] = None
    results: Optional[List[SearchResult]] = None
    total: int = 0
    processing_time_ms: Optional[float] = None
    query_parsed: Optional[Dict[str, Any]] = None


class ProcessingJobResponse(BaseModel):
    """Response for a processing job."""
    job_id: str
    status: JobStatus
    progress: int = 0  # 0-100
    processed: int = 0
    total: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class BoundingBox(BaseModel):
    """Bounding box for a detected face."""
    x: float
    y: float
    width: float
    height: float


class SampleFace(BaseModel):
    """A sample face from a cluster."""
    asset_id: str
    face_index: int
    thumbnail_url: Optional[str] = None
    is_from_video: bool = False
    bounding_box: Optional[Dict[str, Any]] = None


class FaceClusterResponse(BaseModel):
    """Response for a face cluster."""
    cluster_id: str
    name: Optional[str] = None
    knox_contact_id: Optional[str] = None
    face_count: int
    sample_faces: List[SampleFace]


class FaceClustersListResponse(BaseModel):
    """Response listing all face clusters."""
    clusters: List[FaceClusterResponse]
    total_faces: int
    unlabeled_clusters: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    device: str
    models_loaded: List[str]
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Information about loaded models."""
    embedder: Optional[str] = None
    detector: Optional[str] = None
    face_model: Optional[str] = None
    ocr_model: Optional[str] = None
    vlm_model: Optional[str] = None
    device: str
    memory_used_mb: Optional[float] = None
