"""Pydantic schemas for API requests and responses."""

from .requests import (
    SearchRequest,
    ProcessImageRequest,
    BatchProcessRequest,
    ClusterFacesRequest,
    AssignClusterRequest,
    MergeClustersRequest,
    FaceSearchRequest,
    TagSyncPreviewRequest,
    ApplyTagSyncRequest,
)
from .responses import (
    SearchResponse,
    SearchResult,
    ProcessingJobResponse,
    FaceClusterResponse,
    HealthResponse,
    TagSyncPreviewResponse,
    TagSyncSummary,
    AssetTagPreview,
    ManualTag,
    AIDetection,
    TagMatch,
    BoundingBox,
)

__all__ = [
    # Requests
    "SearchRequest",
    "ProcessImageRequest",
    "BatchProcessRequest",
    "ClusterFacesRequest",
    "AssignClusterRequest",
    "MergeClustersRequest",
    "FaceSearchRequest",
    "TagSyncPreviewRequest",
    "ApplyTagSyncRequest",
    # Responses
    "SearchResponse",
    "SearchResult",
    "ProcessingJobResponse",
    "FaceClusterResponse",
    "HealthResponse",
    "TagSyncPreviewResponse",
    "TagSyncSummary",
    "AssetTagPreview",
    "ManualTag",
    "AIDetection",
    "TagMatch",
    "BoundingBox",
]
