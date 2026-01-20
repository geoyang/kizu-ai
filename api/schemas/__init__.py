"""Pydantic schemas for API requests and responses."""

from .requests import (
    SearchRequest,
    ProcessImageRequest,
    BatchProcessRequest,
    ClusterFacesRequest,
    AssignClusterRequest,
    FaceSearchRequest,
)
from .responses import (
    SearchResponse,
    SearchResult,
    ProcessingJobResponse,
    FaceClusterResponse,
    HealthResponse,
)

__all__ = [
    # Requests
    "SearchRequest",
    "ProcessImageRequest",
    "BatchProcessRequest",
    "ClusterFacesRequest",
    "AssignClusterRequest",
    "FaceSearchRequest",
    # Responses
    "SearchResponse",
    "SearchResult",
    "ProcessingJobResponse",
    "FaceClusterResponse",
    "HealthResponse",
]
