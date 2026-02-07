"""Abstract base classes for AI model components."""

from .base_embedder import BaseEmbedder, EmbeddingResult
from .base_detector import BaseDetector, DetectionResult, DetectedObject, BoundingBox
from .base_face_model import BaseFaceModel, FaceDetectionResult, DetectedFace, FaceMatch
from .base_ocr import BaseOCR, OCRResult, TextRegion
from .base_vlm import BaseVLM, VLMResponse, ImageDescription
from .base_vector_store import BaseVectorStore, VectorSearchResult, StoredVector
from .base_restorer import BaseRestorer, RestoreResult

__all__ = [
    # Base classes
    "BaseEmbedder",
    "BaseDetector",
    "BaseFaceModel",
    "BaseOCR",
    "BaseVLM",
    "BaseVectorStore",
    "BaseRestorer",
    # Data classes
    "EmbeddingResult",
    "DetectionResult",
    "DetectedObject",
    "BoundingBox",
    "FaceDetectionResult",
    "DetectedFace",
    "FaceMatch",
    "OCRResult",
    "TextRegion",
    "VLMResponse",
    "ImageDescription",
    "VectorSearchResult",
    "StoredVector",
    "RestoreResult",
]
