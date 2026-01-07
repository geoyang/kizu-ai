"""Abstract base class for face detection and recognition models."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image

from .base_detector import BoundingBox


@dataclass
class DetectedFace:
    """A detected face in an image."""
    bounding_box: BoundingBox
    confidence: float
    landmarks: Optional[dict] = None  # eye, nose, mouth positions
    embedding: Optional[np.ndarray] = None
    face_index: int = 0


@dataclass
class FaceDetectionResult:
    """Result from face detection."""
    faces: List[DetectedFace]
    model_version: str
    image_width: int
    image_height: int


@dataclass
class FaceMatch:
    """Result of comparing two faces."""
    similarity: float
    is_match: bool
    threshold_used: float


class BaseFaceModel(ABC):
    """Abstract base class for face detection/recognition (InsightFace, etc.)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the face embedding vector dimension."""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model from memory."""
        pass

    @abstractmethod
    def detect_faces(
        self,
        image: Image.Image,
        min_face_size: int = 20
    ) -> FaceDetectionResult:
        """Detect faces in an image without generating embeddings."""
        pass

    @abstractmethod
    def detect_and_embed(
        self,
        image: Image.Image,
        min_face_size: int = 20
    ) -> FaceDetectionResult:
        """Detect faces and generate embeddings for each."""
        pass

    @abstractmethod
    def get_embedding(
        self,
        face_image: Image.Image
    ) -> np.ndarray:
        """Generate embedding for a cropped face image."""
        pass

    def compare_faces(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        threshold: float = 0.6
    ) -> FaceMatch:
        """Compare two face embeddings."""
        similarity = self._compute_similarity(embedding1, embedding2)
        return FaceMatch(
            similarity=similarity,
            is_match=similarity >= threshold,
            threshold_used=threshold
        )

    def _compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
