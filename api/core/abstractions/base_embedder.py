"""Abstract base class for image/text embedding models."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class EmbeddingResult:
    """Result from an embedding operation."""
    embedding: np.ndarray
    model_version: str
    dimension: int


class BaseEmbedder(ABC):
    """Abstract base class for embedding models (CLIP, SigLIP, etc.)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the embedding vector dimension."""
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
    def embed_image(self, image: Image.Image) -> EmbeddingResult:
        """Generate embedding for a single image."""
        pass

    @abstractmethod
    def embed_images(self, images: List[Image.Image]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple images."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for text query."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple text queries."""
        pass

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
