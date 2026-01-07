"""Abstract base class for vector storage backends."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class VectorSearchResult:
    """Result from a vector similarity search."""
    id: str
    score: float
    metadata: Optional[dict] = None


@dataclass
class StoredVector:
    """A vector stored in the database."""
    id: str
    embedding: np.ndarray
    metadata: dict


class BaseVectorStore(ABC):
    """Abstract base class for vector storage (Supabase/pgvector, Qdrant, etc.)."""

    @property
    @abstractmethod
    def store_name(self) -> str:
        """Return the store identifier."""
        pass

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the vector store."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the vector store."""
        pass

    @abstractmethod
    async def store_embedding(
        self,
        collection: str,
        id: str,
        embedding: np.ndarray,
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Store a single embedding.

        Args:
            collection: Collection/table name
            id: Unique identifier for the vector
            embedding: The embedding vector
            metadata: Optional metadata to store with the vector
        """
        pass

    @abstractmethod
    async def store_embeddings_batch(
        self,
        collection: str,
        items: List[Tuple[str, np.ndarray, Optional[dict]]]
    ) -> int:
        """
        Store multiple embeddings.

        Args:
            collection: Collection/table name
            items: List of (id, embedding, metadata) tuples

        Returns:
            Number of successfully stored items
        """
        pass

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_embedding: np.ndarray,
        limit: int = 10,
        threshold: Optional[float] = None,
        filters: Optional[dict] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors.

        Args:
            collection: Collection/table name
            query_embedding: Query vector
            limit: Maximum results to return
            threshold: Minimum similarity threshold
            filters: Optional metadata filters
        """
        pass

    @abstractmethod
    async def get_embedding(
        self,
        collection: str,
        id: str
    ) -> Optional[StoredVector]:
        """Retrieve a stored embedding by ID."""
        pass

    @abstractmethod
    async def delete_embedding(
        self,
        collection: str,
        id: str
    ) -> bool:
        """Delete an embedding by ID."""
        pass

    @abstractmethod
    async def count(
        self,
        collection: str,
        filters: Optional[dict] = None
    ) -> int:
        """Count vectors in a collection."""
        pass
