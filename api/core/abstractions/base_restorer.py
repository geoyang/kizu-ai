"""Abstract base class for image restoration models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL import Image


@dataclass
class RestoreResult:
    """Result from a restoration operation."""
    restored_image: Image.Image
    model_version: str
    processing_time_ms: int


class BaseRestorer(ABC):
    """Abstract base class for image restoration models (Real-ESRGAN, GFP-GAN, etc.)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
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
    def restore(self, image: Image.Image) -> RestoreResult:
        """Restore/enhance a single image."""
        pass
