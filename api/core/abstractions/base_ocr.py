"""Abstract base class for OCR models."""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from PIL import Image

from .base_detector import BoundingBox


@dataclass
class TextRegion:
    """A region of detected text."""
    text: str
    confidence: float
    bounding_box: BoundingBox
    language: Optional[str] = None


@dataclass
class OCRResult:
    """Result from OCR processing."""
    full_text: str
    regions: List[TextRegion]
    model_version: str
    languages_detected: List[str]


class BaseOCR(ABC):
    """Abstract base class for OCR models (EasyOCR, Tesseract, etc.)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass

    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        pass

    @abstractmethod
    def load_model(self, languages: Optional[List[str]] = None) -> None:
        """
        Load the model into memory.

        Args:
            languages: Optional list of language codes to load
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model from memory."""
        pass

    @abstractmethod
    def extract_text(
        self,
        image: Image.Image,
        languages: Optional[List[str]] = None
    ) -> OCRResult:
        """
        Extract text from an image.

        Args:
            image: PIL Image to process
            languages: Optional language hints
        """
        pass

    @abstractmethod
    def extract_text_batch(
        self,
        images: List[Image.Image],
        languages: Optional[List[str]] = None
    ) -> List[OCRResult]:
        """Extract text from multiple images."""
        pass

    def has_text(self, image: Image.Image, min_confidence: float = 0.5) -> bool:
        """Quick check if image contains readable text."""
        result = self.extract_text(image)
        return any(r.confidence >= min_confidence for r in result.regions)
