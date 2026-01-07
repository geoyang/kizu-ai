"""Abstract base class for object detection models."""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from PIL import Image


@dataclass
class BoundingBox:
    """Bounding box for detected object."""
    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class DetectedObject:
    """A detected object in an image."""
    class_name: str
    confidence: float
    bounding_box: BoundingBox
    class_id: Optional[int] = None


@dataclass
class DetectionResult:
    """Result from object detection."""
    objects: List[DetectedObject]
    model_version: str
    image_width: int
    image_height: int


class BaseDetector(ABC):
    """Abstract base class for object detection models (YOLO, etc.)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass

    @property
    @abstractmethod
    def supported_classes(self) -> List[str]:
        """Return list of detectable object classes."""
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
    def detect(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.5,
        classes: Optional[List[str]] = None
    ) -> DetectionResult:
        """
        Detect objects in an image.

        Args:
            image: PIL Image to process
            confidence_threshold: Minimum confidence for detections
            classes: Optional filter for specific classes
        """
        pass

    @abstractmethod
    def detect_batch(
        self,
        images: List[Image.Image],
        confidence_threshold: float = 0.5,
        classes: Optional[List[str]] = None
    ) -> List[DetectionResult]:
        """Detect objects in multiple images."""
        pass
