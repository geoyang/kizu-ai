"""Abstract base class for Vision-Language Models."""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from PIL import Image


@dataclass
class VLMResponse:
    """Response from a vision-language model."""
    text: str
    model_version: str
    tokens_used: Optional[int] = None
    processing_time_ms: Optional[float] = None


@dataclass
class ImageDescription:
    """Structured image description."""
    summary: str
    objects: List[str]
    scene: Optional[str] = None
    activities: List[str] = None
    mood: Optional[str] = None


class BaseVLM(ABC):
    """Abstract base class for Vision-Language Models (LLaVA, Qwen-VL, etc.)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass

    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """Return maximum context length in tokens."""
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
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> VLMResponse:
        """
        Generate text response for an image with a prompt.

        Args:
            image: PIL Image to analyze
            prompt: Text prompt/question about the image
            max_tokens: Maximum response length
            temperature: Sampling temperature
        """
        pass

    @abstractmethod
    def describe_image(
        self,
        image: Image.Image,
        detail_level: str = "medium"
    ) -> VLMResponse:
        """
        Generate a natural language description of an image.

        Args:
            image: PIL Image to describe
            detail_level: 'brief', 'medium', or 'detailed'
        """
        pass

    @abstractmethod
    def answer_question(
        self,
        image: Image.Image,
        question: str
    ) -> VLMResponse:
        """Answer a specific question about an image."""
        pass

    def parse_query(self, query: str) -> dict:
        """
        Parse a natural language query to extract search parameters.
        Override in implementations for query understanding.
        """
        return {"raw_query": query}
