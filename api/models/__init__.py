"""Concrete AI model implementations."""

from .embedders import CLIPEmbedder
from .detectors import YOLODetector
from .faces import InsightFaceModel
from .ocr import EasyOCRModel
from .vlm import LLaVAModel

__all__ = [
    "CLIPEmbedder",
    "YOLODetector",
    "InsightFaceModel",
    "EasyOCRModel",
    "LLaVAModel",
]
