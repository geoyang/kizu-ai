"""Model registry for dynamic model loading and management."""

from typing import Dict, Type, Optional, Any
from enum import Enum
import logging

from .abstractions import (
    BaseEmbedder,
    BaseDetector,
    BaseFaceModel,
    BaseOCR,
    BaseVLM,
    BaseVectorStore,
    BaseRestorer,
)

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of AI models supported."""
    EMBEDDER = "embedder"
    DETECTOR = "detector"
    FACE = "face"
    OCR = "ocr"
    VLM = "vlm"
    VECTOR_STORE = "vector_store"
    RESTORER = "restorer"


class ModelRegistry:
    """
    Registry for AI models with lazy loading support.
    Allows swapping implementations at runtime.
    """

    _registered: Dict[ModelType, Dict[str, Type]] = {
        ModelType.EMBEDDER: {},
        ModelType.DETECTOR: {},
        ModelType.FACE: {},
        ModelType.OCR: {},
        ModelType.VLM: {},
        ModelType.VECTOR_STORE: {},
        ModelType.RESTORER: {},
    }

    _instances: Dict[str, Any] = {}
    _default_models: Dict[ModelType, str] = {}

    @classmethod
    def register(
        cls,
        model_type: ModelType,
        name: str,
        model_class: Type,
        is_default: bool = False
    ) -> None:
        """Register a model implementation."""
        cls._registered[model_type][name] = model_class
        if is_default:
            cls._default_models[model_type] = name
        logger.info(f"Registered {model_type.value}: {name}")

    @classmethod
    def get(
        cls,
        model_type: ModelType,
        name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Get a model instance (lazy loaded, singleton per name).

        Args:
            model_type: Type of model to get
            name: Specific model name, or None for default
            **kwargs: Arguments to pass to model constructor
        """
        if name is None:
            name = cls._default_models.get(model_type)
            if name is None:
                raise ValueError(f"No default {model_type.value} registered")

        instance_key = f"{model_type.value}:{name}"

        if instance_key not in cls._instances:
            if name not in cls._registered[model_type]:
                available = list(cls._registered[model_type].keys())
                raise ValueError(
                    f"Unknown {model_type.value}: {name}. "
                    f"Available: {available}"
                )
            model_class = cls._registered[model_type][name]
            cls._instances[instance_key] = model_class(**kwargs)
            logger.info(f"Created instance: {instance_key}")

        return cls._instances[instance_key]

    @classmethod
    def list_available(cls, model_type: ModelType) -> list:
        """List available models of a type."""
        return list(cls._registered[model_type].keys())

    @classmethod
    def get_default(cls, model_type: ModelType) -> Optional[str]:
        """Get the default model name for a type."""
        return cls._default_models.get(model_type)

    @classmethod
    def unload(cls, model_type: ModelType, name: str) -> None:
        """Unload a model instance to free memory."""
        instance_key = f"{model_type.value}:{name}"
        if instance_key in cls._instances:
            instance = cls._instances[instance_key]
            if hasattr(instance, 'unload_model'):
                instance.unload_model()
            del cls._instances[instance_key]
            logger.info(f"Unloaded: {instance_key}")

    @classmethod
    def unload_all(cls) -> None:
        """Unload all model instances."""
        for key in list(cls._instances.keys()):
            model_type_str, name = key.split(":", 1)
            model_type = ModelType(model_type_str)
            cls.unload(model_type, name)
