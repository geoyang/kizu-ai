"""Health and system status API router."""

import time
from fastapi import APIRouter

from api.schemas.responses import HealthResponse, ModelInfoResponse
from api.core import ModelRegistry
from api.core.registry import ModelType
from api.utils.device_utils import get_device, get_device_info
from api.config import settings

router = APIRouter(tags=["system"])

_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    models_loaded = []

    for model_type in ModelType:
        try:
            instances = ModelRegistry._instances
            for key in instances:
                if key.startswith(model_type.value):
                    models_loaded.append(key)
        except Exception:
            pass

    return HealthResponse(
        status="healthy",
        version=settings.version,
        device=get_device(),
        models_loaded=models_loaded,
        uptime_seconds=time.time() - _start_time
    )


@router.get("/models", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about available and loaded models."""
    info = ModelInfoResponse(device=get_device())

    # Check what's registered as default
    info.embedder = ModelRegistry.get_default(ModelType.EMBEDDER)
    info.detector = ModelRegistry.get_default(ModelType.DETECTOR)
    info.face_model = ModelRegistry.get_default(ModelType.FACE)
    info.ocr_model = ModelRegistry.get_default(ModelType.OCR)
    info.vlm_model = ModelRegistry.get_default(ModelType.VLM)

    return info


@router.get("/device")
async def get_device_details():
    """Get detailed device information."""
    return get_device_info()


@router.get("/models/available")
async def list_available_models():
    """List all available (registered) models."""
    available = {}

    for model_type in ModelType:
        available[model_type.value] = {
            "options": ModelRegistry.list_available(model_type),
            "default": ModelRegistry.get_default(model_type)
        }

    return available


@router.post("/models/load/{model_type}/{model_name}")
async def load_model(model_type: str, model_name: str):
    """Manually load a specific model."""
    try:
        mt = ModelType(model_type)
        model = ModelRegistry.get(mt, model_name)

        if hasattr(model, 'load_model'):
            model.load_model()

        return {
            "status": "loaded",
            "model_type": model_type,
            "model_name": model_name
        }

    except ValueError as e:
        return {"status": "error", "error": str(e)}


@router.post("/models/unload/{model_type}/{model_name}")
async def unload_model(model_type: str, model_name: str):
    """Manually unload a model to free memory."""
    try:
        mt = ModelType(model_type)
        ModelRegistry.unload(mt, model_name)

        return {
            "status": "unloaded",
            "model_type": model_type,
            "model_name": model_name
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.post("/models/unload-all")
async def unload_all_models():
    """Unload all models to free memory."""
    ModelRegistry.unload_all()
    return {"status": "all models unloaded"}
