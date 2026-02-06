"""Kizu AI - Personal AI Image Intelligence Engine."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.core import ModelRegistry
from api.core.registry import ModelType
from api.routers import (
    search_router,
    process_router,
    faces_router,
    jobs_router,
    health_router,
    video_router,
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def register_models():
    """Register all available model implementations."""
    from api.models import (
        CLIPEmbedder,
        YOLODetector,
        InsightFaceModel,
        EasyOCRModel,
        BLIP2Model,
        Florence2Model,
    )

    # Register embedders
    ModelRegistry.register(
        ModelType.EMBEDDER,
        "clip",
        lambda: CLIPEmbedder(
            model_name=settings.clip_model,
            pretrained=settings.clip_pretrained,
            cache_dir=settings.model_cache_dir
        ),
        is_default=True
    )

    # Register detectors
    ModelRegistry.register(
        ModelType.DETECTOR,
        "yolo",
        lambda: YOLODetector(
            model_size=settings.yolo_size,
            cache_dir=settings.model_cache_dir
        ),
        is_default=True
    )

    # Register face models
    ModelRegistry.register(
        ModelType.FACE,
        "insightface",
        lambda: InsightFaceModel(
            model_name=settings.face_model_name,
            cache_dir=settings.model_cache_dir
        ),
        is_default=True
    )

    # Register OCR
    ModelRegistry.register(
        ModelType.OCR,
        "easyocr",
        lambda: EasyOCRModel(),
        is_default=True
    )

    # Register VLM (BLIP-2 - lightweight alternative to LLaVA)
    ModelRegistry.register(
        ModelType.VLM,
        "blip2",
        lambda: BLIP2Model(cache_dir=settings.model_cache_dir),
        is_default=settings.default_vlm == "blip2"
    )

    # Register Florence-2 VLM (Microsoft's efficient multimodal model)
    ModelRegistry.register(
        ModelType.VLM,
        "florence2",
        lambda: Florence2Model(
            model_name=settings.florence_model,
            cache_dir=settings.model_cache_dir,
            quantization=settings.vlm_quantization
        ),
        is_default=settings.default_vlm == "florence2"
    )

    logger.info("Models registered")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    register_models()

    yield

    # Shutdown
    logger.info("Shutting down...")
    ModelRegistry.unload_all()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Personal AI Image Intelligence Engine",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(search_router, prefix="/api/v1")
app.include_router(process_router, prefix="/api/v1")
app.include_router(faces_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")
app.include_router(video_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
