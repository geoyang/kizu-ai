"""Application configuration."""

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # App info
    app_name: str = "Kizu AI"
    version: str = "0.1.0"
    debug: bool = False

    # Supabase
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: Optional[str] = None

    # Model settings
    model_cache_dir: str = "./model_cache"
    default_embedder: str = "clip"
    default_detector: str = "yolo"
    default_face_model: str = "insightface"
    default_ocr: str = "easyocr"
    default_vlm: Optional[str] = "florence2"  # Options: blip2, florence2, llava
    florence_model: str = "microsoft/Florence-2-base"  # or "microsoft/Florence-2-large"
    vlm_quantization: Optional[str] = "int8"  # Options: None, "int8", "int4"

    # CLIP settings
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"

    # YOLO settings
    yolo_size: str = "m"  # n, s, m, l, x

    # Face settings
    face_model_name: str = "buffalo_l"
    face_threshold: float = 0.6

    # Restoration settings
    restore_upscale: int = 2
    restore_model: str = "realesrgan"
    restore_max_input_size: int = 512  # Max px before downscaling (prevents OOM on CPU)

    # Processing settings
    max_image_size: int = 2048
    batch_size: int = 8

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Redis (for Celery)
    redis_url: str = "redis://localhost:6379/0"

    # External API flags (all disabled by default)
    allow_external_apis: bool = False
    openai_api_key: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
