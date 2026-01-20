#!/usr/bin/env python3
"""
Kizu AI Local Worker Runner

Starts a local worker that pulls image processing jobs from Supabase.
Run this on any machine with GPU/CPU to process images locally.

Usage:
    python run_worker.py
    python run_worker.py --worker-id my-nas-worker
    python run_worker.py --poll-interval 10 --debug
"""

import asyncio
import argparse
import logging
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.workers.local_worker import LocalWorker
from api.core import ModelRegistry
from api.config import settings


def setup_logging(debug: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def register_models() -> None:
    """Register AI models (same as main.py)."""
    from api.models import (
        CLIPEmbedder,
        YOLODetector,
        InsightFaceModel,
        EasyOCRModel,
        Florence2Model,
    )
    from api.core.registry import ModelType

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

    ModelRegistry.register(
        ModelType.DETECTOR,
        "yolo",
        lambda: YOLODetector(
            model_size=settings.yolo_size,
            cache_dir=settings.model_cache_dir
        ),
        is_default=True
    )

    ModelRegistry.register(
        ModelType.FACE,
        "insightface",
        lambda: InsightFaceModel(
            model_name=settings.face_model_name,
            cache_dir=settings.model_cache_dir
        ),
        is_default=True
    )

    ModelRegistry.register(
        ModelType.OCR,
        "easyocr",
        lambda: EasyOCRModel(),
        is_default=True
    )

    ModelRegistry.register(
        ModelType.VLM,
        "florence2",
        lambda: Florence2Model(
            model_name=settings.florence_model,
            cache_dir=settings.model_cache_dir,
            quantization=settings.vlm_quantization
        ),
        is_default=True
    )


async def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    logger = logging.getLogger(__name__)

    # Validate required settings
    if not settings.supabase_url or not settings.supabase_service_role_key:
        logger.error(
            "Missing required environment variables: "
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY"
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Kizu AI Local Worker")
    logger.info("=" * 60)
    logger.info(f"Supabase URL: {settings.supabase_url}")
    logger.info(f"Poll interval: {args.poll_interval}s")
    logger.info(f"Worker ID: {args.worker_id or 'auto-generated'}")
    logger.info("=" * 60)

    # Register AI models
    logger.info("Registering AI models...")
    register_models()

    # Create and start worker
    worker = LocalWorker(
        worker_id=args.worker_id,
        poll_interval=args.poll_interval,
        max_attempts=args.max_attempts
    )

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await worker.stop()
        ModelRegistry.unload_all()
        logger.info("Worker shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kizu AI Local Worker - Pull-based image processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_worker.py
    python run_worker.py --worker-id my-nas
    python run_worker.py --poll-interval 10 --debug

Environment variables (set in .env):
    SUPABASE_URL              - Your Supabase project URL
    SUPABASE_ANON_KEY         - Supabase anon/public key
    SUPABASE_SERVICE_ROLE_KEY - Supabase service role key (required)
        """
    )

    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Unique identifier for this worker (default: auto-generated)"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between polling for jobs (default: 5.0)"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Max retry attempts per job (default: 3)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    setup_logging(debug=args.debug)
    asyncio.run(main(args))
