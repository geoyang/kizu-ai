#!/usr/bin/env python3
"""
Kizu AI Video Transcoding Worker Runner

Starts a worker that pulls video transcoding jobs from Supabase and processes
them using FFmpeg. Requires FFmpeg to be installed on the system.

Usage:
    python run_video_worker.py
    python run_video_worker.py --worker-id my-transcoder
    python run_video_worker.py --poll-interval 30 --debug
"""

import asyncio
import argparse
import logging
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.workers.video_worker import VideoWorker
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
    logger.info("Kizu AI Video Transcoding Worker")
    logger.info("=" * 60)
    logger.info(f"Supabase URL: {settings.supabase_url}")
    logger.info(f"Poll interval: {args.poll_interval}s")
    logger.info(f"Worker ID: {args.worker_id or 'auto-generated'}")
    logger.info("=" * 60)

    # Create and start worker
    worker = VideoWorker(
        worker_id=args.worker_id,
        poll_interval=args.poll_interval,
    )

    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await worker.stop()
        logger.info("Video worker shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kizu AI Video Transcoding Worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_video_worker.py
    python run_video_worker.py --worker-id my-transcoder
    python run_video_worker.py --poll-interval 30 --debug

Requirements:
    - FFmpeg installed and available in PATH

Environment variables (set in .env):
    SUPABASE_URL              - Your Supabase project URL
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
        default=10.0,
        help="Seconds between polling for jobs (default: 10.0)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    setup_logging(debug=args.debug)
    asyncio.run(main(args))
