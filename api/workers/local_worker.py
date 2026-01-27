"""Local worker that pulls jobs from Supabase ai_processing_jobs table."""

import asyncio
import logging
import signal
import uuid
from datetime import datetime
from typing import Optional

from supabase import create_client, Client

from api.config import settings
from api.services.process_service import ProcessService
from api.stores import SupabaseVectorStore
from api.schemas.requests import ProcessingOperation
from api.utils.image_utils import load_image

logger = logging.getLogger(__name__)


class LocalWorker:
    """
    Pull-based worker that polls ai_processing_jobs table
    and processes images locally.
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        poll_interval: float = 5.0,
        max_attempts: int = 3
    ):
        self._worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self._poll_interval = poll_interval
        self._max_attempts = max_attempts
        self._running = False
        self._client: Optional[Client] = None
        self._process_service: Optional[ProcessService] = None
        self._current_job_id: Optional[str] = None

    @property
    def worker_id(self) -> str:
        return self._worker_id

    async def start(self) -> None:
        """Start the worker."""
        logger.info(f"Starting local worker: {self._worker_id}")

        # Initialize Supabase client
        self._client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key
        )

        # Initialize process service
        vector_store = SupabaseVectorStore(
            supabase_url=settings.supabase_url,
            supabase_key=settings.supabase_anon_key,
            service_role_key=settings.supabase_service_role_key
        )
        self._process_service = ProcessService(vector_store)

        self._running = True

        # Set up signal handlers for graceful shutdown
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle_shutdown)

        # Start the main loop
        await self._run_loop()

    async def _run_loop(self) -> None:
        """Main worker loop with realtime subscription and polling fallback."""
        logger.info(f"Worker {self._worker_id} entering main loop")

        # Start realtime subscription in background
        realtime_task = asyncio.create_task(self._subscribe_realtime())

        # Also poll periodically as fallback
        try:
            while self._running:
                # Try to claim and process a job
                await self._poll_and_process()

                # Wait before next poll
                await asyncio.sleep(self._poll_interval)

        except asyncio.CancelledError:
            logger.info("Worker loop cancelled")
        finally:
            realtime_task.cancel()
            try:
                await realtime_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Worker {self._worker_id} stopped")

    async def _subscribe_realtime(self) -> None:
        """Subscribe to Supabase Realtime for instant job notifications."""
        try:
            channel = self._client.channel('ai_processing_jobs_changes')

            def on_insert(payload):
                """Handle new job inserted."""
                logger.info(f"Realtime: New job detected: {payload}")
                asyncio.create_task(self._poll_and_process())

            channel.on_postgres_changes(
                event='INSERT',
                schema='public',
                table='ai_processing_jobs',
                callback=on_insert
            )

            await channel.subscribe()
            logger.info("Subscribed to Supabase Realtime for ai_processing_jobs")

            while self._running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.warning(f"Realtime subscription error: {e}")
            logger.info("Falling back to polling only")

    async def _poll_and_process(self) -> None:
        """Poll for a job and process it."""
        if self._current_job_id:
            return

        try:
            # Find oldest pending job
            result = self._client.table('ai_processing_jobs').select(
                '*'
            ).eq(
                'status', 'pending'
            ).order(
                'created_at', desc=False
            ).limit(1).execute()

            if not result.data or len(result.data) == 0:
                return

            job = result.data[0]

            # Claim the job by updating status to processing
            from api.config import settings
            update_result = self._client.table('ai_processing_jobs').update({
                'status': 'processing',
                'worker_id': self._worker_id,
                'picked_up_at': datetime.utcnow().isoformat(),
                'ai_version': settings.version,
            }).eq('id', job['id']).eq('status', 'pending').execute()

            if not update_result.data or len(update_result.data) == 0:
                # Job was claimed by another worker
                logger.debug(f"Job {job['id']} already claimed by another worker")
                return

            self._current_job_id = job['id']
            input_params = job.get('input_params') or {}

            logger.info(
                f"Claimed job {job['id']} for asset {input_params.get('asset_id')}"
            )

            await self._process_job(job)

        except Exception as e:
            logger.error(f"Error in poll_and_process: {e}")
        finally:
            self._current_job_id = None

    async def _process_job(self, job: dict) -> None:
        """Process a single job."""
        job_id = job['id']
        user_id = job['user_id']
        input_params = job.get('input_params') or {}
        asset_id = input_params.get('asset_id')
        image_url = input_params.get('image_url')
        operations = input_params.get('operations', ['embedding', 'faces', 'objects'])

        try:
            # Load image from URL or storage
            image = None
            if image_url:
                image = await self._load_image_from_url(image_url)
            if image is None and asset_id:
                image = await self._load_asset_image(asset_id, user_id)

            if image is None:
                raise ValueError(f"Could not load image for job {job_id}")

            # Convert operation strings to enums
            ops = [
                ProcessingOperation(op)
                for op in operations
                if op in [e.value for e in ProcessingOperation]
            ]

            if not ops:
                ops = [
                    ProcessingOperation.EMBEDDING,
                    ProcessingOperation.OBJECTS,
                    ProcessingOperation.FACES
                ]

            # Process the image
            result = await self._process_service.process_image(
                asset_id=asset_id or job_id,
                image=image,
                operations=ops,
                user_id=user_id,
                store_results=True,
                worker_id=self._worker_id
            )

            # Add version info to result
            from api.config import settings
            result['_meta'] = {
                'worker_id': self._worker_id,
                'ai_version': settings.version,
                'models': {
                    'clip': settings.clip_model,
                    'yolo': f"v8-{settings.yolo_size}",
                    'face': settings.face_model_name,
                    'vlm': settings.default_vlm,
                }
            }

            # Mark job as completed
            self._client.table('ai_processing_jobs').update({
                'status': 'completed',
                'progress': 100,
                'processed': 1,
                'result': result,
                'completed_at': datetime.utcnow().isoformat(),
            }).eq('id', job_id).execute()

            logger.info(f"Completed job {job_id} for asset {asset_id}")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")

            # Truncate error message and remove HTML
            error_msg = str(e)[:200]
            if '<!DOCTYPE' in error_msg or '<html' in error_msg.lower():
                error_msg = "Failed to load image: URL returned HTML instead of image"

            # Mark job as failed
            self._client.table('ai_processing_jobs').update({
                'status': 'failed',
                'error_message': error_msg,
                'completed_at': datetime.utcnow().isoformat(),
            }).eq('id', job_id).execute()

    async def _load_image_from_url(self, url: str):
        """Load image from URL."""
        try:
            import httpx
            from PIL import Image
            import io

            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()

                # Check content type is an image
                content_type = response.headers.get('content-type', '')
                if 'image' not in content_type.lower():
                    logger.warning(f"URL returned non-image content-type: {content_type}")
                    return None

                image = Image.open(io.BytesIO(response.content))
                return image

        except Exception as e:
            logger.warning(f"Failed to load image from URL: {e}")
            return None

    async def _load_asset_image(self, asset_id: str, user_id: str):
        """Load image from Supabase storage."""
        try:
            result = self._client.table('assets').select(
                'path, web_uri, user_id'
            ).eq('id', asset_id).single().execute()

            if not result.data:
                logger.error(f"Asset {asset_id} not found")
                return None

            # Try web_uri first (public URL), then path (storage path)
            web_uri = result.data.get('web_uri')
            if web_uri:
                image = await self._load_image_from_url(web_uri)
                if image:
                    return image

            # Fall back to storage path
            storage_path = result.data.get('path')
            if not storage_path:
                logger.error(f"Asset {asset_id} has no path or web_uri")
                return None

            # Determine bucket from path or use default
            bucket = 'assets'
            if storage_path.startswith('thumbnails/'):
                bucket = 'thumbnails'

            file_data = self._client.storage.from_(bucket).download(storage_path)

            from PIL import Image
            import io
            image = Image.open(io.BytesIO(file_data))

            return image

        except Exception as e:
            logger.error(f"Failed to load asset image {asset_id}: {e}")
            return None

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signal."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info(f"Stopping worker {self._worker_id}")
        self._running = False


async def run_worker(
    worker_id: Optional[str] = None,
    poll_interval: float = 5.0
) -> None:
    """Run the local worker."""
    worker = LocalWorker(
        worker_id=worker_id,
        poll_interval=poll_interval
    )
    await worker.start()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kizu AI Local Worker")
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Unique worker identifier"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between job polls"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    asyncio.run(run_worker(
        worker_id=args.worker_id,
        poll_interval=args.poll_interval
    ))
