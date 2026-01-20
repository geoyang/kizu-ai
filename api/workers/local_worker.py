"""Local worker that pulls jobs from Supabase processing queue."""

import asyncio
import logging
import signal
import uuid
from typing import Optional

from supabase import create_client, Client
from realtime.connection import Socket

from api.config import settings
from api.services.process_service import ProcessService
from api.stores import SupabaseVectorStore
from api.schemas.requests import ProcessingOperation
from api.utils.image_utils import load_image

logger = logging.getLogger(__name__)


class LocalWorker:
    """
    Pull-based worker that subscribes to Supabase Realtime
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
            # Create realtime channel for processing_queue inserts
            channel = self._client.channel('processing_queue_changes')

            def on_insert(payload):
                """Handle new job inserted."""
                logger.info(f"Realtime: New job detected: {payload}")
                # Trigger immediate poll
                asyncio.create_task(self._poll_and_process())

            channel.on_postgres_changes(
                event='INSERT',
                schema='public',
                table='processing_queue',
                callback=on_insert
            )

            await channel.subscribe()
            logger.info("Subscribed to Supabase Realtime for processing_queue")

            # Keep subscription alive
            while self._running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.warning(f"Realtime subscription error: {e}")
            logger.info("Falling back to polling only")

    async def _poll_and_process(self) -> None:
        """Poll for a job and process it."""
        if self._current_job_id:
            # Already processing a job
            return

        try:
            # Claim a job using the atomic function
            result = self._client.rpc(
                'claim_processing_job',
                {
                    'p_worker_id': self._worker_id,
                    'p_max_attempts': self._max_attempts
                }
            ).execute()

            if not result.data or len(result.data) == 0:
                # No jobs available
                return

            job = result.data[0]
            self._current_job_id = job['id']

            logger.info(
                f"Claimed job {job['id']} for asset {job['asset_id']} "
                f"(attempt {job['attempts']})"
            )

            # Process the job
            await self._process_job(job)

        except Exception as e:
            logger.error(f"Error in poll_and_process: {e}")
        finally:
            self._current_job_id = None

    async def _process_job(self, job: dict) -> None:
        """Process a single job."""
        job_id = job['id']
        asset_id = job['asset_id']
        user_id = job['user_id']
        operations = job['operations']

        try:
            # Load the image from Supabase storage
            image = await self._load_asset_image(asset_id, user_id)

            if image is None:
                raise ValueError(f"Could not load image for asset {asset_id}")

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
                asset_id=asset_id,
                image=image,
                operations=ops,
                user_id=user_id,
                store_results=True
            )

            # Mark job as completed
            self._client.rpc(
                'complete_processing_job',
                {
                    'p_job_id': job_id,
                    'p_result': result
                }
            ).execute()

            logger.info(f"Completed job {job_id} for asset {asset_id}")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")

            # Mark job as failed
            self._client.rpc(
                'fail_processing_job',
                {
                    'p_job_id': job_id,
                    'p_error': str(e),
                    'p_max_attempts': self._max_attempts
                }
            ).execute()

    async def _load_asset_image(self, asset_id: str, user_id: str):
        """Load image from Supabase storage."""
        try:
            # Get asset record to find storage path
            result = self._client.table('assets').select(
                'storage_path, bucket'
            ).eq('id', asset_id).single().execute()

            if not result.data:
                logger.error(f"Asset {asset_id} not found")
                return None

            storage_path = result.data.get('storage_path')
            bucket = result.data.get('bucket', 'assets')

            if not storage_path:
                logger.error(f"Asset {asset_id} has no storage_path")
                return None

            # Download from Supabase storage
            file_data = self._client.storage.from_(bucket).download(storage_path)

            # Load as PIL Image
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
