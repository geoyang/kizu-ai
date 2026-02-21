"""Unified worker that processes all job types from Supabase.

Handles multiple job types:
- ai_processing_jobs table:
  - Image AI processing (embeddings, faces, objects) - default
  - Moments generation (generate_moments job_type)
  - Album video generation (album_video job_type)
- image_processing_jobs table:
  - Thumbnail and web version generation
"""

import asyncio
import base64
import io
import logging
import os
import signal
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable

import httpx
from PIL import Image, ImageOps
from supabase import create_client, Client

from api.config import settings
from api.services.process_service import ProcessService
from api.stores import SupabaseVectorStore
from api.schemas.requests import ProcessingOperation
from api.utils.image_utils import load_image

# Import pillow-heif for HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False

logger = logging.getLogger(__name__)

# Default settings for moments
DEFAULT_MOMENT_SETTINGS = {
    'moments': True,
    'moments_time': '09:00',
    'moments_auto_delete_days': 7,
    'moments_location': True,
    'moments_people': True,
    'moments_events': True,
    'moments_on_this_day': True,
    'moments_then_and_now': True,
}

# Image processing settings
THUMBNAIL_SIZE = (300, 300)
THUMBNAIL_QUALITY = 80
WEB_VERSION_QUALITY = 92
WEB_MAX_SIZE = 4096

# EXIF orientation mapping
ORIENTATION_MAP = {
    1: (0, False),      # Normal
    2: (0, True),       # Flipped horizontally
    3: (180, False),    # Rotated 180
    4: (180, True),     # Flipped vertically
    5: (270, True),     # Rotated 90 CCW + flipped
    6: (270, False),    # Rotated 90 CCW
    7: (90, True),      # Rotated 90 CW + flipped
    8: (90, False),     # Rotated 90 CW
}


class UnifiedWorker:
    """
    Unified worker that polls multiple job tables and processes all job types.
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
        self._current_job_table: Optional[str] = None

    @property
    def worker_id(self) -> str:
        return self._worker_id

    async def start(self) -> None:
        """Start the worker."""
        logger.info(f"Starting unified worker: {self._worker_id}")

        if not HEIF_SUPPORTED:
            logger.warning("HEIC/HEIF support not available - install pillow-heif")

        # Initialize Supabase client
        self._client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key
        )

        # Initialize process service for AI jobs
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
        """Main worker loop - polls all job tables."""
        logger.info(f"Worker {self._worker_id} entering main loop")

        # Start realtime subscription in background
        realtime_task = asyncio.create_task(self._subscribe_realtime())

        try:
            while self._running:
                # Poll AI processing jobs (includes album_video job type)
                await self._poll_ai_jobs()

                # Poll image processing jobs
                await self._poll_image_jobs()

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
            channel = self._client.channel('job_changes')

            def on_ai_insert(payload):
                logger.info(f"Realtime: New AI job detected")
                asyncio.create_task(self._poll_ai_jobs())

            def on_image_insert(payload):
                logger.info(f"Realtime: New image job detected")
                asyncio.create_task(self._poll_image_jobs())

            channel.on_postgres_changes(
                event='INSERT',
                schema='public',
                table='ai_processing_jobs',
                callback=on_ai_insert
            )

            channel.on_postgres_changes(
                event='INSERT',
                schema='public',
                table='image_processing_jobs',
                callback=on_image_insert
            )

            await channel.subscribe()
            logger.info("Subscribed to Supabase Realtime for job tables")

            while self._running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.warning(f"Realtime subscription error: {e}")
            logger.info("Falling back to polling only")

    # =========================================================================
    # AI Processing Jobs (embeddings, faces, objects, moments)
    # =========================================================================

    async def _poll_ai_jobs(self) -> None:
        """Poll for AI processing jobs."""
        if self._current_job_id:
            return

        try:
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

            # Claim the job
            update_result = self._client.table('ai_processing_jobs').update({
                'status': 'processing',
                'worker_id': self._worker_id,
                'picked_up_at': datetime.utcnow().isoformat(),
                'ai_version': settings.version,
            }).eq('id', job['id']).eq('status', 'pending').execute()

            if not update_result.data or len(update_result.data) == 0:
                logger.debug(f"AI job {job['id']} already claimed")
                return

            self._current_job_id = job['id']
            self._current_job_table = 'ai_processing_jobs'

            job_type = job.get('job_type', 'image_ai')
            logger.info(f"Claimed AI job {job['id']} (type: {job_type})")

            await self._process_ai_job(job)

        except Exception as e:
            logger.error(f"Error polling AI jobs: {e}")
        finally:
            self._current_job_id = None
            self._current_job_table = None

    async def _process_ai_job(self, job: dict) -> None:
        """Route AI job to appropriate handler."""
        job_type = job.get('job_type', 'image_ai')

        if job_type == 'generate_moments':
            await self._process_moments_job(job)
        elif job_type == 'album_video':
            await self._process_video_job(job)
        else:
            await self._process_image_ai_job(job)

    async def _process_image_ai_job(self, job: dict) -> None:
        """Process image AI job (embeddings, faces, objects)."""
        job_id = job['id']
        user_id = job['user_id']
        input_params = job.get('input_params') or {}
        asset_id = input_params.get('asset_id')
        image_url = input_params.get('image_url')
        operations = input_params.get('operations', ['embedding', 'faces', 'objects'])

        try:
            # Load image
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

            self._client.table('ai_processing_jobs').update({
                'status': 'completed',
                'progress': 100,
                'processed': 1,
                'result': result,
                'completed_at': datetime.utcnow().isoformat(),
            }).eq('id', job_id).execute()

            logger.info(f"Completed AI job {job_id} for asset {asset_id}")

        except Exception as e:
            logger.error(f"AI job {job_id} failed: {e}")
            error_msg = str(e)[:200]
            if '<!DOCTYPE' in error_msg or '<html' in error_msg.lower():
                error_msg = "Failed to load image: URL returned HTML instead of image"

            self._client.table('ai_processing_jobs').update({
                'status': 'failed',
                'error_message': error_msg,
                'completed_at': datetime.utcnow().isoformat(),
            }).eq('id', job_id).execute()

    async def _process_moments_job(self, job: dict) -> None:
        """Process moments generation job."""
        from api.services.moments_service import MomentsService

        job_id = job['id']
        user_id = job.get('user_id')

        if not user_id:
            logger.error(f"[{job_id}] No user_id in moments job")
            self._fail_ai_job(job_id, "No user_id provided")
            return

        try:
            settings_data = await self._get_user_moment_settings(user_id)

            if not settings_data.get('moments', True):
                logger.info(f"[{job_id}] Moments disabled for user {user_id}")
                self._complete_ai_job(job_id, {'skipped': True, 'reason': 'moments_disabled'})
                return

            self._update_ai_job_progress(job_id, 10, 'initializing')

            moments_service = MomentsService(self._client)

            def on_progress(stage: str, detail: str, progress: int):
                self._update_ai_job_progress(job_id, progress, f"{stage}: {detail}")

            moments = await moments_service.generate_moments_for_user(
                user_id,
                settings_data,
                on_progress=on_progress
            )

            logger.info(f"[{job_id}] Generated {len(moments)} moments")

            saved_count = 0
            auto_delete_days = settings_data.get('moments_auto_delete_days', 7)

            for moment in moments:
                try:
                    await self._save_moment(user_id, moment, auto_delete_days)
                    saved_count += 1
                except Exception as e:
                    logger.error(f"[{job_id}] Failed to save moment: {e}")

            self._complete_ai_job(job_id, {
                'moments_generated': len(moments),
                'moments_saved': saved_count,
            })

            if saved_count > 0:
                await self._queue_moments_notification(user_id, saved_count)

            logger.info(f"[{job_id}] Moments job completed: {saved_count} saved")

        except Exception as e:
            error_msg = str(e)[:500]
            logger.error(f"[{job_id}] Moments job failed: {error_msg}")
            self._fail_ai_job(job_id, error_msg)

    # =========================================================================
    # Image Processing Jobs (thumbnails, web versions)
    # =========================================================================

    async def _poll_image_jobs(self) -> None:
        """Poll for image processing jobs."""
        if self._current_job_id:
            return

        try:
            result = self._client.table('image_processing_jobs').select(
                '*'
            ).eq(
                'status', 'pending'
            ).order(
                'created_at', desc=False
            ).limit(1).execute()

            if not result.data or len(result.data) == 0:
                return

            job = result.data[0]

            # Claim the job
            update_result = self._client.table('image_processing_jobs').update({
                'status': 'processing',
                'started_at': datetime.utcnow().isoformat(),
                'current_step': 'downloading',
            }).eq('id', job['id']).eq('status', 'pending').execute()

            if not update_result.data or len(update_result.data) == 0:
                logger.debug(f"Image job {job['id']} already claimed")
                return

            self._current_job_id = job['id']
            self._current_job_table = 'image_processing_jobs'

            # Update asset status
            self._client.table('assets').update({
                'image_processing_status': 'processing'
            }).eq('id', job['asset_id']).execute()

            logger.info(f"Claimed image job {job['id']} for asset {job['asset_id']}")

            await self._process_image_job(job)

        except Exception as e:
            logger.error(f"Error polling image jobs: {e}")
        finally:
            self._current_job_id = None
            self._current_job_table = None

    async def _process_image_job(self, job: dict) -> None:
        """Process image job (thumbnail, web version)."""
        job_id = job['id']
        asset_id = job['asset_id']
        user_id = job['user_id']
        input_url = job['input_url']
        orientation = job.get('input_orientation')
        needs_web_version = job.get('needs_web_version', False)
        metadata = job.get('metadata') or {}

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download input image
                logger.info(f"[{job_id}] Downloading image")
                self._update_image_job_progress(job_id, 10, 'downloading')

                input_path = os.path.join(temp_dir, "input")
                await self._download_file(input_url, input_path)

                # Open image
                logger.info(f"[{job_id}] Processing image")
                self._update_image_job_progress(job_id, 20, 'processing')
                img = Image.open(input_path)

                # Handle EXIF orientation
                img = self._apply_orientation(img, orientation)

                # Generate thumbnail
                thumbnail_url = None
                if metadata.get('generate_thumbnail', True):
                    logger.info(f"[{job_id}] Generating thumbnail")
                    self._update_image_job_progress(job_id, 40, 'generating_thumbnail')
                    thumbnail_url = await self._generate_and_upload_thumbnail(
                        img, user_id, asset_id, temp_dir
                    )

                # Generate web version
                web_url = None
                if needs_web_version and metadata.get('generate_web_version', True):
                    logger.info(f"[{job_id}] Generating web version")
                    self._update_image_job_progress(job_id, 70, 'generating_web_version')
                    web_url = await self._generate_and_upload_web_version(
                        img, user_id, asset_id, temp_dir
                    )

                # Complete job
                self._update_image_job_progress(job_id, 90, 'uploading')

                self._client.table('image_processing_jobs').update({
                    'status': 'completed',
                    'progress': 100,
                    'current_step': 'completed',
                    'thumbnail_url': thumbnail_url,
                    'web_url': web_url,
                    'completed_at': datetime.utcnow().isoformat(),
                }).eq('id', job_id).execute()

                # Update asset
                update_data = {'image_processing_status': 'completed'}
                if thumbnail_url:
                    update_data['thumbnail'] = thumbnail_url
                if web_url:
                    update_data['web_uri'] = web_url

                self._client.table('assets').update(update_data).eq('id', asset_id).execute()

                logger.info(f"[{job_id}] Image job completed")

        except Exception as e:
            error_msg = str(e)[:500]
            logger.error(f"[{job_id}] Image job failed: {error_msg}")

            self._client.table('image_processing_jobs').update({
                'status': 'failed',
                'error_message': error_msg,
                'completed_at': datetime.utcnow().isoformat(),
            }).eq('id', job_id).execute()

            self._client.table('assets').update({
                'image_processing_status': 'failed'
            }).eq('id', asset_id).execute()

    def _apply_orientation(self, img: Image.Image, orientation: Optional[int]) -> Image.Image:
        """Apply EXIF orientation correction.

        If no explicit orientation is provided, reads EXIF from the image itself.
        This handles chat images where orientation metadata isn't passed through.
        """
        if not orientation or orientation == 1:
            # No explicit orientation — auto-detect from EXIF embedded in the image
            try:
                return ImageOps.exif_transpose(img)
            except Exception:
                return img

        if orientation not in ORIENTATION_MAP:
            return img

        rotation, flip = ORIENTATION_MAP[orientation]

        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if rotation:
            img = img.rotate(-rotation, expand=True)

        return img

    async def _generate_and_upload_thumbnail(
        self, img: Image.Image, user_id: str, asset_id: str, temp_dir: str
    ) -> str:
        """Generate thumbnail and return as base64 data URI."""
        thumb = img.copy()

        # Center crop to square
        width, height = thumb.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        thumb = thumb.crop((left, top, left + min_dim, top + min_dim))

        # Resize
        thumb = thumb.resize(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)

        # Convert to RGB
        if thumb.mode in ('RGBA', 'P'):
            thumb = thumb.convert('RGB')

        # Save to buffer
        buffer = io.BytesIO()
        thumb.save(buffer, format='JPEG', quality=THUMBNAIL_QUALITY)
        buffer.seek(0)

        # Encode as base64 data URI
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"

    async def _generate_and_upload_web_version(
        self, img: Image.Image, user_id: str, asset_id: str, temp_dir: str
    ) -> str:
        """Generate and upload web version."""
        web_img = img.copy()

        # Resize if needed
        width, height = web_img.size
        if width > WEB_MAX_SIZE or height > WEB_MAX_SIZE:
            if width > height:
                new_width = WEB_MAX_SIZE
                new_height = int(height * (WEB_MAX_SIZE / width))
            else:
                new_height = WEB_MAX_SIZE
                new_width = int(width * (WEB_MAX_SIZE / height))
            web_img = web_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert to RGB
        if web_img.mode in ('RGBA', 'P'):
            web_img = web_img.convert('RGB')

        # Save to buffer
        buffer = io.BytesIO()
        web_img.save(buffer, format='JPEG', quality=WEB_VERSION_QUALITY)
        buffer.seek(0)

        # Upload
        upload_path = f"{user_id}/processed/{asset_id}/web.jpg"
        return await self._upload_to_supabase(buffer.getvalue(), upload_path, "image/jpeg")

    async def _download_file(self, url: str, dest_path: str) -> None:
        """Download file from URL."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(dest_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)

    async def _upload_to_supabase(self, file_data: bytes, upload_path: str, content_type: str) -> str:
        """Upload file to Supabase Storage."""
        bucket = "assets"

        async with httpx.AsyncClient(timeout=120.0) as client:
            upload_url = f"{settings.supabase_url}/storage/v1/object/{bucket}/{upload_path}"

            response = await client.post(
                upload_url,
                content=file_data,
                headers={
                    "Authorization": f"Bearer {settings.supabase_service_role_key}",
                    "Content-Type": content_type,
                    "x-upsert": "true"
                }
            )

            if response.status_code not in [200, 201]:
                raise RuntimeError(f"Upload failed: {response.status_code}")

        return f"{settings.supabase_url}/storage/v1/object/public/{bucket}/{upload_path}"

    def _update_image_job_progress(self, job_id: str, progress: int, step: str) -> None:
        """Update image job progress."""
        self._client.table('image_processing_jobs').update({
            'progress': progress,
            'current_step': step,
        }).eq('id', job_id).execute()

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _get_user_moment_settings(self, user_id: str) -> dict:
        """Get user's moment preferences."""
        try:
            result = self._client.table('profiles').select(
                'notification_settings'
            ).eq('id', user_id).single().execute()

            if result.data:
                ns = result.data.get('notification_settings') or {}
                return {
                    'moments': ns.get('moments', DEFAULT_MOMENT_SETTINGS['moments']),
                    'moments_time': ns.get('moments_time', DEFAULT_MOMENT_SETTINGS['moments_time']),
                    'moments_auto_delete_days': ns.get('moments_auto_delete_days', DEFAULT_MOMENT_SETTINGS['moments_auto_delete_days']),
                    'moments_location': ns.get('moments_location', DEFAULT_MOMENT_SETTINGS['moments_location']),
                    'moments_people': ns.get('moments_people', DEFAULT_MOMENT_SETTINGS['moments_people']),
                    'moments_events': ns.get('moments_events', DEFAULT_MOMENT_SETTINGS['moments_events']),
                    'moments_on_this_day': ns.get('moments_on_this_day', DEFAULT_MOMENT_SETTINGS['moments_on_this_day']),
                    'moments_then_and_now': ns.get('moments_then_and_now', DEFAULT_MOMENT_SETTINGS['moments_then_and_now']),
                }
        except Exception as e:
            logger.warning(f"Could not fetch user settings: {e}")

        return DEFAULT_MOMENT_SETTINGS.copy()

    async def _save_moment(self, user_id: str, moment, auto_delete_days: int) -> str:
        """Save a generated moment to the database."""
        moment_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(days=auto_delete_days)

        moment_record = {
            'id': moment_id,
            'user_id': user_id,
            'grouping_type': moment.grouping_type,
            'grouping_criteria': moment.grouping_criteria,
            'title': moment.title,
            'subtitle': moment.subtitle,
            'cover_asset_ids': moment.cover_asset_ids,
            'date_range_start': moment.date_range_start.isoformat() if moment.date_range_start else None,
            'date_range_end': moment.date_range_end.isoformat() if moment.date_range_end else None,
            'is_saved': False,
            'expires_at': expires_at.isoformat(),
        }

        self._client.table('moments').insert(moment_record).execute()

        # Insert moment_assets
        asset_records = [
            {
                'id': str(uuid.uuid4()),
                'moment_id': moment_id,
                'asset_id': asset_id,
                'display_order': idx,
            }
            for idx, asset_id in enumerate(moment.all_asset_ids)
        ]

        chunk_size = 100
        for i in range(0, len(asset_records), chunk_size):
            chunk = asset_records[i:i + chunk_size]
            self._client.table('moment_assets').insert(chunk).execute()

        return moment_id

    async def _queue_moments_notification(self, user_id: str, moment_count: int) -> None:
        """Queue push notification for new moments."""
        try:
            action_log_id = str(uuid.uuid4())
            self._client.table('actions_log').insert({
                'id': action_log_id,
                'user_id': user_id,
                'action_type': 'moment_generated',
                'entity_type': 'moment',
                'metadata': {'count': moment_count},
            }).execute()

            title = "New Moments Available" if moment_count > 1 else "New Moment Available"
            body = f"You have {moment_count} new photo moment{'s' if moment_count > 1 else ''} to explore!"

            self._client.table('notification_queue').insert({
                'id': str(uuid.uuid4()),
                'action_log_id': action_log_id,
                'recipient_user_id': user_id,
                'notification_type': 'moment',
                'title': title,
                'body': body,
                'deep_link_type': 'moment',
                'deep_link_id': 'feed',
            }).execute()

            logger.info(f"Queued notification for {moment_count} moments")

        except Exception as e:
            logger.warning(f"Failed to queue notification: {e}")

    def _update_ai_job_progress(self, job_id: str, progress: int, step: str) -> None:
        """Update AI job progress."""
        self._client.table('ai_processing_jobs').update({
            'progress': progress,
            'current_step': step,
        }).eq('id', job_id).execute()

    def _complete_ai_job(self, job_id: str, result: dict) -> None:
        """Mark AI job as completed."""
        self._client.table('ai_processing_jobs').update({
            'status': 'completed',
            'progress': 100,
            'current_step': 'completed',
            'result': result,
            'completed_at': datetime.utcnow().isoformat(),
        }).eq('id', job_id).execute()

    def _fail_ai_job(self, job_id: str, error_message: str) -> None:
        """Mark AI job as failed."""
        self._client.table('ai_processing_jobs').update({
            'status': 'failed',
            'error_message': error_message,
            'completed_at': datetime.utcnow().isoformat(),
        }).eq('id', job_id).execute()

    async def _load_image_from_url(self, url: str):
        """Load image from URL."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()

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

            web_uri = result.data.get('web_uri')
            if web_uri:
                image = await self._load_image_from_url(web_uri)
                if image:
                    return image

            storage_path = result.data.get('path')
            if not storage_path:
                logger.error(f"Asset {asset_id} has no path or web_uri")
                return None

            bucket = 'assets'
            if storage_path.startswith('thumbnails/'):
                bucket = 'thumbnails'

            file_data = self._client.storage.from_(bucket).download(storage_path)
            image = Image.open(io.BytesIO(file_data))

            return image

        except Exception as e:
            logger.error(f"Failed to load asset image {asset_id}: {e}")
            return None

    # =========================================================================
    # Album Video Jobs (job_type='album_video' in ai_processing_jobs)
    # =========================================================================

    async def _process_video_job(self, job: dict) -> None:
        """Process album video generation job using FFmpeg."""
        from api.services.video_service import VideoService

        job_id = job['id']
        user_id = job['user_id']
        params = job.get('input_params') or {}
        album_id = params.get('album_id')

        def on_progress(step: str, percent: int):
            self._update_ai_job_progress(job_id, percent, step)

        try:
            video_service = VideoService(self._client)
            result = await video_service.generate(
                album_id=album_id,
                user_id=user_id,
                config=params,
                on_progress=on_progress,
            )

            self._complete_ai_job(job_id, result)
            logger.info(f"Completed video job {job_id} for album {album_id}")

        except Exception as e:
            logger.error(f"Video job {job_id} failed: {e}", exc_info=True)
            self._fail_ai_job(job_id, str(e)[:500])

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signal."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info(f"Stopping worker {self._worker_id}")
        self._running = False


# Backward compatibility alias
LocalWorker = UnifiedWorker


async def run_worker(
    worker_id: Optional[str] = None,
    poll_interval: float = 5.0
) -> None:
    """Run the unified worker."""
    worker = UnifiedWorker(
        worker_id=worker_id,
        poll_interval=poll_interval
    )
    await worker.start()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kizu AI Unified Worker")
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
