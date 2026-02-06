"""Video transcoding worker that pulls jobs from Supabase video_transcoding_jobs table."""

import asyncio
import logging
import os
import signal
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from supabase import create_client, Client

from api.config import settings

logger = logging.getLogger(__name__)

# Quality presets (CRF values for H.264 - lower = better quality, larger file)
QUALITY_PRESETS = {
    "low": {"crf": 28, "preset": "veryfast"},
    "medium": {"crf": 23, "preset": "medium"},
    "high": {"crf": 18, "preset": "slow"},
    "lossless": {"crf": 0, "preset": "veryslow"},
}

# Resolution settings
RESOLUTIONS = {
    "480p": {"width": 854, "height": 480},
    "720p": {"width": 1280, "height": 720},
    "1080p": {"width": 1920, "height": 1080},
    "original": None,
}


class VideoWorker:
    """
    Pull-based worker that polls video_transcoding_jobs table
    and transcodes videos using FFmpeg.
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        poll_interval: float = 10.0,
    ):
        self._worker_id = worker_id or f"video-worker-{uuid.uuid4().hex[:8]}"
        self._poll_interval = poll_interval
        self._running = False
        self._client: Optional[Client] = None
        self._current_job_id: Optional[str] = None

    @property
    def worker_id(self) -> str:
        return self._worker_id

    async def start(self) -> None:
        """Start the worker."""
        logger.info(f"Starting video worker: {self._worker_id}")

        # Check FFmpeg availability
        if not self._check_ffmpeg():
            logger.error("FFmpeg not found. Please install FFmpeg to use the video worker.")
            return

        # Initialize Supabase client
        self._client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key
        )

        self._running = True

        # Set up signal handlers for graceful shutdown
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle_shutdown)

        # Start the main loop
        await self._run_loop()

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    async def _run_loop(self) -> None:
        """Main worker loop."""
        logger.info(f"Video worker {self._worker_id} entering main loop")

        try:
            while self._running:
                await self._poll_and_process()
                await asyncio.sleep(self._poll_interval)

        except asyncio.CancelledError:
            logger.info("Video worker loop cancelled")

        logger.info(f"Video worker {self._worker_id} stopped")

    async def _poll_and_process(self) -> None:
        """Poll for a job and process it."""
        if self._current_job_id:
            return

        try:
            # Find oldest pending job
            result = self._client.table('video_transcoding_jobs').select(
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
            update_result = self._client.table('video_transcoding_jobs').update({
                'status': 'processing',
                'started_at': datetime.utcnow().isoformat(),
                'current_step': 'downloading',
            }).eq('id', job['id']).eq('status', 'pending').execute()

            if not update_result.data or len(update_result.data) == 0:
                logger.debug(f"Job {job['id']} already claimed by another worker")
                return

            self._current_job_id = job['id']

            # Update asset status
            self._client.table('assets').update({
                'transcoding_status': 'processing'
            }).eq('id', job['asset_id']).execute()

            logger.info(f"Claimed job {job['id']} for asset {job['asset_id']}")

            await self._process_job(job)

        except Exception as e:
            logger.error(f"Error in poll_and_process: {e}")
        finally:
            self._current_job_id = None

    async def _process_job(self, job: dict) -> None:
        """Process a single transcoding job."""
        job_id = job['id']
        asset_id = job['asset_id']
        user_id = job['user_id']
        input_url = job['input_url']

        # If input_url is just a storage path (not a full URL), construct the full URL
        if not input_url.startswith('http://') and not input_url.startswith('https://'):
            input_url = f"{settings.supabase_url}/storage/v1/object/public/assets/{input_url}"
            logger.info(f"[{job_id}] Constructed full URL: {input_url}")

        quality_preset = job.get('quality_preset', 'medium')
        target_resolution = job.get('target_resolution', '720p')
        metadata = job.get('metadata') or {}
        generate_thumbnail = metadata.get('generate_thumbnail', True)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Determine input filename from URL
                input_ext = Path(input_url.split("?")[0]).suffix or ".mp4"
                input_path = os.path.join(temp_dir, f"input{input_ext}")
                output_path = os.path.join(temp_dir, "output.mp4")
                thumbnail_path = os.path.join(temp_dir, "thumbnail.jpg")

                # Download input video
                logger.info(f"[{job_id}] Downloading video from {input_url}")
                self._update_job_progress(job_id, 10, 'downloading')
                await self._download_file(input_url, input_path)

                # Get input video info
                input_info = self._get_video_info(input_path)
                logger.info(f"[{job_id}] Input: {input_info}")

                # Update job with input info
                self._client.table('video_transcoding_jobs').update({
                    'input_format': input_info.get('format'),
                    'input_duration_ms': input_info.get('duration_ms'),
                    'input_width': input_info.get('width'),
                    'input_height': input_info.get('height'),
                    'input_size_bytes': input_info.get('size_bytes'),
                }).eq('id', job_id).execute()

                # Transcode
                logger.info(f"[{job_id}] Transcoding ({quality_preset}, {target_resolution})")
                self._update_job_progress(job_id, 30, 'transcoding')
                output_info = self._transcode_video(
                    input_path,
                    output_path,
                    quality_preset,
                    target_resolution
                )
                logger.info(f"[{job_id}] Transcode complete: {output_info}")

                # Generate thumbnail
                thumbnail_generated = False
                if generate_thumbnail:
                    self._update_job_progress(job_id, 70, 'generating_thumbnail')
                    thumb_time = min(2.0, (input_info.get('duration_ms', 0) / 1000) * 0.1)
                    thumbnail_generated = self._generate_thumbnail(input_path, thumbnail_path, thumb_time)
                    logger.info(f"[{job_id}] Thumbnail: {'generated' if thumbnail_generated else 'failed'}")

                # Upload to Supabase
                self._update_job_progress(job_id, 80, 'uploading')
                upload_path = f"{user_id}/videos/{asset_id}"

                output_url = await self._upload_to_supabase(
                    output_path,
                    f"{upload_path}/web.mp4",
                    "video/mp4"
                )
                logger.info(f"[{job_id}] Video uploaded: {output_url}")

                thumbnail_url = None
                if thumbnail_generated and os.path.exists(thumbnail_path):
                    thumbnail_url = await self._upload_to_supabase(
                        thumbnail_path,
                        f"{upload_path}/thumbnail.jpg",
                        "image/jpeg"
                    )
                    logger.info(f"[{job_id}] Thumbnail uploaded: {thumbnail_url}")

                # Mark job as completed
                self._client.table('video_transcoding_jobs').update({
                    'status': 'completed',
                    'progress': 100,
                    'current_step': 'completed',
                    'output_url': output_url,
                    'thumbnail_url': thumbnail_url,
                    'output_format': 'mp4',
                    'output_codec': 'h264',
                    'output_width': output_info.get('width'),
                    'output_height': output_info.get('height'),
                    'output_size_bytes': output_info.get('size_bytes'),
                    'output_duration_ms': output_info.get('duration_ms'),
                    'completed_at': datetime.utcnow().isoformat(),
                }).eq('id', job_id).execute()

                # Update asset with transcoded URLs
                self._client.table('assets').update({
                    'web_uri': output_url,
                    'thumbnail': thumbnail_url or input_url,
                    'transcoding_status': 'completed',
                }).eq('id', asset_id).execute()

                logger.info(f"[{job_id}] Job completed successfully")

        except Exception as e:
            error_msg = str(e)[:500]
            logger.error(f"[{job_id}] Job failed: {error_msg}")

            self._client.table('video_transcoding_jobs').update({
                'status': 'failed',
                'error_message': error_msg,
                'completed_at': datetime.utcnow().isoformat(),
            }).eq('id', job_id).execute()

            self._client.table('assets').update({
                'transcoding_status': 'failed'
            }).eq('id', asset_id).execute()

    def _update_job_progress(self, job_id: str, progress: int, step: str) -> None:
        """Update job progress."""
        self._client.table('video_transcoding_jobs').update({
            'progress': progress,
            'current_step': step,
        }).eq('id', job_id).execute()

    async def _download_file(self, url: str, dest_path: str) -> None:
        """Download a file from URL."""
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(dest_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)

    def _get_video_info(self, file_path: str) -> dict:
        """Get video metadata using ffprobe."""
        import json

        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            file_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        data = json.loads(result.stdout)

        video_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
            None
        )
        if not video_stream:
            raise RuntimeError("No video stream found")

        format_info = data.get("format", {})
        duration_sec = float(format_info.get("duration", 0))
        size_bytes = int(format_info.get("size", 0))

        return {
            "duration_ms": int(duration_sec * 1000),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "codec": video_stream.get("codec_name", "unknown"),
            "format": format_info.get("format_name", "unknown"),
            "size_bytes": size_bytes
        }

    def _transcode_video(
        self,
        input_path: str,
        output_path: str,
        quality_preset: str,
        target_resolution: str
    ) -> dict:
        """Transcode video using FFmpeg."""
        quality = QUALITY_PRESETS.get(quality_preset, QUALITY_PRESETS["medium"])
        resolution = RESOLUTIONS.get(target_resolution)

        cmd = ["ffmpeg", "-y", "-i", input_path]

        # Video codec settings
        cmd.extend(["-c:v", "libx264"])
        cmd.extend(["-crf", str(quality["crf"])])
        cmd.extend(["-preset", quality["preset"]])
        # Use main profile for better iOS compatibility
        cmd.extend(["-profile:v", "main", "-level", "4.0"])

        # Build video filter chain with pixel format conversion for iOS compatibility
        # Use setrange=tv to ensure limited/TV color range (not full/PC range)
        vf_filters = ["format=yuv420p", "setrange=tv"]
        if resolution:
            vf_filters.insert(0, f"scale='min({resolution['width']},iw)':'min({resolution['height']},ih)':force_original_aspect_ratio=decrease")
        cmd.extend(["-vf", ",".join(vf_filters)])
        # Force TV color range in output metadata
        cmd.extend(["-color_range", "tv"])

        # Audio settings
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])

        # Fast start for web streaming
        cmd.extend(["-movflags", "+faststart"])

        cmd.append(output_path)

        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {process.stderr[:500]}")

        return self._get_video_info(output_path)

    def _generate_thumbnail(self, input_path: str, output_path: str, time_seconds: float) -> bool:
        """Generate a thumbnail from video."""
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(time_seconds),
            "-i", input_path,
            "-vframes", "1",
            "-vf", "scale=640:-1",
            "-q:v", "2",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Try at start if specified time fails
            cmd[2] = "0"
            result = subprocess.run(cmd, capture_output=True, text=True)

        return result.returncode == 0

    async def _upload_to_supabase(self, file_path: str, upload_path: str, content_type: str) -> str:
        """Upload file to Supabase Storage."""
        bucket = "assets"

        with open(file_path, "rb") as f:
            file_data = f.read()

        supabase_url = settings.supabase_url
        supabase_key = settings.supabase_service_role_key

        async with httpx.AsyncClient(timeout=300.0) as client:
            upload_url = f"{supabase_url}/storage/v1/object/{bucket}/{upload_path}"

            response = await client.post(
                upload_url,
                content=file_data,
                headers={
                    "Authorization": f"Bearer {supabase_key}",
                    "Content-Type": content_type,
                    "x-upsert": "true"
                }
            )

            if response.status_code not in [200, 201]:
                raise RuntimeError(f"Upload failed: {response.status_code} - {response.text[:200]}")

        return f"{supabase_url}/storage/v1/object/public/{bucket}/{upload_path}"

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signal."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info(f"Stopping video worker {self._worker_id}")
        self._running = False


async def run_video_worker(
    worker_id: Optional[str] = None,
    poll_interval: float = 10.0
) -> None:
    """Run the video transcoding worker."""
    worker = VideoWorker(
        worker_id=worker_id,
        poll_interval=poll_interval
    )
    await worker.start()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kizu AI Video Transcoding Worker")
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Unique worker identifier"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
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

    asyncio.run(run_video_worker(
        worker_id=args.worker_id,
        poll_interval=args.poll_interval
    ))
