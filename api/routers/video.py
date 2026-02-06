"""Video processing API router - transcoding and thumbnail generation."""

import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import httpx

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video", tags=["video"])


class TranscodeRequest(BaseModel):
    """Request to transcode a video."""
    job_id: str
    input_url: str
    output_format: str = "mp4"
    output_codec: str = "h264"
    quality_preset: str = "medium"  # low, medium, high, lossless
    target_resolution: str = "720p"  # 480p, 720p, 1080p, original
    generate_thumbnail: bool = True
    thumbnail_time_seconds: float = 2.0
    callback_url: Optional[str] = None
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    upload_path: Optional[str] = None


class TranscodeResponse(BaseModel):
    """Response from transcoding."""
    success: bool
    job_id: str
    output_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration_ms: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    size_bytes: Optional[int] = None
    error: Optional[str] = None


class VideoInfo(BaseModel):
    """Video metadata."""
    duration_ms: int
    width: int
    height: int
    codec: str
    bitrate_kbps: Optional[int] = None
    format: str
    size_bytes: int


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


def get_video_info(file_path: str) -> VideoInfo:
    """Get video metadata using ffprobe."""
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

    import json
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
    bitrate = int(format_info.get("bit_rate", 0)) // 1000 if format_info.get("bit_rate") else None

    return VideoInfo(
        duration_ms=int(duration_sec * 1000),
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        codec=video_stream.get("codec_name", "unknown"),
        bitrate_kbps=bitrate,
        format=format_info.get("format_name", "unknown"),
        size_bytes=size_bytes
    )


def transcode_video(
    input_path: str,
    output_path: str,
    quality_preset: str = "medium",
    target_resolution: str = "720p",
    output_codec: str = "h264",
    progress_callback=None
) -> dict:
    """Transcode video using FFmpeg."""
    quality = QUALITY_PRESETS.get(quality_preset, QUALITY_PRESETS["medium"])
    resolution = RESOLUTIONS.get(target_resolution)

    # Build FFmpeg command
    cmd = ["ffmpeg", "-y", "-i", input_path]

    # Video codec settings
    if output_codec == "h264":
        cmd.extend(["-c:v", "libx264"])
        cmd.extend(["-crf", str(quality["crf"])])
        cmd.extend(["-preset", quality["preset"]])
        # Ensure compatibility with web players
        cmd.extend(["-profile:v", "high", "-level", "4.1"])
        cmd.extend(["-pix_fmt", "yuv420p"])
    elif output_codec == "vp9":
        cmd.extend(["-c:v", "libvpx-vp9"])
        cmd.extend(["-crf", str(quality["crf"])])
        cmd.extend(["-b:v", "0"])
    elif output_codec == "copy":
        cmd.extend(["-c:v", "copy"])

    # Resolution scaling (maintain aspect ratio)
    if resolution:
        cmd.extend([
            "-vf", f"scale='min({resolution['width']},iw)':min'({resolution['height']},ih)':force_original_aspect_ratio=decrease,pad={resolution['width']}:{resolution['height']}:(ow-iw)/2:(oh-ih)/2"
        ])

    # Audio settings (AAC for compatibility)
    cmd.extend(["-c:a", "aac", "-b:a", "128k"])

    # Fast start for web streaming
    cmd.extend(["-movflags", "+faststart"])

    # Output
    cmd.append(output_path)

    logger.info(f"Running FFmpeg: {' '.join(cmd)}")

    # Run FFmpeg with progress parsing
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {stderr}")

    # Get output file info
    output_info = get_video_info(output_path)

    return {
        "duration_ms": output_info.duration_ms,
        "width": output_info.width,
        "height": output_info.height,
        "size_bytes": output_info.size_bytes,
        "codec": output_info.codec,
    }


def generate_thumbnail(
    input_path: str,
    output_path: str,
    time_seconds: float = 2.0,
    width: int = 640
) -> bool:
    """Generate a thumbnail from video at specified time."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(time_seconds),
        "-i", input_path,
        "-vframes", "1",
        "-vf", f"scale={width}:-1",
        "-q:v", "2",
        output_path
    ]

    logger.info(f"Generating thumbnail: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Try at the start if the specified time fails
        cmd[2] = "0"
        result = subprocess.run(cmd, capture_output=True, text=True)

    return result.returncode == 0


async def download_file(url: str, dest_path: str) -> None:
    """Download a file from URL."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)


async def upload_to_supabase(
    file_path: str,
    upload_path: str,
    supabase_url: str,
    supabase_key: str,
    content_type: str = "video/mp4"
) -> str:
    """Upload file to Supabase Storage and return public URL."""
    bucket = "assets"

    with open(file_path, "rb") as f:
        file_data = f.read()

    async with httpx.AsyncClient(timeout=300.0) as client:
        # Upload to Supabase Storage
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
            raise RuntimeError(f"Upload failed: {response.status_code} - {response.text}")

    # Return the public URL
    return f"{supabase_url}/storage/v1/object/public/{bucket}/{upload_path}"


@router.get("/health")
async def video_health():
    """Check if video processing is available."""
    # Check if FFmpeg is available
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True
        )
        ffmpeg_available = result.returncode == 0
        ffmpeg_version = result.stdout.split("\n")[0] if ffmpeg_available else None
    except FileNotFoundError:
        ffmpeg_available = False
        ffmpeg_version = None

    return {
        "status": "healthy" if ffmpeg_available else "degraded",
        "ffmpeg_available": ffmpeg_available,
        "ffmpeg_version": ffmpeg_version
    }


@router.post("/transcode", response_model=TranscodeResponse)
async def transcode(request: TranscodeRequest, background_tasks: BackgroundTasks):
    """
    Transcode a video to web-compatible format.

    This endpoint downloads the video, transcodes it using FFmpeg,
    generates a thumbnail, and uploads the results to Supabase Storage.
    """
    job_id = request.job_id

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Determine input filename from URL
            input_ext = Path(request.input_url.split("?")[0]).suffix or ".mp4"
            input_path = os.path.join(temp_dir, f"input{input_ext}")
            output_path = os.path.join(temp_dir, f"output.{request.output_format}")
            thumbnail_path = os.path.join(temp_dir, "thumbnail.jpg")

            # Download input video
            logger.info(f"[{job_id}] Downloading video from {request.input_url}")
            await download_file(request.input_url, input_path)

            # Get input video info
            input_info = get_video_info(input_path)
            logger.info(f"[{job_id}] Input video: {input_info.width}x{input_info.height}, {input_info.duration_ms}ms, {input_info.codec}")

            # Transcode
            logger.info(f"[{job_id}] Transcoding with preset={request.quality_preset}, resolution={request.target_resolution}")
            output_info = transcode_video(
                input_path,
                output_path,
                quality_preset=request.quality_preset,
                target_resolution=request.target_resolution,
                output_codec=request.output_codec
            )
            logger.info(f"[{job_id}] Transcoding complete: {output_info}")

            # Generate thumbnail
            thumbnail_url = None
            if request.generate_thumbnail:
                # Use 10% into the video or specified time, whichever is less
                thumb_time = min(
                    request.thumbnail_time_seconds,
                    input_info.duration_ms / 1000 * 0.1
                )
                if generate_thumbnail(input_path, thumbnail_path, thumb_time):
                    logger.info(f"[{job_id}] Thumbnail generated")
                else:
                    logger.warning(f"[{job_id}] Thumbnail generation failed")

            # Upload to Supabase if credentials provided
            output_url = None
            if request.supabase_url and request.supabase_key and request.upload_path:
                # Upload transcoded video
                video_upload_path = f"{request.upload_path}/web.{request.output_format}"
                output_url = await upload_to_supabase(
                    output_path,
                    video_upload_path,
                    request.supabase_url,
                    request.supabase_key,
                    f"video/{request.output_format}"
                )
                logger.info(f"[{job_id}] Video uploaded to {output_url}")

                # Upload thumbnail
                if os.path.exists(thumbnail_path):
                    thumb_upload_path = f"{request.upload_path}/thumbnail.jpg"
                    thumbnail_url = await upload_to_supabase(
                        thumbnail_path,
                        thumb_upload_path,
                        request.supabase_url,
                        request.supabase_key,
                        "image/jpeg"
                    )
                    logger.info(f"[{job_id}] Thumbnail uploaded to {thumbnail_url}")

            return TranscodeResponse(
                success=True,
                job_id=job_id,
                output_url=output_url,
                thumbnail_url=thumbnail_url,
                duration_ms=output_info["duration_ms"],
                width=output_info["width"],
                height=output_info["height"],
                size_bytes=output_info["size_bytes"]
            )

    except Exception as e:
        logger.error(f"[{job_id}] Transcoding failed: {e}")
        return TranscodeResponse(
            success=False,
            job_id=job_id,
            error=str(e)
        )


@router.post("/thumbnail")
async def create_thumbnail(
    video_url: str,
    time_seconds: float = 2.0,
    width: int = 640
):
    """Generate a thumbnail from a video URL."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_ext = Path(video_url.split("?")[0]).suffix or ".mp4"
            input_path = os.path.join(temp_dir, f"input{input_ext}")
            thumbnail_path = os.path.join(temp_dir, "thumbnail.jpg")

            await download_file(video_url, input_path)

            if generate_thumbnail(input_path, thumbnail_path, time_seconds, width):
                with open(thumbnail_path, "rb") as f:
                    import base64
                    thumbnail_base64 = base64.b64encode(f.read()).decode()

                return {
                    "success": True,
                    "thumbnail_base64": thumbnail_base64,
                    "content_type": "image/jpeg"
                }
            else:
                return {"success": False, "error": "Thumbnail generation failed"}

    except Exception as e:
        logger.error(f"Thumbnail generation failed: {e}")
        return {"success": False, "error": str(e)}


@router.post("/info")
async def get_info(video_url: str):
    """Get video metadata from URL."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_ext = Path(video_url.split("?")[0]).suffix or ".mp4"
            input_path = os.path.join(temp_dir, f"input{input_ext}")

            await download_file(video_url, input_path)
            info = get_video_info(input_path)

            return {
                "success": True,
                "info": info.model_dump()
            }

    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return {"success": False, "error": str(e)}
