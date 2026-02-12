"""Album video generation service using FFmpeg.

Downloads album photos from Supabase storage, applies transitions
between them, overlays memories/reactions text, adds background music,
and produces an MP4 video.
"""

import logging
import os
import random
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Optional

import httpx
from supabase import Client

logger = logging.getLogger(__name__)

# Transitions supported by FFmpeg xfade filter
TRANSITIONS = [
    'fade', 'fadeblack', 'fadewhite', 'dissolve',
    'wipeleft', 'wiperight', 'slideleft', 'slideright',
    'circlecrop', 'circleopen', 'circleclose',
    'smoothleft', 'smoothright',
    'diagbl', 'diagbr',
    'zoomin',
]

TRANSITION_DURATION = 1.0  # seconds


class VideoService:
    """Generates album slideshow videos with transitions and optional music."""

    def __init__(self, client: Client):
        self._client = client

    async def generate(
        self,
        album_id: str,
        user_id: str,
        config: dict,
        on_progress: Callable[[str, int], None],
    ) -> dict:
        """Generate album video end-to-end.

        Args:
            album_id: The album to generate video for.
            user_id: Owner of the job.
            config: Video configuration from the mobile app.
            on_progress: Callback(step, percent) to report progress.

        Returns:
            dict with output_url, output_asset_id, duration_seconds.
        """
        photo_duration = config.get('photo_duration', 5)
        music_url = config.get('music_url')
        show_memories = config.get('show_memories', True)
        show_reactions = config.get('show_reactions', True)

        with tempfile.TemporaryDirectory(prefix='kizu-video-') as tmp:
            tmp_path = Path(tmp)

            # 1. Fetch album assets in order
            on_progress('downloading_photos', 5)
            assets = self._fetch_album_assets(album_id)
            if not assets:
                raise ValueError('Album has no photos')

            # 2. Download photos
            on_progress('downloading_photos', 10)
            image_paths = await self._download_assets(assets, tmp_path, on_progress)

            # 3. Download music if provided
            music_path = None
            if music_url:
                on_progress('downloading_photos', 30)
                music_path = await self._download_file(music_url, tmp_path / 'music.mp3')

            # 4. Fetch overlay text (memories & reactions)
            overlays = {}
            if show_memories or show_reactions:
                on_progress('rendering_overlays', 35)
                overlays = self._fetch_overlays(album_id, assets, show_memories, show_reactions)

            # 5. Generate video with FFmpeg
            on_progress('encoding_video', 40)
            output_path = tmp_path / 'output.mp4'
            duration = self._render_video(
                image_paths=image_paths,
                output_path=output_path,
                photo_duration=photo_duration,
                music_path=music_path,
                overlays=overlays,
                on_progress=on_progress,
            )

            # 6. Upload to Supabase storage
            on_progress('uploading', 85)
            storage_path = f'{user_id}/album_videos/{album_id}.mp4'
            with open(output_path, 'rb') as f:
                video_bytes = f.read()

            self._client.storage.from_('assets').upload(
                path=storage_path,
                file=video_bytes,
                file_options={'content-type': 'video/mp4', 'upsert': 'true'},
            )

            public_url = self._client.storage.from_('assets').get_public_url(storage_path)

            # 7. Create asset record
            on_progress('uploading', 95)
            asset_file_id = str(uuid.uuid4())
            asset_result = self._client.table('assets').insert({
                'user_id': user_id,
                'asset_file_id': asset_file_id,
                'path': storage_path,
                'web_uri': public_url,
                'mediaType': 'video',
                'media_type': 'video',
            }).execute()

            asset_id = asset_result.data[0]['id'] if asset_result.data else None

            # 8. Link video to album
            if asset_id:
                self._client.table('albums').update({
                    'video_asset_id': asset_id,
                }).eq('id', album_id).execute()

            on_progress('uploading', 100)

            return {
                'output_url': public_url,
                'output_asset_id': asset_id,
                'output_duration_seconds': duration,
            }

    def _fetch_album_assets(self, album_id: str) -> list[dict]:
        """Fetch album assets sorted by display order."""
        result = self._client.table('album_assets').select(
            'asset_id, display_order, assets(id, path, web_uri, thumbnail, media_type)'
        ).eq('album_id', album_id).order('display_order').execute()

        assets = []
        for row in (result.data or []):
            asset = row.get('assets')
            if asset and asset.get('media_type', 'photo') == 'photo':
                assets.append({
                    'id': asset['id'],
                    'url': asset.get('web_uri') or asset.get('path') or asset.get('thumbnail'),
                    'display_order': row.get('display_order', 0),
                })
        return assets

    async def _download_assets(
        self, assets: list[dict], tmp_path: Path, on_progress: Callable
    ) -> list[Path]:
        """Download all asset images to temp directory."""
        paths = []
        total = len(assets)
        async with httpx.AsyncClient(timeout=30.0) as http:
            for i, asset in enumerate(assets):
                url = asset['url']
                if not url.startswith('http'):
                    # Construct full URL from storage path
                    url = self._client.storage.from_('assets').get_public_url(url)

                ext = '.jpg'
                out_path = tmp_path / f'img_{i:04d}{ext}'
                resp = await http.get(url)
                resp.raise_for_status()
                out_path.write_bytes(resp.content)
                paths.append(out_path)

                pct = 10 + int(20 * (i + 1) / total)
                on_progress('downloading_photos', pct)

        return paths

    async def _download_file(self, url: str, dest: Path) -> Optional[Path]:
        """Download a single file."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as http:
                resp = await http.get(url)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                return dest
        except Exception as e:
            logger.warning(f'Failed to download music: {e}')
            return None

    def _fetch_overlays(
        self, album_id: str, assets: list[dict], show_memories: bool, show_reactions: bool
    ) -> dict[str, list[str]]:
        """Fetch memory/reaction text per asset for overlay.

        Returns: { asset_id: [line1, line2, ...] }
        """
        overlays: dict[str, list[str]] = {}
        asset_ids = [a['id'] for a in assets]

        if show_memories:
            result = self._client.table('memories').select(
                'asset_id, content_text, memory_type, external_author_name'
            ).in_('asset_id', asset_ids).eq('is_deleted', False).execute()
            for row in (result.data or []):
                aid = row['asset_id']
                text = row.get('content_text', '')
                if not text or row.get('memory_type') != 'text':
                    continue
                author = row.get('external_author_name', '')
                line = f'{author}: {text}' if author else text
                overlays.setdefault(aid, []).append(line)

        if show_reactions:
            result = self._client.table('reactions').select(
                'asset_id, emoji'
            ).in_('asset_id', asset_ids).execute()
            for row in (result.data or []):
                aid = row['asset_id']
                emoji = row.get('emoji', '')
                if emoji:
                    overlays.setdefault(aid, []).append(emoji)

        return overlays

    def _render_video(
        self,
        image_paths: list[Path],
        output_path: Path,
        photo_duration: float,
        music_path: Optional[Path],
        overlays: dict[str, list[str]],
        on_progress: Callable,
    ) -> float:
        """Render final video with FFmpeg using xfade transitions.

        Returns video duration in seconds.
        """
        if len(image_paths) == 0:
            raise ValueError('No images to render')

        if len(image_paths) == 1:
            return self._render_single_image(
                image_paths[0], output_path, photo_duration, music_path
            )

        # Build FFmpeg complex filter with xfade transitions
        n = len(image_paths)
        td = TRANSITION_DURATION
        inputs = []
        filter_parts = []

        # Input file arguments
        for path in image_paths:
            inputs.extend(['-loop', '1', '-t', str(photo_duration), '-i', str(path)])

        if music_path:
            inputs.extend(['-i', str(music_path)])

        # Scale all inputs to same size (1080p)
        for i in range(n):
            filter_parts.append(
                f'[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,'
                f'pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=30[v{i}]'
            )

        # Chain xfade transitions
        prev = 'v0'
        for i in range(1, n):
            transition = random.choice(TRANSITIONS)
            offset = photo_duration * i - td * i
            if offset < 0:
                offset = 0
            out_label = f'xf{i}'
            filter_parts.append(
                f'[{prev}][v{i}]xfade=transition={transition}:'
                f'duration={td}:offset={offset:.2f}[{out_label}]'
            )
            prev = out_label

        filter_complex = ';\n'.join(filter_parts)

        cmd = ['ffmpeg', '-y'] + inputs + [
            '-filter_complex', filter_complex,
            '-map', f'[{prev}]',
        ]

        if music_path:
            audio_input_idx = n
            cmd.extend(['-map', f'{audio_input_idx}:a', '-shortest'])

        cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '192k',
            str(output_path),
        ])

        logger.info(f'Running FFmpeg with {n} images, photo_duration={photo_duration}s')
        on_progress('encoding_video', 50)

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            logger.error(f'FFmpeg stderr: {result.stderr[-500:]}')
            raise RuntimeError(f'FFmpeg failed: {result.stderr[-200:]}')

        on_progress('encoding_video', 80)

        # Calculate duration
        total_duration = photo_duration * n - td * (n - 1)
        return max(total_duration, 0)

    def _render_single_image(
        self, image_path: Path, output_path: Path,
        duration: float, music_path: Optional[Path]
    ) -> float:
        """Render a video from a single image."""
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1', '-t', str(duration),
            '-i', str(image_path),
        ]
        if music_path:
            cmd.extend(['-i', str(music_path), '-shortest'])

        cmd.extend([
            '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,'
                    'pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,fps=30',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '192k',
            str(output_path),
        ])

        subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
        return duration
