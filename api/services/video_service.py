"""Album video generation service using FFmpeg.

Downloads album photos from Supabase storage, applies transitions
between them, overlays memories/reactions text, adds background music,
and produces an MP4 video.
"""

import io
import logging
import os
import random
import subprocess
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Callable, Optional

import httpx
from PIL import Image, ImageDraw, ImageFont, ImageOps
from supabase import Client

logger = logging.getLogger(__name__)

# Output video dimensions
OUT_W, OUT_H = 1920, 1080
OUT_ASPECT = OUT_W / OUT_H

# Overlay-style transitions where the new image appears on top of the previous
TRANSITIONS = [
    'slideleft', 'slideright', 'slideup', 'slidedown',
    'smoothleft', 'smoothright', 'smoothup', 'smoothdown',
    'circlecrop', 'circleopen',
    'diagbl', 'diagbr', 'diagtl', 'diagtr',
]

TRANSITION_DURATION = 1.0  # seconds

# Ken Burns zoom factors (how much to scale beyond 1080p for pan room)
KB_FACTORS = {'none': 0.0, 'mild': 0.15, 'medium': 0.25}

# Max images per FFmpeg batch (keeps total inputs including overlay PNGs manageable)
MAX_BATCH_SIZE = 15


class VideoService:
    """Generates album slideshow videos with transitions and optional music."""

    def __init__(self, client: Client):
        self._client = client
        self._face_model = None

    def _get_face_model(self):
        """Lazy-load face detection model."""
        if self._face_model is None:
            from api.models.faces.insightface_model import InsightFaceModel
            self._face_model = InsightFaceModel()
        return self._face_model

    def _compute_face_crop(
        self, image_path: Path, target_w: int = OUT_W, target_h: int = OUT_H,
    ) -> tuple[int, int]:
        """Detect faces and compute crop offset that keeps faces visible.

        Returns (crop_x, crop_y) for FFmpeg crop=OUT_W:OUT_H:x:y after
        scaling the image up to cover target_w x target_h.
        """
        try:
            img = Image.open(image_path)
            w, h = img.size

            # Calculate scaled dimensions (same as force_original_aspect_ratio=increase)
            scale = max(target_w / w, target_h / h)
            scaled_w = round(w * scale)
            scaled_h = round(h * scale)

            # Detect faces on original image
            model = self._get_face_model()
            result = model.detect_faces(img, min_face_size=20)

            if not result.faces:
                # No faces: center crop
                return (scaled_w - OUT_W) // 2, (scaled_h - OUT_H) // 2

            # Compute bounding box around all faces (in original image coords)
            face_min_x = min(f.bounding_box.x for f in result.faces)
            face_min_y = min(f.bounding_box.y for f in result.faces)
            face_max_x = max(f.bounding_box.x + f.bounding_box.width for f in result.faces)
            face_max_y = max(f.bounding_box.y + f.bounding_box.height for f in result.faces)

            # Face center in original coords, then scale to output coords
            face_cx = (face_min_x + face_max_x) / 2 * scale
            face_cy = (face_min_y + face_max_y) / 2 * scale

            # Compute crop origin centered on faces, clamped to valid range
            crop_x = int(max(0, min(face_cx - OUT_W / 2, scaled_w - OUT_W)))
            crop_y = int(max(0, min(face_cy - OUT_H / 2, scaled_h - OUT_H)))

            return crop_x, crop_y

        except Exception as e:
            logger.warning(f'Face detection failed for {image_path}: {e}')
            # Fallback: center crop
            try:
                img = Image.open(image_path)
                scale = max(target_w / img.width, target_h / img.height)
                sw, sh = round(img.width * scale), round(img.height * scale)
                return (sw - OUT_W) // 2, (sh - OUT_H) // 2
            except Exception:
                return 0, 0

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
        ken_burns = config.get('ken_burns', 'mild')  # 'none', 'mild', 'medium'
        max_photos = config.get('max_photos')

        with tempfile.TemporaryDirectory(prefix='kizu-video-') as tmp:
            tmp_path = Path(tmp)

            # 1. Fetch album assets in order
            on_progress('downloading_photos', 5)
            assets = self._fetch_album_assets(album_id)
            if not assets:
                raise ValueError('Album has no photos')

            # Limit photo count if max_photos specified
            if max_photos and len(assets) > max_photos:
                assets = assets[:max_photos]

            # 2. Download photos
            on_progress('downloading_photos', 10)
            image_paths = await self._download_assets(assets, tmp_path, on_progress)

            # 3. Detect faces for smart cropping (scale larger for Ken Burns margin)
            on_progress('downloading_photos', 32)
            kb_factor = KB_FACTORS.get(ken_burns, 0.05)
            kb_w = round(OUT_W * (1 + kb_factor))
            kb_h = round(OUT_H * (1 + kb_factor))
            crop_offsets = []
            for path in image_paths:
                crop_offsets.append(self._compute_face_crop(path, kb_w, kb_h))

            # 3b. Download music if provided
            music_path = None
            if music_url:
                on_progress('downloading_photos', 35)
                music_path = await self._download_file(music_url, tmp_path / 'music.mp3')

            # 4. Fetch overlay data (memories & reactions)
            overlays = {}
            memory_cards: dict[str, list[Path]] = {}
            if show_memories or show_reactions:
                on_progress('rendering_overlays', 35)
                overlays = self._fetch_overlays(album_id, assets, show_memories, show_reactions)

            # 4b. Pre-render memory cards as PNGs (avatar + name + text)
            reaction_pngs: dict[str, list[Path]] = {}
            if overlays:
                on_progress('rendering_overlays', 37)
                memory_cards = await self._prepare_memory_cards(overlays, tmp_path)
                reaction_pngs = self._prepare_reaction_pngs(overlays, tmp_path)

            # 5. Generate video with FFmpeg
            on_progress('encoding_video', 40)
            output_path = tmp_path / 'output.mp4'
            asset_ids = [a['id'] for a in assets]
            duration = self._render_video(
                image_paths=image_paths,
                output_path=output_path,
                photo_duration=photo_duration,
                music_path=music_path,
                overlays=overlays,
                asset_ids=asset_ids,
                crop_offsets=crop_offsets,
                memory_cards=memory_cards,
                reaction_pngs=reaction_pngs,
                ken_burns=ken_burns,
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
        self, assets: list[dict], tmp_path: Path, on_progress: Callable,
        max_dimension: int = 2400,
    ) -> list[Path]:
        """Download all asset images and pre-resize to limit FFmpeg memory."""
        paths = []
        total = len(assets)
        async with httpx.AsyncClient(timeout=30.0) as http:
            for i, asset in enumerate(assets):
                url = asset['url']
                if not url.startswith('http'):
                    url = self._client.storage.from_('assets').get_public_url(url)

                out_path = tmp_path / f'img_{i:04d}.jpg'
                resp = await http.get(url)
                resp.raise_for_status()

                # Fix EXIF orientation and pre-resize to save FFmpeg memory
                img = Image.open(io.BytesIO(resp.content))
                img = ImageOps.exif_transpose(img)
                w, h = img.size
                if max(w, h) > max_dimension:
                    ratio = max_dimension / max(w, h)
                    img = img.resize(
                        (round(w * ratio), round(h * ratio)), Image.LANCZOS
                    )
                img.save(out_path, 'JPEG', quality=92)
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

    # Map reaction codes stored in DB to actual emoji characters
    EMOJI_MAP = {
        'thumbsup': '\U0001f44d',
        'heart': '\u2764\ufe0f',
        'laugh': '\U0001f602',
        'wow': '\U0001f62e',
        'sad': '\U0001f622',
        'angry': '\U0001f621',
        'crazy': '\U0001f92a',
        'kiss': '\U0001f618',
        'puke': '\U0001f92e',
        'wink': '\U0001f609',
        'cool': '\U0001f60e',
        'angel': '\U0001f607',
    }

    EMOJI_COLORS = {
        'thumbsup': '#FFD700', 'heart': '#FF0000', 'laugh': '#FFD700',
        'wow': '#FFD700', 'sad': '#4169E1', 'angry': '#FF4500',
        'crazy': '#FFD700', 'kiss': '#FF69B4', 'puke': '#32CD32',
        'wink': '#FFD700', 'cool': '#FFD700', 'angel': '#87CEEB',
    }

    def _fetch_overlays(
        self, album_id: str, assets: list[dict], show_memories: bool, show_reactions: bool
    ) -> dict:
        """Fetch memory/reaction data per asset for overlay.

        Returns: { asset_id: { 'memories': [{text, author_name, avatar_url}], 'reactions': [str] } }
        """
        overlays: dict[str, dict] = {}
        asset_ids = [a['id'] for a in assets]

        if show_memories:
            result = self._client.table('memories').select(
                'asset_id, content_text, memory_type, user_id'
            ).in_('asset_id', asset_ids).eq('is_deleted', False).execute()

            memory_rows = [
                r for r in (result.data or [])
                if r.get('content_text') and r.get('memory_type') == 'text'
            ]

            # Fetch profiles for memory authors (include email as fallback name)
            user_ids = list({r['user_id'] for r in memory_rows if r.get('user_id')})
            profiles = {}
            if user_ids:
                prof_result = self._client.table('profiles').select(
                    'id, full_name, avatar_url, email'
                ).in_('id', user_ids).execute()
                for p in (prof_result.data or []):
                    profiles[p['id']] = p
                logger.info(
                    f'Overlay profiles: {len(profiles)} found for {len(user_ids)} users: '
                    f'{[(p.get("full_name"), p.get("email"), bool(p.get("avatar_url"))) for p in profiles.values()]}'
                )

            for row in memory_rows:
                aid = row['asset_id']
                profile = profiles.get(row.get('user_id'), {})
                # Name fallback: full_name → email prefix → "Someone"
                name = profile.get('full_name')
                if not name:
                    email = profile.get('email', '')
                    name = email.split('@')[0].replace('.', ' ').title() if email else 'Someone'
                overlays.setdefault(aid, {'memories': [], 'reactions': []})
                overlays[aid]['memories'].append({
                    'text': row['content_text'],
                    'author_name': name,
                    'avatar_url': profile.get('avatar_url') or '',
                })

        if show_reactions:
            result = self._client.table('reactions').select(
                'asset_id, emoji'
            ).in_('asset_id', asset_ids).execute()
            for row in (result.data or []):
                aid = row['asset_id']
                code = row.get('emoji', '')
                if code:
                    emoji_char = self.EMOJI_MAP.get(code, code)
                    color = self.EMOJI_COLORS.get(code, 'white')
                    overlays.setdefault(aid, {'memories': [], 'reactions': []})
                    overlays[aid]['reactions'].append({
                        'emoji': emoji_char, 'color': color,
                    })

        return overlays

    @staticmethod
    def _escape_drawtext(text: str) -> str:
        """Escape text for FFmpeg drawtext filter."""
        # FFmpeg drawtext needs these chars escaped
        return text.replace('\\', '\\\\').replace("'", "'\\''").replace(':', '\\:').replace('%', '%%')

    def _render_memory_card(
        self, text: str, author_name: str, avatar_path: Optional[Path],
        save_path: Path, card_w: int = 1080,
    ) -> Path:
        """Render a memory card as a transparent PNG with avatar, name, and text."""
        font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
        font_path_reg = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

        try:
            name_font = ImageFont.truetype(font_path, 42)
            text_font = ImageFont.truetype(font_path_reg, 36)
            initial_font = ImageFont.truetype(font_path, 44)
        except Exception:
            name_font = ImageFont.load_default()
            text_font = name_font
            initial_font = name_font

        # Wrap and limit text
        wrapped = textwrap.fill(text, width=38)
        lines = wrapped.split('\n')[:4]

        avatar_size = 96
        pad = 24
        text_x = pad + avatar_size + 20
        name_h, line_h = 52, 44
        card_h = pad + name_h + len(lines) * line_h + pad
        card_h = max(card_h, avatar_size + pad * 2)

        card = Image.new('RGBA', (card_w, card_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(card)

        # Semi-transparent dark rounded background
        draw.rounded_rectangle(
            [0, 0, card_w - 1, card_h - 1], radius=16, fill=(0, 0, 0, 180)
        )

        # Avatar
        av_y = (card_h - avatar_size) // 2
        if avatar_path and avatar_path.exists():
            try:
                av = Image.open(avatar_path).convert('RGBA')
                av = av.resize((avatar_size, avatar_size), Image.LANCZOS)
                mask = Image.new('L', (avatar_size, avatar_size), 0)
                ImageDraw.Draw(mask).ellipse(
                    [0, 0, avatar_size - 1, avatar_size - 1], fill=255
                )
                card.paste(av, (pad, av_y), mask)
            except Exception:
                draw.ellipse(
                    [pad, av_y, pad + avatar_size, av_y + avatar_size],
                    fill=(100, 100, 100, 200),
                )
        else:
            # Placeholder circle with initial
            draw.ellipse(
                [pad, av_y, pad + avatar_size, av_y + avatar_size],
                fill=(100, 100, 100, 200),
            )
            if author_name:
                letter = author_name[0].upper()
                bbox = draw.textbbox((0, 0), letter, font=initial_font)
                lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                draw.text(
                    (pad + (avatar_size - lw) // 2, av_y + (avatar_size - lh) // 2),
                    letter, fill=(255, 255, 255, 220), font=initial_font,
                )

        # Author name (gold/yellow)
        ty = pad
        draw.text((text_x, ty), author_name, fill=(255, 220, 100, 255), font=name_font)
        ty += name_h

        # Memory text (white)
        for line in lines:
            draw.text((text_x, ty), line, fill=(255, 255, 255, 240), font=text_font)
            ty += line_h

        card.save(save_path)
        return save_path

    def _render_reaction_png(
        self, emoji_char: str, color: str, save_path: Path, size: int = 80,
    ) -> Path:
        """Render a single emoji reaction as a transparent PNG.

        NotoColorEmoji only works at size 109 (native CBDT bitmap size),
        so we render at 109 then scale down to the target size.
        """
        emoji_font_path = '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf'
        native_size = 109  # only size NotoColorEmoji supports in Pillow

        rendered = False
        try:
            font = ImageFont.truetype(emoji_font_path, native_size)
            # Render at native bitmap size (136x128 glyph)
            canvas = Image.new('RGBA', (150, 150), (0, 0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((7, 7), emoji_char, font=font, embedded_color=True)
            # Crop to content
            bbox = canvas.getbbox()
            if bbox:
                canvas = canvas.crop(bbox)
            # Scale to target size
            canvas = canvas.resize((size, size), Image.LANCZOS)
            canvas.save(save_path)
            rendered = True
        except Exception as e:
            logger.warning(f'NotoColorEmoji render failed for {emoji_char}: {e}')

        if not rendered:
            # Fallback: colored circle
            card = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(card)
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            draw.ellipse([8, 8, size - 8, size - 8], fill=(r, g, b, 220))
            card.save(save_path)

        return save_path

    async def _prepare_memory_cards(
        self, overlays: dict, tmp_path: Path,
    ) -> dict[str, list[Path]]:
        """Pre-render memory card PNGs for all assets.

        Downloads avatars, renders cards with Pillow.
        Returns: {asset_id: [card_path, ...]}
        """
        # Collect unique avatar URLs
        unique_urls: dict[str, Optional[Path]] = {}
        for data in overlays.values():
            for mem in data.get('memories', []):
                if isinstance(mem, dict) and mem.get('avatar_url'):
                    unique_urls.setdefault(mem['avatar_url'], None)

        # Download avatars
        async with httpx.AsyncClient(timeout=10.0) as http:
            for i, url in enumerate(unique_urls):
                try:
                    full_url = url
                    if not full_url.startswith('http'):
                        full_url = self._client.storage.from_('avatars').get_public_url(url)
                    av_path = tmp_path / f'avatar_{i}.jpg'
                    resp = await http.get(full_url)
                    resp.raise_for_status()
                    av_path.write_bytes(resp.content)
                    unique_urls[url] = av_path
                except Exception as e:
                    logger.warning(f'Avatar download failed for {url}: {e}')

        # Render cards
        cards: dict[str, list[Path]] = {}
        card_idx = 0
        for aid, data in overlays.items():
            aid_cards = []
            for mem in data.get('memories', [])[:3]:
                if not isinstance(mem, dict):
                    continue
                av_path = unique_urls.get(mem.get('avatar_url', ''))
                card_path = tmp_path / f'card_{card_idx}.png'
                self._render_memory_card(
                    text=mem.get('text', ''),
                    author_name=mem.get('author_name', 'Someone'),
                    avatar_path=av_path,
                    save_path=card_path,
                )
                aid_cards.append(card_path)
                card_idx += 1
            if aid_cards:
                cards[aid] = aid_cards

        return cards

    def _prepare_reaction_pngs(
        self, overlays: dict, tmp_path: Path,
    ) -> dict[str, list[Path]]:
        """Pre-render reaction emoji PNGs for all assets.

        Returns: {asset_id: [emoji_png_path, ...]}
        """
        pngs: dict[str, list[Path]] = {}
        png_idx = 0
        for aid, data in overlays.items():
            reactions = data.get('reactions', [])
            if not reactions:
                continue
            aid_pngs = []
            for reaction in reactions[:6]:
                if isinstance(reaction, dict):
                    emoji_char = reaction.get('emoji', '')
                    color = reaction.get('color', '#FFFFFF')
                else:
                    emoji_char = reaction
                    color = '#FFFFFF'
                png_path = tmp_path / f'reaction_{png_idx}.png'
                self._render_reaction_png(emoji_char, color, png_path)
                aid_pngs.append(png_path)
                png_idx += 1
            if aid_pngs:
                pngs[aid] = aid_pngs
        return pngs

    def _build_overlay_filters(
        self, idx: int, asset_id: str, overlays: dict,
        card_input_map: dict[str, list[int]],
        reaction_input_map: dict[str, list[int]],
        photo_duration: float, label_in: str,
    ) -> tuple[list[str], str]:
        """Build overlay filters for memory cards and reaction emojis (both pre-rendered PNGs).

        Both types float from bottom to top with horizontal drift.
        Returns (filter_parts, final_label).
        """
        card_indices = card_input_map.get(asset_id, [])
        reaction_indices = reaction_input_map.get(asset_id, [])

        if not card_indices and not reaction_indices:
            return [], label_in

        parts = []
        current = label_in

        # Memory cards: overlay PNGs floating slowly with wide horizontal drift
        card_w = 1080  # must match _render_memory_card card_w
        for j, card_idx in enumerate(card_indices[:3]):
            faded = f'cf{idx}_{j}'
            out = f'mc{idx}_{j}'
            delay = j * 2.0
            rise_time = max(photo_duration * 2.0, 3.0)  # slow float
            # Center the card with room to drift 30-50% of screen width
            max_drift = OUT_W - card_w  # max x before card goes offscreen
            mid_x = max_drift // 2
            drift_amp = max(max_drift // 2, 1)  # full usable range / 2
            drift_speed = 0.4 + j * 0.15  # slow sinusoidal
            fade_start = max(delay + 1.0, photo_duration - 2.0)

            # Ensure RGBA format and apply fade-out on alpha
            parts.append(
                f'[{card_idx}:v]format=rgba,'
                f'fade=t=out:st={fade_start:.1f}:d=1.5:alpha=1[{faded}]'
            )

            # Overlay: x drifts across full available width, y floats up
            # Start at 20% from bottom (y = 0.8*H), float upward
            start_y = 0.8
            parts.append(
                f'[{current}][{faded}]overlay='
                f'x={mid_x}+{drift_amp}*sin(t*{drift_speed:.1f}+{j * 2}):'
                f'y=if(lt(t\\,{delay:.1f})\\,{start_y}*H\\,'
                f'{start_y}*H-({start_y}*H+h)*(t-{delay:.1f})/{rise_time:.1f}):'
                f'shortest=1[{out}]'
            )
            current = out

        # Reaction emojis: pre-rendered PNGs floating with drift
        for j, rxn_idx in enumerate(reaction_indices[:6]):
            faded = f'rf{idx}_{j}'
            out = f'emo{idx}_{j}'
            delay = j * 0.6
            rise_time = max(photo_duration * 1.5, 2.0)
            base_x = 80 + (j * 170) % (OUT_W - 200)
            drift_amp = 25
            drift_speed = 1.0 + j * 0.4
            fade_start = max(delay + 0.5, photo_duration - 1.5)

            # Ensure RGBA and apply fade-out
            parts.append(
                f'[{rxn_idx}:v]format=rgba,'
                f'fade=t=out:st={fade_start:.1f}:d=1.0:alpha=1[{faded}]'
            )

            # Overlay with animated y (float up) + x drift (sin wave)
            parts.append(
                f'[{current}][{faded}]overlay='
                f'x={base_x}+{drift_amp}*sin(t*{drift_speed:.1f}):'
                f'y=if(lt(t\\,{delay:.1f})\\,H\\,'
                f'H-(H+80)*(t-{delay:.1f})/{rise_time:.1f}):'
                f'shortest=1[{out}]'
            )
            current = out

        return parts, current

    def _render_video(
        self,
        image_paths: list[Path],
        output_path: Path,
        photo_duration: float,
        music_path: Optional[Path],
        overlays: dict,
        asset_ids: list[str],
        crop_offsets: list[tuple[int, int]],
        memory_cards: dict[str, list[Path]],
        reaction_pngs: dict[str, list[Path]] | None = None,
        ken_burns: str = 'mild',
        on_progress: Callable = lambda s, p: None,
    ) -> float:
        """Render final video with FFmpeg using xfade transitions.

        For large albums (> MAX_BATCH_SIZE), splits into batches, renders
        each separately, then concatenates to avoid FFmpeg OOM.

        Returns video duration in seconds.
        """
        if len(image_paths) == 0:
            raise ValueError('No images to render')

        if len(image_paths) == 1:
            cx, cy = crop_offsets[0] if crop_offsets else (0, 0)
            return self._render_single_image(
                image_paths[0], output_path, photo_duration, music_path, cx, cy
            )

        n = len(image_paths)

        if n > MAX_BATCH_SIZE:
            return self._render_video_batched(
                image_paths=image_paths,
                output_path=output_path,
                photo_duration=photo_duration,
                music_path=music_path,
                overlays=overlays,
                asset_ids=asset_ids,
                crop_offsets=crop_offsets,
                memory_cards=memory_cards,
                reaction_pngs=reaction_pngs,
                ken_burns=ken_burns,
                on_progress=on_progress,
            )

        return self._render_video_batch(
            image_paths=image_paths,
            output_path=output_path,
            photo_duration=photo_duration,
            music_path=None,  # music added during concat or single-batch
            overlays=overlays,
            asset_ids=asset_ids,
            crop_offsets=crop_offsets,
            memory_cards=memory_cards,
            reaction_pngs=reaction_pngs,
            ken_burns=ken_burns,
            on_progress=on_progress,
            music_for_single=music_path,
        )

    def _render_video_batched(
        self,
        image_paths: list[Path],
        output_path: Path,
        photo_duration: float,
        music_path: Optional[Path],
        overlays: dict,
        asset_ids: list[str],
        crop_offsets: list[tuple[int, int]],
        memory_cards: dict[str, list[Path]],
        reaction_pngs: dict[str, list[Path]] | None = None,
        ken_burns: str = 'mild',
        on_progress: Callable = lambda s, p: None,
    ) -> float:
        """Render large album as individual clips then concatenate.

        Each image becomes a short video clip with its overlays (max ~10
        FFmpeg inputs each), avoiding the complex multi-input filter graphs
        that cause segfaults. Clips get fade-in/out for smooth transitions.
        """
        n = len(image_paths)
        logger.info(f'Large album ({n} images): rendering individual clips + concat')

        tmp_dir = output_path.parent
        clip_paths: list[Path] = []
        kb_factor = KB_FACTORS.get(ken_burns, 0.0)
        kb_w = round(OUT_W * (1 + kb_factor))
        kb_h = round(OUT_H * (1 + kb_factor))
        fade_dur = min(0.5, photo_duration / 4)

        for i in range(n):
            clip_path = tmp_dir / f'clip_{i:04d}.mp4'
            cx, cy = crop_offsets[i] if i < len(crop_offsets) else (0, 0)
            aid = asset_ids[i] if i < len(asset_ids) else ''

            inputs = ['-loop', '1', '-t', str(photo_duration), '-i', str(image_paths[i])]
            filter_parts = []

            # Scale + crop (with Ken Burns if enabled)
            if kb_factor > 0:
                margin_x = max(kb_w - OUT_W, 0)
                margin_y = max(kb_h - OUT_H, 0)
                end_x = max(0, min(cx, margin_x))
                end_y = max(0, min(cy, margin_y))
                start_x = margin_x - end_x
                start_y = margin_y - end_y
                filter_parts.append(
                    f'[0:v]scale={kb_w}:{kb_h}:'
                    f'force_original_aspect_ratio=increase,'
                    f'crop={OUT_W}:{OUT_H}:'
                    f'{start_x}+({end_x}-{start_x})*t/{photo_duration:.1f}:'
                    f'{start_y}+({end_y}-{start_y})*t/{photo_duration:.1f},'
                    f'setsar=1,fps=30[base]'
                )
            else:
                filter_parts.append(
                    f'[0:v]scale={OUT_W}:{OUT_H}:'
                    f'force_original_aspect_ratio=increase,'
                    f'crop={OUT_W}:{OUT_H}:{cx}:{cy},'
                    f'setsar=1,fps=30[base]'
                )

            current = 'base'

            # Add overlay PNGs for this image's memory cards
            next_idx = 1
            aid_cards = memory_cards.get(aid, [])
            for j, card_path in enumerate(aid_cards[:3]):
                inputs.extend(['-loop', '1', '-t', str(photo_duration), '-i', str(card_path)])
                faded = f'cf{j}'
                out = f'mc{j}'
                delay = j * 2.0
                rise_time = max(photo_duration * 2.0, 3.0)
                card_w = 1080
                max_drift = OUT_W - card_w
                mid_x = max_drift // 2
                drift_amp = max(max_drift // 2, 1)
                drift_speed = 0.4 + j * 0.15
                fade_start = max(delay + 1.0, photo_duration - 2.0)
                start_y = 0.8

                filter_parts.append(
                    f'[{next_idx}:v]format=rgba,'
                    f'fade=t=out:st={fade_start:.1f}:d=1.5:alpha=1[{faded}]'
                )
                filter_parts.append(
                    f'[{current}][{faded}]overlay='
                    f'x={mid_x}+{drift_amp}*sin(t*{drift_speed:.1f}+{j * 2}):'
                    f'y=if(lt(t\\,{delay:.1f})\\,{start_y}*H\\,'
                    f'{start_y}*H-({start_y}*H+h)*(t-{delay:.1f})/{rise_time:.1f}):'
                    f'eof_action=pass[{out}]'
                )
                current = out
                next_idx += 1

            # Add overlay PNGs for this image's reactions
            if reaction_pngs:
                aid_rxns = reaction_pngs.get(aid, [])
                for j, rxn_path in enumerate(aid_rxns[:6]):
                    inputs.extend(['-loop', '1', '-t', str(photo_duration), '-i', str(rxn_path)])
                    faded = f'rf{j}'
                    out = f'emo{j}'
                    delay = j * 0.6
                    rise_time = max(photo_duration * 1.5, 2.0)
                    base_x = 80 + (j * 170) % (OUT_W - 200)
                    drift_amp = 25
                    drift_speed = 1.0 + j * 0.4
                    fade_start = max(delay + 0.5, photo_duration - 1.5)

                    filter_parts.append(
                        f'[{next_idx}:v]format=rgba,'
                        f'fade=t=out:st={fade_start:.1f}:d=1.0:alpha=1[{faded}]'
                    )
                    filter_parts.append(
                        f'[{current}][{faded}]overlay='
                        f'x={base_x}+{drift_amp}*sin(t*{drift_speed:.1f}):'
                        f'y=if(lt(t\\,{delay:.1f})\\,H\\,'
                        f'H-(H+80)*(t-{delay:.1f})/{rise_time:.1f}):'
                        f'eof_action=pass[{out}]'
                    )
                    current = out
                    next_idx += 1

            # Add fade-in/out for smooth transitions between clips
            fade_out_start = photo_duration - fade_dur
            filter_parts.append(
                f'[{current}]fade=t=in:st=0:d={fade_dur:.2f},'
                f'fade=t=out:st={fade_out_start:.2f}:d={fade_dur:.2f}[out]'
            )

            filter_complex = ';\n'.join(filter_parts)
            filter_script = tmp_dir / f'clip_{i:04d}.filter'
            filter_script.write_text(filter_complex)

            cmd = ['ffmpeg', '-y'] + inputs + [
                '-filter_complex_script', str(filter_script),
                '-map', '[out]',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '28',
                '-pix_fmt', 'yuv420p',
                str(clip_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.error(f'Clip {i} FFmpeg stderr:\n{result.stderr}')
                raise RuntimeError(
                    f'FFmpeg clip {i} failed (rc={result.returncode}): '
                    f'{result.stderr[-300:]}'
                )

            clip_paths.append(clip_path)

            pct = 40 + int(35 * (i + 1) / n)
            on_progress('encoding_video', pct)

        # Concatenate all clips
        on_progress('encoding_video', 78)
        concat_list = tmp_dir / 'concat.txt'
        concat_list.write_text(
            '\n'.join(f"file '{p}'" for p in clip_paths)
        )

        concat_cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', str(concat_list),
        ]

        if music_path:
            concat_cmd.extend(['-i', str(music_path), '-shortest'])
            concat_cmd.extend([
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '128k',
                str(output_path),
            ])
        else:
            concat_cmd.extend([
                '-c', 'copy',
                str(output_path),
            ])

        logger.info(f'Concatenating {len(clip_paths)} individual clips')
        result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error(f'Concat FFmpeg stderr:\n{result.stderr}')
            raise RuntimeError(f'FFmpeg concat failed (rc={result.returncode})')

        total_duration = photo_duration * n
        return total_duration

    def _render_video_batch(
        self,
        image_paths: list[Path],
        output_path: Path,
        photo_duration: float,
        music_path: Optional[Path],
        overlays: dict,
        asset_ids: list[str],
        crop_offsets: list[tuple[int, int]],
        memory_cards: dict[str, list[Path]],
        reaction_pngs: dict[str, list[Path]] | None = None,
        ken_burns: str = 'mild',
        on_progress: Callable = lambda s, p: None,
        music_for_single: Optional[Path] = None,
    ) -> float:
        """Render a single batch of images into a video clip.

        Returns video duration in seconds.
        """
        n = len(image_paths)
        if n == 0:
            raise ValueError('No images to render')

        # For single-batch (non-batched) path, use music directly
        effective_music = music_for_single or music_path

        if n == 1:
            cx, cy = crop_offsets[0] if crop_offsets else (0, 0)
            return self._render_single_image(
                image_paths[0], output_path, photo_duration, effective_music, cx, cy
            )

        # Build FFmpeg complex filter with xfade transitions
        td = TRANSITION_DURATION
        inputs = []
        filter_parts = []

        # Input file arguments: images
        for path in image_paths:
            inputs.extend(['-loop', '1', '-t', str(photo_duration), '-i', str(path)])

        # Music input (only for single-batch path; batched adds music at concat)
        if effective_music:
            inputs.extend(['-i', str(effective_music)])

        # Memory card PNG inputs (after images + music)
        card_input_map: dict[str, list[int]] = {}
        next_idx = n + (1 if effective_music else 0)
        for aid in asset_ids:
            aid_cards = memory_cards.get(aid, [])
            if aid_cards:
                indices = []
                for card_path in aid_cards:
                    inputs.extend([
                        '-loop', '1', '-t', str(photo_duration),
                        '-i', str(card_path),
                    ])
                    indices.append(next_idx)
                    next_idx += 1
                card_input_map[aid] = indices

        # Reaction emoji PNG inputs
        reaction_input_map: dict[str, list[int]] = {}
        if reaction_pngs:
            for aid in asset_ids:
                aid_rxns = reaction_pngs.get(aid, [])
                if aid_rxns:
                    indices = []
                    for rxn_path in aid_rxns:
                        inputs.extend([
                            '-loop', '1', '-t', str(photo_duration),
                            '-i', str(rxn_path),
                        ])
                        indices.append(next_idx)
                        next_idx += 1
                    reaction_input_map[aid] = indices

        # Scale all inputs with face-aware crop + optional Ken Burns pan
        kb_factor = KB_FACTORS.get(ken_burns, 0.0)
        kb_w = round(OUT_W * (1 + kb_factor))
        kb_h = round(OUT_H * (1 + kb_factor))

        for i in range(n):
            cx, cy = crop_offsets[i] if i < len(crop_offsets) else (0, 0)
            scale_label = f'sc{i}'

            if kb_factor > 0:
                # Ken Burns: scale larger, animated crop that pans toward faces
                margin_x = max(kb_w - OUT_W, 0)
                margin_y = max(kb_h - OUT_H, 0)

                # End on the face-centered crop position
                end_x = max(0, min(cx, margin_x))
                end_y = max(0, min(cy, margin_y))

                # Start from the opposite side so the pan moves toward faces
                start_x = margin_x - end_x
                start_y = margin_y - end_y

                filter_parts.append(
                    f'[{i}:v]scale={kb_w}:{kb_h}:'
                    f'force_original_aspect_ratio=increase,'
                    f'crop={OUT_W}:{OUT_H}:'
                    f'{start_x}+({end_x}-{start_x})*t/{photo_duration:.1f}:'
                    f'{start_y}+({end_y}-{start_y})*t/{photo_duration:.1f},'
                    f'setsar=1,fps=30[{scale_label}]'
                )
            else:
                # No Ken Burns: static crop
                filter_parts.append(
                    f'[{i}:v]scale={OUT_W}:{OUT_H}:'
                    f'force_original_aspect_ratio=increase,'
                    f'crop={OUT_W}:{OUT_H}:{cx}:{cy},'
                    f'setsar=1,fps=30[{scale_label}]'
                )

            # Add memory card overlays and reaction overlays for this image
            aid = asset_ids[i] if i < len(asset_ids) else ''
            overlay_parts, final_label = self._build_overlay_filters(
                i, aid, overlays, card_input_map, reaction_input_map,
                photo_duration, scale_label,
            )
            if overlay_parts:
                filter_parts.extend(overlay_parts)
                # Rename final label to expected v{i}
                filter_parts.append(f'[{final_label}]null[v{i}]')
            else:
                filter_parts.append(f'[{scale_label}]null[v{i}]')

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

        # Write filter to a script file (avoids CLI argument parsing issues)
        filter_script = output_path.with_suffix('.filter')
        filter_script.write_text(filter_complex)
        logger.info(f'Filter complex ({len(filter_parts)} parts):\n{filter_complex}')

        cmd = ['ffmpeg', '-y'] + inputs + [
            '-filter_complex_script', str(filter_script),
            '-map', f'[{prev}]',
        ]

        if effective_music:
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
            logger.error(f'FFmpeg returncode: {result.returncode}')
            logger.error(f'FFmpeg stderr:\n{result.stderr}')
            # Extract actual error lines (not progress) for the error message
            error_lines = [
                l for l in result.stderr.splitlines()
                if any(k in l.lower() for k in [
                    'error', 'invalid', 'no such', 'not found',
                    'failed', 'unable', 'undefined', 'unknown',
                ])
            ]
            if error_lines:
                err_msg = '\n'.join(error_lines[-5:])
            elif result.returncode < 0:
                import signal as _sig
                try:
                    sig_name = _sig.Signals(-result.returncode).name
                except (ValueError, AttributeError):
                    sig_name = str(-result.returncode)
                err_msg = f'FFmpeg killed by signal {sig_name} (likely OOM)'
            else:
                err_msg = result.stderr[-500:]
            raise RuntimeError(f'FFmpeg failed (rc={result.returncode}): {err_msg}')

        on_progress('encoding_video', 80)

        # Calculate duration
        total_duration = photo_duration * n - td * (n - 1)
        return max(total_duration, 0)

    def _render_single_image(
        self, image_path: Path, output_path: Path,
        duration: float, music_path: Optional[Path],
        crop_x: int = 0, crop_y: int = 0,
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
            '-vf', f'scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,'
                    f'crop={OUT_W}:{OUT_H}:{crop_x}:{crop_y},fps=30',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '192k',
            str(output_path),
        ])

        subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
        return duration
