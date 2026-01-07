"""Image loading and processing utilities."""

import logging
import base64
from typing import Optional, Tuple, Union
from pathlib import Path
from io import BytesIO
from PIL import Image
import httpx

logger = logging.getLogger(__name__)


def load_image(source: Union[str, bytes, Path], is_base64: bool = False) -> Image.Image:
    """
    Load an image from various sources.

    Args:
        source: File path, URL, base64 string, or bytes
        is_base64: If True, treat source as base64-encoded string

    Returns:
        PIL Image in RGB format
    """
    if isinstance(source, bytes):
        img = Image.open(BytesIO(source))
    elif isinstance(source, (str, Path)):
        source_str = str(source)
        if is_base64:
            # Handle data URL format: data:image/jpeg;base64,/9j/4AAQ...
            if source_str.startswith('data:'):
                # Extract the base64 part after the comma
                source_str = source_str.split(',', 1)[1]
            image_bytes = base64.b64decode(source_str)
            img = Image.open(BytesIO(image_bytes))
        elif source_str.startswith(('http://', 'https://')):
            img = load_image_from_url(source_str)
        else:
            img = Image.open(source_str)
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    return img


def load_image_from_url(url: str, timeout: float = 30.0) -> Image.Image:
    """Load an image from a URL."""
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()

    # Check content type - skip videos and non-images
    content_type = response.headers.get('content-type', '').lower()
    if content_type.startswith('video/'):
        raise ValueError(f"URL is a video, not an image: {content_type}")
    if not content_type.startswith('image/') and 'octet-stream' not in content_type:
        logger.warning(f"Unexpected content type: {content_type}, attempting to load anyway")

    try:
        return Image.open(BytesIO(response.content))
    except Exception as e:
        raise ValueError(f"Cannot load image from URL: {e}")


def resize_image(
    image: Image.Image,
    max_size: int = 1024,
    min_size: Optional[int] = None
) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.

    Args:
        image: PIL Image
        max_size: Maximum dimension (width or height)
        min_size: Optional minimum dimension
    """
    width, height = image.size

    # Check if resize needed
    if width <= max_size and height <= max_size:
        if min_size is None or (width >= min_size and height >= min_size):
            return image

    # Calculate new dimensions
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def crop_region(
    image: Image.Image,
    x: float,
    y: float,
    width: float,
    height: float,
    padding: float = 0.1
) -> Image.Image:
    """
    Crop a region from an image with optional padding.

    Args:
        image: Source image
        x, y, width, height: Region coordinates
        padding: Padding ratio (0.1 = 10% padding)
    """
    img_width, img_height = image.size

    # Add padding
    pad_w = width * padding
    pad_h = height * padding

    left = max(0, x - pad_w)
    top = max(0, y - pad_h)
    right = min(img_width, x + width + pad_w)
    bottom = min(img_height, y + height + pad_h)

    return image.crop((int(left), int(top), int(right), int(bottom)))


def get_image_hash(image: Image.Image) -> str:
    """Generate a perceptual hash for an image."""
    import imagehash
    return str(imagehash.phash(image))


def get_image_info(image: Image.Image) -> dict:
    """Get basic image information."""
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
    }
