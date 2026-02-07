"""Real-ESRGAN image restoration model."""

import logging
import time
from typing import Optional

import numpy as np
import torch
from PIL import Image

from api.core.abstractions import BaseRestorer, RestoreResult

logger = logging.getLogger(__name__)


class RealESRGANRestorer(BaseRestorer):
    """Real-ESRGAN restorer for general image upscaling and denoising."""

    def __init__(
        self,
        outscale: int = 2,
        cache_dir: str = "./model_cache",
    ):
        self._outscale = outscale
        self._cache_dir = cache_dir
        self._model = None

    @property
    def model_name(self) -> str:
        return "RealESRGAN_x4plus"

    WEIGHTS_URL = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.1.0/RealESRGAN_x4plus.pth"
    )

    def load_model(self) -> None:
        """Load Real-ESRGAN model."""
        if self._model is not None:
            return

        import os
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        from realesrgan.utils import load_file_from_url

        # Download weights if not cached
        model_path = os.path.join(self._cache_dir, "RealESRGAN_x4plus.pth")
        if not os.path.exists(model_path):
            os.makedirs(self._cache_dir, exist_ok=True)
            model_path = load_file_from_url(
                self.WEIGHTS_URL,
                model_dir=self._cache_dir,
            )

        rrdb_net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )

        has_cuda = torch.cuda.is_available()

        # Use tiling on CPU to prevent OOM (400px tiles)
        tile = 0 if has_cuda else 400

        self._model = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=rrdb_net,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=has_cuda,
        )

        logger.info("Real-ESRGAN model loaded")

    def unload_model(self) -> None:
        """Unload model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Real-ESRGAN model unloaded")

    def restore(self, image: Image.Image) -> RestoreResult:
        """Restore/upscale image using Real-ESRGAN."""
        self.load_model()

        start_time = time.time()

        # Convert PIL to numpy BGR (OpenCV format)
        img_rgb = np.array(image)
        img_bgr = img_rgb[:, :, ::-1]

        # Run enhancement
        output_bgr, _ = self._model.enhance(
            img_bgr, outscale=self._outscale
        )

        # Convert back to PIL RGB
        output_rgb = output_bgr[:, :, ::-1]
        restored = Image.fromarray(output_rgb)

        processing_time = int((time.time() - start_time) * 1000)

        logger.info(
            f"Real-ESRGAN: {image.size} -> {restored.size} "
            f"in {processing_time}ms"
        )

        return RestoreResult(
            restored_image=restored,
            model_version="RealESRGAN_x4plus",
            processing_time_ms=processing_time,
        )
