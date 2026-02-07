"""GFP-GAN face restoration model."""

import logging
import time

import numpy as np
import torch
from PIL import Image

from api.core.abstractions import BaseRestorer, RestoreResult

logger = logging.getLogger(__name__)


class GFPGANRestorer(BaseRestorer):
    """GFP-GAN restorer for face enhancement with background upsampling."""

    def __init__(
        self,
        upscale: int = 2,
        cache_dir: str = "./model_cache",
    ):
        self._upscale = upscale
        self._cache_dir = cache_dir
        self._model = None

    @property
    def model_name(self) -> str:
        return "GFPGANv1.3"

    GFPGAN_URL = (
        "https://github.com/TencentARC/GFPGAN/releases/download/"
        "v1.3.0/GFPGANv1.3.pth"
    )

    def load_model(self) -> None:
        """Load GFP-GAN model with Real-ESRGAN background upsampler."""
        if self._model is not None:
            return

        import os
        from gfpgan import GFPGANer
        from realesrgan.utils import load_file_from_url

        os.makedirs(self._cache_dir, exist_ok=True)

        # Download GFP-GAN weights to persistent cache
        model_path = os.path.join(self._cache_dir, "GFPGANv1.3.pth")
        if not os.path.exists(model_path):
            model_path = load_file_from_url(
                self.GFPGAN_URL, model_dir=self._cache_dir,
            )

        # Symlink gfpgan/weights -> cache_dir so facexlib weights persist
        default_weights = os.path.join(os.getcwd(), "gfpgan", "weights")
        if not os.path.exists(default_weights):
            os.makedirs(os.path.dirname(default_weights), exist_ok=True)
            os.symlink(self._cache_dir, default_weights)

        # Skip heavy background upsampler on CPU to prevent OOM
        use_bg = torch.cuda.is_available()
        bg_upsampler = self._create_bg_upsampler() if use_bg else None
        if not use_bg:
            logger.info("CPU mode: skipping background upsampler")

        self._model = GFPGANer(
            model_path=model_path,
            upscale=self._upscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=bg_upsampler,
        )

        logger.info("GFP-GAN model loaded")

    def _create_bg_upsampler(self):
        """Create Real-ESRGAN background upsampler for non-face regions."""
        import os
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        from realesrgan.utils import load_file_from_url

        weights_url = (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.1.0/RealESRGAN_x4plus.pth"
        )

        model_path = os.path.join(self._cache_dir, "RealESRGAN_x4plus.pth")
        if not os.path.exists(model_path):
            os.makedirs(self._cache_dir, exist_ok=True)
            model_path = load_file_from_url(
                weights_url, model_dir=self._cache_dir,
            )

        rrdb_net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )

        half = torch.cuda.is_available()

        bg_upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=rrdb_net,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=half,
        )

        return bg_upsampler

    def unload_model(self) -> None:
        """Unload model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("GFP-GAN model unloaded")

    def restore(self, image: Image.Image) -> RestoreResult:
        """Restore image with face enhancement and background upscaling."""
        self.load_model()

        start_time = time.time()

        # Convert PIL to numpy BGR (OpenCV format)
        img_rgb = np.array(image)
        img_bgr = img_rgb[:, :, ::-1]

        # Run GFP-GAN enhancement
        _, _, output_bgr = self._model.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )

        # Convert back to PIL RGB
        output_rgb = output_bgr[:, :, ::-1]
        restored = Image.fromarray(output_rgb)

        processing_time = int((time.time() - start_time) * 1000)

        logger.info(
            f"GFP-GAN: {image.size} -> {restored.size} "
            f"in {processing_time}ms"
        )

        return RestoreResult(
            restored_image=restored,
            model_version="GFPGANv1.3",
            processing_time_ms=processing_time,
        )
