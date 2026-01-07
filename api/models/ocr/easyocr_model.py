"""EasyOCR text extraction model implementation."""

import logging
from typing import List, Optional
import numpy as np
from PIL import Image

from api.core.abstractions import (
    BaseOCR,
    OCRResult,
    TextRegion,
    BoundingBox
)

logger = logging.getLogger(__name__)


class EasyOCRModel(BaseOCR):
    """EasyOCR-based text extraction."""

    SUPPORTED_LANGUAGES = [
        'en', 'ch_sim', 'ch_tra', 'ja', 'ko', 'th', 'vi',
        'fr', 'de', 'es', 'it', 'pt', 'ru', 'ar', 'hi'
    ]

    def __init__(
        self,
        default_languages: List[str] = None,
        gpu: bool = False
    ):
        self._default_languages = default_languages or ['en']
        self._gpu = gpu
        self._reader = None
        self._loaded_languages = None
        self._model_name = "easyocr"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supported_languages(self) -> List[str]:
        return self.SUPPORTED_LANGUAGES

    def load_model(self, languages: Optional[List[str]] = None) -> None:
        """Load EasyOCR with specified languages."""
        import easyocr

        langs = languages or self._default_languages

        # Reload if languages changed
        if self._reader is not None and set(langs) == set(self._loaded_languages):
            return

        logger.info(f"Loading EasyOCR with languages: {langs}")

        self._reader = easyocr.Reader(
            langs,
            gpu=self._gpu,
            verbose=False
        )
        self._loaded_languages = langs

        logger.info(f"Loaded EasyOCR")

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self._reader is not None:
            del self._reader
            self._reader = None
            self._loaded_languages = None
            logger.info("Unloaded EasyOCR")

    def extract_text(
        self,
        image: Image.Image,
        languages: Optional[List[str]] = None
    ) -> OCRResult:
        """Extract text from an image."""
        self.load_model(languages)

        img_array = np.array(image.convert('RGB'))
        results = self._reader.readtext(img_array)

        regions = []
        full_text_parts = []

        for bbox, text, confidence in results:
            # bbox is [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]

            region = TextRegion(
                text=text,
                confidence=float(confidence),
                bounding_box=BoundingBox(
                    x=min(x_coords),
                    y=min(y_coords),
                    width=max(x_coords) - min(x_coords),
                    height=max(y_coords) - min(y_coords)
                )
            )
            regions.append(region)
            full_text_parts.append(text)

        return OCRResult(
            full_text=' '.join(full_text_parts),
            regions=regions,
            model_version=self._model_name,
            languages_detected=self._loaded_languages or []
        )

    def extract_text_batch(
        self,
        images: List[Image.Image],
        languages: Optional[List[str]] = None
    ) -> List[OCRResult]:
        """Extract text from multiple images."""
        return [self.extract_text(img, languages) for img in images]
