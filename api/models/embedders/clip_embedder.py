"""OpenCLIP embedding model implementation."""

import logging
from typing import List, Optional
import numpy as np
from PIL import Image

from api.core.abstractions import BaseEmbedder, EmbeddingResult
from api.utils.device_utils import get_device

logger = logging.getLogger(__name__)


class CLIPEmbedder(BaseEmbedder):
    """OpenCLIP-based image and text embedder."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        cache_dir: Optional[str] = None
    ):
        self._model_name = f"openclip-{model_name}-{pretrained}"
        self._clip_model_name = model_name
        self._pretrained = pretrained
        self._cache_dir = cache_dir
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = None
        self._dimension = 512 if "B-32" in model_name else 768

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def load_model(self) -> None:
        """Load OpenCLIP model."""
        if self._model is not None:
            return

        import open_clip
        import torch

        self._device = get_device()
        logger.info(f"Loading {self._model_name} on {self._device}")

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._clip_model_name,
            pretrained=self._pretrained,
            cache_dir=self._cache_dir
        )
        self._model = self._model.to(self._device)
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(self._clip_model_name)

        logger.info(f"Loaded {self._model_name}")

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            del self._preprocess
            del self._tokenizer
            self._model = None
            self._preprocess = None
            self._tokenizer = None

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Unloaded {self._model_name}")

    def embed_image(self, image: Image.Image) -> EmbeddingResult:
        """Generate embedding for a single image."""
        self.load_model()
        import torch

        image_input = self._preprocess(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            embedding = self._model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return EmbeddingResult(
            embedding=embedding.cpu().numpy().flatten(),
            model_version=self._model_name,
            dimension=self._dimension
        )

    def embed_images(self, images: List[Image.Image]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple images."""
        self.load_model()
        import torch

        processed = torch.stack([self._preprocess(img) for img in images])
        processed = processed.to(self._device)

        with torch.no_grad():
            embeddings = self._model.encode_image(processed)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        results = []
        for emb in embeddings.cpu().numpy():
            results.append(EmbeddingResult(
                embedding=emb,
                model_version=self._model_name,
                dimension=self._dimension
            ))
        return results

    def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for text query."""
        self.load_model()
        import torch

        tokens = self._tokenizer([text]).to(self._device)

        with torch.no_grad():
            embedding = self._model.encode_text(tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return EmbeddingResult(
            embedding=embedding.cpu().numpy().flatten(),
            model_version=self._model_name,
            dimension=self._dimension
        )

    def embed_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        self.load_model()
        import torch

        tokens = self._tokenizer(texts).to(self._device)

        with torch.no_grad():
            embeddings = self._model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        results = []
        for emb in embeddings.cpu().numpy():
            results.append(EmbeddingResult(
                embedding=emb,
                model_version=self._model_name,
                dimension=self._dimension
            ))
        return results
