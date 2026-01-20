"""BLIP-2 Vision-Language Model implementation."""

import logging
import time
from typing import Optional
from PIL import Image

from api.core.abstractions import BaseVLM, VLMResponse
from api.utils.device_utils import get_device

logger = logging.getLogger(__name__)


class BLIP2Model(BaseVLM):
    """BLIP-2 vision-language model - lightweight alternative to LLaVA."""

    # BLIP-2 works best with question format or no prompt for captioning
    DESCRIPTION_PROMPTS = {
        'brief': "Question: What is in this image? Answer:",
        'medium': "Question: Describe what you see in this image in detail. Answer:",
        'detailed': "Question: Provide a comprehensive description of everything in this image. Answer:",
    }

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        cache_dir: Optional[str] = None
    ):
        self._model_id = model_name
        self._model_name = model_name.split('/')[-1]
        self._cache_dir = cache_dir
        self._model = None
        self._processor = None
        self._device = None
        self._max_context = 512

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_context_length(self) -> int:
        return self._max_context

    def load_model(self) -> None:
        """Load BLIP-2 model."""
        if self._model is not None:
            return

        import torch
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        self._device = get_device()
        logger.info(f"Loading {self._model_name} on {self._device}")

        self._processor = Blip2Processor.from_pretrained(
            self._model_id,
            cache_dir=self._cache_dir
        )

        dtype = torch.float16 if self._device != "cpu" else torch.float32
        self._model = Blip2ForConditionalGeneration.from_pretrained(
            self._model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=self._cache_dir
        ).to(self._device)

        logger.info(f"Loaded {self._model_name}")

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Unloaded {self._model_name}")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> VLMResponse:
        """Generate response for image with prompt."""
        self.load_model()
        import torch

        start_time = time.time()

        inputs = self._processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0
            )

        response_text = self._processor.decode(
            output[0],
            skip_special_tokens=True
        ).strip()

        elapsed_ms = (time.time() - start_time) * 1000

        return VLMResponse(
            text=response_text,
            model_version=self._model_name,
            processing_time_ms=elapsed_ms
        )

    def describe_image(
        self,
        image: Image.Image,
        detail_level: str = "medium"
    ) -> VLMResponse:
        """Generate natural language description."""
        self.load_model()
        import torch

        start_time = time.time()

        # For brief, use image captioning (no prompt) which BLIP-2 excels at
        if detail_level == 'brief':
            inputs = self._processor(
                images=image,
                return_tensors="pt"
            ).to(self._device)
        else:
            prompt = self.DESCRIPTION_PROMPTS.get(
                detail_level, self.DESCRIPTION_PROMPTS['medium']
            )
            inputs = self._processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False
            )

        response_text = self._processor.decode(
            output[0],
            skip_special_tokens=True
        ).strip()

        elapsed_ms = (time.time() - start_time) * 1000

        return VLMResponse(
            text=response_text,
            model_version=self._model_name,
            processing_time_ms=elapsed_ms
        )

    def answer_question(
        self,
        image: Image.Image,
        question: str
    ) -> VLMResponse:
        """Answer a question about the image."""
        prompt = f"Question: {question} Answer:"
        return self.generate(image, prompt, max_tokens=100, temperature=0.3)

    def parse_query(self, query: str) -> dict:
        """Parse search query to extract intent."""
        query_lower = query.lower()
        return {
            "raw_query": query,
            "has_person_reference": any(
                w in query_lower for w in ['person', 'people', 'man', 'woman']
            ),
            "has_location_reference": any(
                w in query_lower for w in ['beach', 'mountain', 'city', 'park']
            ),
            "has_time_reference": any(
                w in query_lower for w in ['sunset', 'sunrise', 'night', 'day']
            ),
        }
