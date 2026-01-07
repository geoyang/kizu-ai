"""LLaVA Vision-Language Model implementation."""

import logging
import time
from typing import Optional
from PIL import Image

from api.core.abstractions import BaseVLM, VLMResponse
from api.utils.device_utils import get_device

logger = logging.getLogger(__name__)


class LLaVAModel(BaseVLM):
    """LLaVA-based vision-language model."""

    DESCRIPTION_PROMPTS = {
        'brief': "Describe this image in one sentence.",
        'medium': "Describe this image in 2-3 sentences, mentioning the main subjects and setting.",
        'detailed': "Provide a detailed description of this image including subjects, setting, colors, mood, and any notable details."
    }

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        cache_dir: Optional[str] = None
    ):
        self._model_id = model_name
        self._model_name = model_name.split('/')[-1]
        self._cache_dir = cache_dir
        self._model = None
        self._processor = None
        self._device = None
        self._max_context = 4096

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_context_length(self) -> int:
        return self._max_context

    def load_model(self) -> None:
        """Load LLaVA model."""
        if self._model is not None:
            return

        import torch
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        self._device = get_device()
        logger.info(f"Loading {self._model_name} on {self._device}")

        self._processor = LlavaNextProcessor.from_pretrained(
            self._model_id,
            cache_dir=self._cache_dir
        )

        dtype = torch.float16 if self._device != "cpu" else torch.float32
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
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
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> VLMResponse:
        """Generate response for image with prompt."""
        self.load_model()
        import torch

        start_time = time.time()

        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]

        text_prompt = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        inputs = self._processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0
            )

        response_text = self._processor.decode(
            output[0],
            skip_special_tokens=True
        )

        # Extract just the assistant's response
        if "[/INST]" in response_text:
            response_text = response_text.split("[/INST]")[-1].strip()

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
        prompt = self.DESCRIPTION_PROMPTS.get(detail_level, self.DESCRIPTION_PROMPTS['medium'])
        return self.generate(image, prompt, max_tokens=256)

    def answer_question(
        self,
        image: Image.Image,
        question: str
    ) -> VLMResponse:
        """Answer a question about the image."""
        return self.generate(image, question, max_tokens=256, temperature=0.3)

    def parse_query(self, query: str) -> dict:
        """Parse search query to extract intent."""
        # Simple keyword extraction - can be enhanced
        query_lower = query.lower()

        parsed = {
            "raw_query": query,
            "has_person_reference": any(w in query_lower for w in ['person', 'people', 'man', 'woman', 'child']),
            "has_location_reference": any(w in query_lower for w in ['beach', 'mountain', 'city', 'park', 'home']),
            "has_time_reference": any(w in query_lower for w in ['sunset', 'sunrise', 'night', 'day', 'morning']),
        }

        return parsed
