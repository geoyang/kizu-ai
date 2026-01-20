"""Florence-2 Vision-Language Model implementation."""

import logging
import time
from typing import Optional
from PIL import Image

from api.core.abstractions import BaseVLM, VLMResponse
from api.utils.device_utils import get_device

logger = logging.getLogger(__name__)


class Florence2Model(BaseVLM):
    """Florence-2 vision-language model - Microsoft's efficient multimodal model."""

    # Florence-2 task prompts for different detail levels
    DETAIL_TASKS = {
        'brief': '<CAPTION>',
        'medium': '<DETAILED_CAPTION>',
        'detailed': '<MORE_DETAILED_CAPTION>',
    }

    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-base",
        cache_dir: Optional[str] = None,
        quantization: Optional[str] = None
    ):
        self._model_id = model_name
        self._model_name = model_name.split('/')[-1]
        self._cache_dir = cache_dir
        self._quantization = quantization
        self._model = None
        self._processor = None
        self._device = None
        self._max_context = 1024

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def max_context_length(self) -> int:
        return self._max_context

    def load_model(self) -> None:
        """Load Florence-2 model with optional quantization."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        self._device = get_device()
        quant_str = f" with {self._quantization} quantization" if self._quantization else ""
        logger.info(f"Loading {self._model_name} on {self._device}{quant_str}")

        self._processor = AutoProcessor.from_pretrained(
            self._model_id,
            cache_dir=self._cache_dir,
            trust_remote_code=True
        )

        model_kwargs = {
            "low_cpu_mem_usage": True,
            "cache_dir": self._cache_dir,
            "trust_remote_code": True,
        }

        if self._quantization and self._device != "cpu":
            from transformers import BitsAndBytesConfig
            if self._quantization == "int8":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            elif self._quantization == "int4":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id, **model_kwargs
            )
            self._device = self._model.device
        else:
            dtype = torch.float16 if self._device != "cpu" else torch.float32
            model_kwargs["torch_dtype"] = dtype
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id, **model_kwargs
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

    def _run_inference(self, image: Image.Image, task: str, max_tokens: int) -> str:
        """Run Florence-2 inference with a task prompt."""
        import torch

        inputs = self._processor(
            text=task,
            images=image,
            return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_tokens,
                num_beams=3,
                do_sample=False
            )

        output_text = self._processor.batch_decode(
            output_ids, skip_special_tokens=False
        )[0]

        # Parse the response from Florence-2 format
        parsed = self._processor.post_process_generation(
            output_text, task=task, image_size=image.size
        )
        return parsed.get(task, output_text)

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> VLMResponse:
        """Generate response for image with prompt."""
        self.load_model()
        start_time = time.time()

        # Use detailed caption task for general prompts
        task = '<MORE_DETAILED_CAPTION>'
        result = self._run_inference(image, task, max_tokens)

        elapsed_ms = (time.time() - start_time) * 1000
        return VLMResponse(
            text=result,
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
        start_time = time.time()

        task = self.DETAIL_TASKS.get(detail_level, self.DETAIL_TASKS['medium'])
        result = self._run_inference(image, task, max_tokens=150)

        elapsed_ms = (time.time() - start_time) * 1000
        return VLMResponse(
            text=result,
            model_version=self._model_name,
            processing_time_ms=elapsed_ms
        )

    def answer_question(
        self,
        image: Image.Image,
        question: str
    ) -> VLMResponse:
        """Answer a question about the image using VQA task."""
        self.load_model()
        start_time = time.time()

        task = '<VQA>'
        # Florence-2 VQA format: <VQA> followed by question
        full_prompt = f"{task}{question}"
        result = self._run_inference(image, full_prompt, max_tokens=100)

        elapsed_ms = (time.time() - start_time) * 1000
        return VLMResponse(
            text=result,
            model_version=self._model_name,
            processing_time_ms=elapsed_ms
        )

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
