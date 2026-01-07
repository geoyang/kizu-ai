"""YOLOv8 object detection model implementation."""

import logging
from typing import List, Optional
from PIL import Image

from api.core.abstractions import (
    BaseDetector,
    DetectionResult,
    DetectedObject,
    BoundingBox
)
from api.utils.device_utils import get_device

logger = logging.getLogger(__name__)

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class YOLODetector(BaseDetector):
    """YOLOv8-based object detector."""

    def __init__(
        self,
        model_size: str = "m",  # n, s, m, l, x
        cache_dir: Optional[str] = None
    ):
        self._model_size = model_size
        self._model_name = f"yolov8-{model_size}"
        self._cache_dir = cache_dir
        self._model = None
        self._device = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supported_classes(self) -> List[str]:
        return COCO_CLASSES

    def load_model(self) -> None:
        """Load YOLOv8 model."""
        if self._model is not None:
            return

        from ultralytics import YOLO

        self._device = get_device()
        model_path = f"yolov8{self._model_size}.pt"

        logger.info(f"Loading {self._model_name} on {self._device}")
        self._model = YOLO(model_path)
        logger.info(f"Loaded {self._model_name}")

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info(f"Unloaded {self._model_name}")

    def detect(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.5,
        classes: Optional[List[str]] = None
    ) -> DetectionResult:
        """Detect objects in an image."""
        self.load_model()

        class_filter = None
        if classes:
            class_filter = [
                i for i, name in enumerate(COCO_CLASSES)
                if name in classes
            ]

        results = self._model(
            image,
            conf=confidence_threshold,
            classes=class_filter,
            device=self._device,
            verbose=False
        )

        detected = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                detected.append(DetectedObject(
                    class_name=COCO_CLASSES[int(box.cls)],
                    class_id=int(box.cls),
                    confidence=float(box.conf),
                    bounding_box=BoundingBox(
                        x=float(x1),
                        y=float(y1),
                        width=float(x2 - x1),
                        height=float(y2 - y1)
                    )
                ))

        return DetectionResult(
            objects=detected,
            model_version=self._model_name,
            image_width=image.width,
            image_height=image.height
        )

    def detect_batch(
        self,
        images: List[Image.Image],
        confidence_threshold: float = 0.5,
        classes: Optional[List[str]] = None
    ) -> List[DetectionResult]:
        """Detect objects in multiple images."""
        return [
            self.detect(img, confidence_threshold, classes)
            for img in images
        ]
