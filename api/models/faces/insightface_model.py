"""InsightFace face detection and recognition model implementation."""

import logging
from typing import Optional
import numpy as np
from PIL import Image

from api.core.abstractions import (
    BaseFaceModel,
    FaceDetectionResult,
    DetectedFace,
    BoundingBox
)

logger = logging.getLogger(__name__)


class InsightFaceModel(BaseFaceModel):
    """InsightFace-based face detection and recognition."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        cache_dir: Optional[str] = None
    ):
        self._model_variant = model_name
        self._model_name = f"insightface-{model_name}"
        self._cache_dir = cache_dir or "~/.insightface"
        self._app = None
        self._embedding_dim = 512

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dim

    def load_model(self) -> None:
        """Load InsightFace model."""
        if self._app is not None:
            return

        from insightface.app import FaceAnalysis

        logger.info(f"Loading {self._model_name}")

        self._app = FaceAnalysis(
            name=self._model_variant,
            root=self._cache_dir,
            providers=['CPUExecutionProvider']  # Works on M1 and AMD
        )
        self._app.prepare(ctx_id=0, det_size=(1280, 1280))

        logger.info(f"Loaded {self._model_name}")

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self._app is not None:
            del self._app
            self._app = None
            logger.info(f"Unloaded {self._model_name}")

    def _pil_to_cv2(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format (BGR)."""
        import cv2
        rgb = np.array(image.convert('RGB'))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def detect_faces(
        self,
        image: Image.Image,
        min_face_size: int = 20
    ) -> FaceDetectionResult:
        """Detect faces without generating embeddings."""
        self.load_model()

        img_cv2 = self._pil_to_cv2(image)
        faces = self._app.get(img_cv2)

        detected = []
        for i, face in enumerate(faces):
            bbox = face.bbox
            if (bbox[2] - bbox[0]) < min_face_size:
                continue

            detected.append(DetectedFace(
                face_index=i,
                confidence=float(face.det_score),
                bounding_box=BoundingBox(
                    x=float(bbox[0]),
                    y=float(bbox[1]),
                    width=float(bbox[2] - bbox[0]),
                    height=float(bbox[3] - bbox[1])
                ),
                landmarks=self._extract_landmarks(face),
                embedding=None
            ))

        return FaceDetectionResult(
            faces=detected,
            model_version=self._model_name,
            image_width=image.width,
            image_height=image.height
        )

    def detect_and_embed(
        self,
        image: Image.Image,
        min_face_size: int = 20
    ) -> FaceDetectionResult:
        """Detect faces and generate embeddings."""
        self.load_model()

        img_cv2 = self._pil_to_cv2(image)
        faces = self._app.get(img_cv2)

        detected = []
        for i, face in enumerate(faces):
            bbox = face.bbox
            if (bbox[2] - bbox[0]) < min_face_size:
                continue

            embedding = face.embedding
            if embedding is not None:
                embedding = embedding / np.linalg.norm(embedding)

            detected.append(DetectedFace(
                face_index=i,
                confidence=float(face.det_score),
                bounding_box=BoundingBox(
                    x=float(bbox[0]),
                    y=float(bbox[1]),
                    width=float(bbox[2] - bbox[0]),
                    height=float(bbox[3] - bbox[1])
                ),
                landmarks=self._extract_landmarks(face),
                embedding=embedding
            ))

        return FaceDetectionResult(
            faces=detected,
            model_version=self._model_name,
            image_width=image.width,
            image_height=image.height
        )

    def get_embedding(self, face_image: Image.Image) -> np.ndarray:
        """Generate embedding for a cropped face image."""
        self.load_model()

        img_cv2 = self._pil_to_cv2(face_image)
        faces = self._app.get(img_cv2)

        if not faces:
            raise ValueError("No face detected in cropped image")

        embedding = faces[0].embedding
        return embedding / np.linalg.norm(embedding)

    def _extract_landmarks(self, face) -> Optional[dict]:
        """Extract facial landmarks if available."""
        if not hasattr(face, 'kps') or face.kps is None:
            return None

        kps = face.kps
        return {
            'left_eye': kps[0].tolist() if len(kps) > 0 else None,
            'right_eye': kps[1].tolist() if len(kps) > 1 else None,
            'nose': kps[2].tolist() if len(kps) > 2 else None,
            'left_mouth': kps[3].tolist() if len(kps) > 3 else None,
            'right_mouth': kps[4].tolist() if len(kps) > 4 else None,
        }
