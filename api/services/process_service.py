"""Service for processing images with AI models."""

import io
import logging
import uuid
from typing import List, Optional, Set
from PIL import Image, ImageOps

from api.core import ModelRegistry
from api.core.registry import ModelType
from api.schemas.requests import ProcessingOperation
from api.stores import SupabaseVectorStore
from api.utils.image_utils import load_image
from api.config import settings

logger = logging.getLogger(__name__)


class ProcessService:
    """Service for processing images with AI models."""

    def __init__(self, vector_store: SupabaseVectorStore):
        self._store = vector_store

    async def process_image(
        self,
        asset_id: str,
        image: Image.Image,
        operations: List[ProcessingOperation],
        user_id: str,
        store_results: bool = True,
        force_reprocess: bool = False,
        worker_id: str = None
    ) -> dict:
        """
        Process a single image with specified operations.

        Args:
            store_results: If False, skip storing to database (for testing)
            force_reprocess: If True, clear existing AI data before processing

        Returns dict with results for each operation.
        """
        # Apply EXIF orientation before any processing - phone photos
        # are often stored rotated with an EXIF tag, which causes face
        # detection to fail on sideways/upside-down images
        image = ImageOps.exif_transpose(image)

        results = {"asset_id": asset_id}
        ops = self._expand_operations(operations)

        # Clear existing AI data if force_reprocess is True
        if force_reprocess and store_results:
            await self._clear_asset_ai_data(asset_id)

        if ProcessingOperation.EMBEDDING in ops:
            results["embedding"] = await self._generate_embedding(
                asset_id, image, user_id, store_results
            )

        if ProcessingOperation.OBJECTS in ops:
            results["objects"] = await self._detect_objects(
                asset_id, image, user_id, store=store_results
            )

        if ProcessingOperation.FACES in ops:
            results["faces"] = await self._detect_faces(
                asset_id, image, user_id, store_results
            )

            # Recognize detected faces against known contacts
            if store_results and results["faces"].get("count", 0) > 0:
                try:
                    recognized = await self._recognize_new_faces(
                        asset_id, user_id, worker_id
                    )
                    results["faces"]["recognized"] = recognized
                except Exception as e:
                    logger.warning(f"Face recognition failed: {e}")

        if ProcessingOperation.OCR in ops:
            results["ocr"] = await self._extract_text(
                asset_id, image, user_id
            )

        if ProcessingOperation.DESCRIBE in ops:
            results["description"] = await self._generate_description(
                asset_id, image, user_id
            )

        # Mark asset as AI processed
        if store_results:
            operation_names = [op.value for op in ops]
            await self._mark_asset_processed(asset_id, operation_names, worker_id)

        return results

    def _expand_operations(
        self,
        operations: List[ProcessingOperation]
    ) -> Set[ProcessingOperation]:
        """Expand 'all' into individual operations."""
        ops = set(operations)
        if ProcessingOperation.ALL in ops:
            ops = {
                ProcessingOperation.EMBEDDING,
                ProcessingOperation.OBJECTS,
                ProcessingOperation.FACES,
                ProcessingOperation.OCR,
                ProcessingOperation.DESCRIBE,
            }
        return ops

    async def _generate_embedding(
        self,
        asset_id: str,
        image: Image.Image,
        user_id: str,
        store: bool = True
    ) -> dict:
        """Generate and optionally store CLIP embedding."""
        embedder = ModelRegistry.get(ModelType.EMBEDDER)
        result = embedder.embed_image(image)

        if store:
            await self._store.store_embedding(
                collection="image_embeddings",
                id=asset_id,
                embedding=result.embedding,
                metadata={
                    "model_version": result.model_version,
                    "user_id": user_id
                }
            )

        return {"success": True, "dimension": result.dimension, "stored": store}

    async def _detect_objects(
        self,
        asset_id: str,
        image: Image.Image,
        user_id: str,
        store: bool = True
    ) -> dict:
        """Detect objects in image."""
        detector = ModelRegistry.get(ModelType.DETECTOR)
        result = detector.detect(image, confidence_threshold=0.5)

        objects = [
            {"class": obj.class_name, "confidence": obj.confidence}
            for obj in result.objects
        ]

        # Store detected objects to database
        if store and objects:
            try:
                await self._store_detected_objects(asset_id, user_id, objects)
            except Exception as e:
                logger.warning(f"Failed to store detected objects: {e}")

        return {
            "success": True,
            "count": len(result.objects),
            "objects": objects
        }

    async def _store_detected_objects(
        self,
        asset_id: str,
        user_id: str,
        objects: list
    ) -> None:
        """Store detected objects to database and asset_tags."""
        from supabase import create_client
        from api.config import settings
        import uuid

        supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

        # Delete existing objects for this asset first
        try:
            supabase.table('detected_objects').delete().eq('asset_id', asset_id).execute()
        except Exception as e:
            logger.warning(f"Could not delete existing objects (table may not exist): {e}")

        # Delete existing AI-generated object tags (those without a user creator)
        try:
            supabase.table('asset_tags').delete().eq('asset_id', asset_id).eq('tag_type', 'object').is_('created_by', 'null').execute()
        except Exception as e:
            logger.warning(f"Could not delete existing object tags: {e}")

        # Insert new objects
        records = [
            {
                "asset_id": asset_id,
                "user_id": user_id,
                "object_class": obj["class"],
                "confidence": obj["confidence"],
                "model_version": "yolov8-m"
            }
            for obj in objects
        ]

        if records:
            try:
                supabase.table('detected_objects').insert(records).execute()
                logger.info(f"Stored {len(records)} detected objects for {asset_id}")
            except Exception as e:
                logger.warning(f"Could not store detected objects (table may not exist): {e}")

        # Also add unique objects as asset_tags for UI display
        unique_objects = list(set(obj["class"] for obj in objects))
        tag_records = [
            {
                "id": str(uuid.uuid4()),
                "asset_id": asset_id,
                "tag_type": "object",
                "tag_value": obj_class,
            }
            for obj_class in unique_objects
        ]

        if tag_records:
            try:
                supabase.table('asset_tags').insert(tag_records).execute()
                logger.info(f"Added {len(tag_records)} object tags for {asset_id}")
            except Exception as e:
                logger.warning(f"Could not add object tags: {e}")

    async def _detect_faces(
        self,
        asset_id: str,
        image: Image.Image,
        user_id: str,
        store: bool = True,
        is_from_video: bool = False
    ) -> dict:
        """Detect faces, extract thumbnails, and optionally store embeddings."""
        face_model = ModelRegistry.get(ModelType.FACE)
        result = face_model.detect_and_embed(image)

        # Store each face embedding if requested
        if store:
            for face in result.faces:
                if face.embedding is not None:
                    await self._store.store_embedding(
                        collection="face_embeddings",
                        id=f"{asset_id}_{face.face_index}",
                        embedding=face.embedding,
                        metadata={
                            "asset_id": asset_id,
                            "face_index": face.face_index,
                            "bounding_box": face.bounding_box.to_dict(),
                            "user_id": user_id,
                            "is_from_video": is_from_video
                        }
                    )

        return {
            "success": True,
            "count": len(result.faces),
            "stored": store,
            "faces": [
                {
                    "index": f.face_index,
                    "confidence": f.confidence,
                    "bounding_box": f.bounding_box.to_dict() if f.bounding_box else None
                }
                for f in result.faces
            ]
        }

    async def _extract_text(
        self,
        asset_id: str,
        image: Image.Image,
        user_id: str
    ) -> dict:
        """Extract text from image using OCR."""
        ocr = ModelRegistry.get(ModelType.OCR)
        result = ocr.extract_text(image)

        # Store extracted text (would go to image_text table)
        return {
            "success": True,
            "has_text": len(result.full_text) > 0,
            "text": result.full_text[:500] if result.full_text else None
        }

    async def _generate_description(
        self,
        asset_id: str,
        image: Image.Image,
        user_id: str
    ) -> dict:
        """Generate natural language description."""
        try:
            vlm = ModelRegistry.get(ModelType.VLM)
            result = vlm.describe_image(image, detail_level="medium")

            return {
                "success": True,
                "description": result.text,
                "processing_time_ms": result.processing_time_ms
            }
        except Exception as e:
            logger.warning(f"VLM description failed: {e}")
            return {"success": False, "error": str(e)}

    async def _recognize_new_faces(
        self, asset_id: str, user_id: str, worker_id: str = None
    ) -> int:
        """Recognize newly detected faces against labeled embeddings."""
        from api.services.clustering_service import ClusteringService

        clustering = ClusteringService(self._store)
        return await clustering.recognize_faces(
            asset_id, user_id, worker_id=worker_id
        )

    async def _clear_asset_ai_data(self, asset_id: str) -> None:
        """Clear all existing AI data for an asset before reprocessing."""
        from supabase import create_client

        try:
            supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

            # Delete image embeddings
            try:
                supabase.table('image_embeddings').delete().eq('asset_id', asset_id).execute()
            except Exception as e:
                logger.debug(f"Could not delete image_embeddings: {e}")

            # Delete face embeddings
            try:
                supabase.table('face_embeddings').delete().eq('asset_id', asset_id).execute()
            except Exception as e:
                logger.debug(f"Could not delete face_embeddings: {e}")

            # Delete detected objects
            try:
                supabase.table('detected_objects').delete().eq('asset_id', asset_id).execute()
            except Exception as e:
                logger.debug(f"Could not delete detected_objects: {e}")

            # Delete image descriptions
            try:
                supabase.table('image_descriptions').delete().eq('asset_id', asset_id).execute()
            except Exception as e:
                logger.debug(f"Could not delete image_descriptions: {e}")

            # Delete image text (OCR)
            try:
                supabase.table('image_text').delete().eq('asset_id', asset_id).execute()
            except Exception as e:
                logger.debug(f"Could not delete image_text: {e}")

            logger.info(f"Cleared AI data for asset {asset_id}")

        except Exception as e:
            logger.warning(f"Failed to clear AI data for asset {asset_id}: {e}")

    async def _mark_asset_processed(
        self,
        asset_id: str,
        operations: List[str],
        worker_id: str = None
    ) -> None:
        """Mark an asset as AI processed with timestamp, operations, and model versions."""
        from supabase import create_client
        from datetime import datetime, timezone

        try:
            supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

            # Build model versions based on operations performed
            model_versions = {}
            if 'embedding' in operations:
                model_versions['clip'] = settings.clip_model
            if 'objects' in operations:
                model_versions['yolo'] = f"v8-{settings.yolo_size}"
            if 'faces' in operations:
                model_versions['insightface'] = settings.face_model_name
            if 'ocr' in operations:
                model_versions['ocr'] = 'easyocr'
            if 'describe' in operations:
                model_versions['vlm'] = settings.default_vlm

            now = datetime.now(timezone.utc)

            supabase.table('assets').update({
                'ai_processed_at': now.isoformat(),
                'ai_processed_date': now.date().isoformat(),
                'ai_processed_by': worker_id,
                'ai_processed_operations': operations,
                'ai_model_versions': model_versions
            }).eq('id', asset_id).execute()

            logger.debug(
                f"Marked asset {asset_id} as AI processed: "
                f"operations={operations}, worker={worker_id}, models={model_versions}"
            )

        except Exception as e:
            logger.warning(f"Failed to mark asset {asset_id} as processed: {e}")

    async def check_asset_processed(self, asset_id: str) -> bool:
        """Check if an asset has already been AI processed."""
        from supabase import create_client

        try:
            supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

            result = supabase.table('assets').select('ai_processed_at').eq('id', asset_id).single().execute()

            return result.data and result.data.get('ai_processed_at') is not None

        except Exception as e:
            logger.warning(f"Failed to check processing status for asset {asset_id}: {e}")
            return False
