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
        worker_id: str = None,
        restore_type: str = "both",
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

        if ProcessingOperation.RESTORE in ops:
            results["restore"] = await self._restore_image(
                asset_id, image, user_id, store_results, restore_type
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
                    # Extract and upload face thumbnail
                    face_thumbnail_url = await self._extract_and_upload_face_thumbnail(
                        image, face.bounding_box, asset_id, face.face_index, user_id
                    )

                    await self._store.store_embedding(
                        collection="face_embeddings",
                        id=f"{asset_id}_{face.face_index}",
                        embedding=face.embedding,
                        metadata={
                            "asset_id": asset_id,
                            "face_index": face.face_index,
                            "bounding_box": face.bounding_box.to_dict(),
                            "user_id": user_id,
                            "face_thumbnail_url": face_thumbnail_url,
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

    async def _extract_and_upload_face_thumbnail(
        self,
        image: Image.Image,
        bounding_box,
        asset_id: str,
        face_index: int,
        user_id: str,
        padding: float = 0.3,
        thumbnail_size: int = 150
    ) -> Optional[str]:
        """Extract face crop from image and upload to storage."""
        try:
            if not bounding_box:
                return None

            # EXIF orientation is already applied in process_image()
            # Get bounding box coordinates
            x = bounding_box.x
            y = bounding_box.y
            width = bounding_box.width
            height = bounding_box.height

            # Add padding around the face (30% extra on each side)
            pad_x = width * padding
            pad_y = height * padding

            left = max(0, int(x - pad_x))
            top = max(0, int(y - pad_y))
            right = min(image.width, int(x + width + pad_x))
            bottom = min(image.height, int(y + height + pad_y))

            # Crop the face region
            face_crop = image.crop((left, top, right, bottom))

            # Resize to thumbnail size (square)
            face_crop.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)

            # Convert to JPEG bytes
            buffer = io.BytesIO()
            face_crop.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            image_bytes = buffer.getvalue()

            # Upload to Supabase storage
            from supabase import create_client

            supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

            # Create unique filename: user_id/asset_id_face_index.jpg
            filename = f"{user_id}/{asset_id}_{face_index}.jpg"

            # Upload to face-thumbnails bucket
            result = supabase.storage.from_('face-thumbnails').upload(
                filename,
                image_bytes,
                file_options={"content-type": "image/jpeg", "upsert": "true"}
            )

            # Get public URL
            public_url = supabase.storage.from_('face-thumbnails').get_public_url(filename)

            logger.info(f"Uploaded face thumbnail: {filename}")
            return public_url

        except Exception as e:
            logger.error(f"Failed to extract/upload face thumbnail: {e}")
            return None

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

    async def _restore_image(
        self,
        asset_id: str,
        image: Image.Image,
        user_id: str,
        store: bool = True,
        restore_type: str = "both",
    ) -> dict:
        """Restore/enhance image using Real-ESRGAN and/or GFP-GAN."""
        import time

        # Determine which model to use
        if restore_type == "faces" or restore_type == "both":
            model_name = "gfpgan"
        else:
            model_name = "realesrgan"

        try:
            # Free memory by unloading other models before restore
            for mt in [ModelType.EMBEDDER, ModelType.DETECTOR,
                       ModelType.FACE, ModelType.OCR, ModelType.VLM]:
                for name in list(ModelRegistry._instances.keys()):
                    if name.startswith(mt.value + ":"):
                        ModelRegistry.unload(mt, name.split(":", 1)[1])

            # Limit input image size to prevent OOM on CPU
            max_dim = settings.restore_max_input_size
            if max(image.size) > max_dim:
                ratio = max_dim / max(image.size)
                new_size = (
                    int(image.width * ratio),
                    int(image.height * ratio),
                )
                logger.info(
                    f"Downscaling restore input {image.size} -> {new_size} "
                    f"(max {max_dim}px)"
                )
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            restorer = ModelRegistry.get(ModelType.RESTORER, model_name)
            result = restorer.restore(image)

            if store:
                # Convert restored image to JPEG bytes
                buffer = io.BytesIO()
                restored_rgb = result.restored_image
                if restored_rgb.mode in ('RGBA', 'P'):
                    restored_rgb = restored_rgb.convert('RGB')
                restored_rgb.save(buffer, format='JPEG', quality=92)
                buffer.seek(0)
                image_bytes = buffer.getvalue()

                # Upload to Supabase Storage
                from supabase import create_client

                supabase = create_client(
                    settings.supabase_url,
                    settings.supabase_service_role_key,
                )

                timestamp = int(time.time())
                storage_path = (
                    f"{user_id}/restored/{asset_id}_{timestamp}.jpg"
                )

                import httpx
                async with httpx.AsyncClient(timeout=120.0) as client:
                    upload_url = (
                        f"{settings.supabase_url}/storage/v1/object"
                        f"/assets/{storage_path}"
                    )
                    response = await client.post(
                        upload_url,
                        content=image_bytes,
                        headers={
                            "Authorization": f"Bearer {settings.supabase_service_role_key}",
                            "Content-Type": "image/jpeg",
                            "x-upsert": "true",
                        },
                    )
                    if response.status_code not in [200, 201]:
                        raise RuntimeError(
                            f"Upload failed: {response.status_code}"
                        )

                web_uri = (
                    f"{settings.supabase_url}/storage/v1/object/public"
                    f"/assets/{storage_path}"
                )

                # Insert record into restored_assets table
                restored_record = {
                    "original_asset_id": asset_id,
                    "user_id": user_id,
                    "storage_path": storage_path,
                    "web_uri": web_uri,
                    "restore_type": restore_type,
                    "model_used": result.model_version,
                    "upscale_factor": settings.restore_upscale,
                    "processing_time_ms": result.processing_time_ms,
                    "file_size_bytes": len(image_bytes),
                }

                insert_result = supabase.table(
                    'restored_assets'
                ).insert(restored_record).execute()

                restored_asset_id = (
                    insert_result.data[0].get('id')
                    if insert_result.data else None
                )

                logger.info(
                    f"Restored asset {asset_id} -> {restored_asset_id} "
                    f"({restore_type}, {result.processing_time_ms}ms)"
                )

                # Unload model after processing to free memory
                ModelRegistry.unload(ModelType.RESTORER, model_name)

                return {
                    "success": True,
                    "restored_asset_id": restored_asset_id,
                    "storage_path": storage_path,
                    "web_uri": web_uri,
                    "processing_time_ms": result.processing_time_ms,
                }

            # Non-store mode (testing)
            ModelRegistry.unload(ModelType.RESTORER, model_name)

            return {
                "success": True,
                "width": result.restored_image.width,
                "height": result.restored_image.height,
                "processing_time_ms": result.processing_time_ms,
            }

        except Exception as e:
            logger.error(f"Restoration failed for {asset_id}: {e}")
            # Ensure model is unloaded even on failure
            try:
                ModelRegistry.unload(ModelType.RESTORER, model_name)
            except Exception:
                pass
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
            if 'restore' in operations:
                model_versions['restorer'] = settings.restore_model

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
