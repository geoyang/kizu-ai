"""Celery tasks for background processing."""

import logging
from typing import List, Optional
from celery import shared_task

from api.workers.celery_app import celery_app
from api.routers.jobs import update_job, create_job
from api.schemas.requests import ProcessingOperation

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="process_image_task")
def process_image_task(
    self,
    job_id: str,
    asset_id: str,
    image_url: str,
    operations: List[str],
    user_id: str
):
    """Process a single image (used for webhook uploads)."""
    import asyncio
    from api.utils.image_utils import load_image
    from api.services import ProcessService
    from api.stores import SupabaseVectorStore
    from api.config import settings

    update_job(job_id, status="processing")

    try:
        # Initialize services
        store = SupabaseVectorStore(
            settings.supabase_url,
            settings.supabase_anon_key,
            settings.supabase_service_role_key
        )
        service = ProcessService(store)

        # Load image
        image = load_image(image_url)

        # Convert operation strings to enum
        ops = [ProcessingOperation(op) for op in operations]

        # Process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            service.process_image(asset_id, image, ops, user_id)
        )
        loop.close()

        update_job(
            job_id,
            status="completed",
            progress=100,
            result=result
        )

        return result

    except Exception as e:
        logger.error(f"Task failed: {e}")
        update_job(job_id, status="failed", error_message=str(e))
        raise


@celery_app.task(bind=True, name="batch_process_task")
def batch_process_task(
    self,
    job_id: str,
    asset_ids: Optional[List[str]],
    operations: List[str],
    user_id: str
):
    """Process multiple images in batch."""
    import asyncio
    from api.services import ProcessService
    from api.stores import SupabaseVectorStore
    from api.config import settings

    update_job(job_id, status="processing", total=len(asset_ids or []))

    try:
        store = SupabaseVectorStore(
            settings.supabase_url,
            settings.supabase_anon_key,
            settings.supabase_service_role_key
        )
        service = ProcessService(store)

        # Convert operations
        ops = [ProcessingOperation(op) for op in operations]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Process each asset
        processed = 0
        results = []

        for asset_id in (asset_ids or []):
            try:
                # Fetch image URL from Supabase (placeholder)
                image_url = f"https://example.com/{asset_id}"

                from api.utils.image_utils import load_image
                image = load_image(image_url)

                result = loop.run_until_complete(
                    service.process_image(asset_id, image, ops, user_id)
                )
                results.append(result)
                processed += 1

                # Update progress
                progress = int((processed / len(asset_ids)) * 100)
                update_job(job_id, progress=progress, processed=processed)

            except Exception as e:
                logger.error(f"Failed to process {asset_id}: {e}")
                results.append({"asset_id": asset_id, "error": str(e)})

        loop.close()

        update_job(
            job_id,
            status="completed",
            progress=100,
            processed=processed
        )

        return {"processed": processed, "results": results}

    except Exception as e:
        logger.error(f"Batch task failed: {e}")
        update_job(job_id, status="failed", error_message=str(e))
        raise


@celery_app.task(bind=True, name="cluster_faces_task")
def cluster_faces_task(
    self,
    job_id: str,
    user_id: str,
    threshold: float = 0.6,
    min_cluster_size: int = 2
):
    """Cluster all face embeddings for a user."""
    import asyncio
    from api.services import ClusteringService
    from api.stores import SupabaseVectorStore
    from api.config import settings

    update_job(job_id, status="processing")

    try:
        store = SupabaseVectorStore(
            settings.supabase_url,
            settings.supabase_anon_key,
            settings.supabase_service_role_key
        )
        service = ClusteringService(store)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            service.cluster_faces(user_id, threshold, min_cluster_size)
        )
        loop.close()

        update_job(
            job_id,
            status="completed",
            progress=100,
            result={
                "num_clusters": result.num_clusters,
                "num_faces_clustered": result.num_faces_clustered,
                "num_noise_faces": result.num_noise_faces
            }
        )

        return result

    except Exception as e:
        logger.error(f"Clustering task failed: {e}")
        update_job(job_id, status="failed", error_message=str(e))
        raise
