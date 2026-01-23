"""Image processing API router."""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Optional
import uuid

from api.schemas import (
    ProcessImageRequest,
    BatchProcessRequest,
    ProcessingJobResponse,
)
from api.schemas.responses import JobStatus
from api.services import ProcessService
from api.dependencies import get_process_service, get_current_user_id
from api.utils.image_utils import load_image

router = APIRouter(prefix="/process", tags=["process"])


@router.post("/image")
async def process_single_image(
    request: ProcessImageRequest,
    background_tasks: BackgroundTasks,
    process_service: ProcessService = Depends(get_process_service),
    user_id: str = Depends(get_current_user_id)
):
    """
    Process a single image with AI models.

    Operations: embedding, faces, objects, ocr, describe, all
    """
    # Load image from URL or base64
    if not request.image_base64 and not request.image_url:
        raise HTTPException(
            status_code=400,
            detail="image_url or image_base64 required"
        )

    try:
        if request.image_base64:
            image = load_image(request.image_base64, is_base64=True)
        else:
            image = load_image(request.image_url)
    except ValueError as e:
        # Image loading error (video file, invalid image, etc.)
        return {
            "success": False,
            "status": "skipped",
            "error": str(e),
            "asset_id": request.asset_id
        }
    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "error": f"Failed to load image: {str(e)}",
            "asset_id": request.asset_id
        }

    try:
        # Process synchronously for single image
        result = await process_service.process_image(
            asset_id=request.asset_id,
            image=image,
            operations=request.operations,
            user_id=user_id,
            store_results=request.store_results,
            force_reprocess=request.force_reprocess,
            worker_id="api-direct"
        )

        return {
            "success": True,
            "status": "completed",
            "result": result
        }

    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "error": str(e),
            "asset_id": request.asset_id
        }


@router.post("/batch", response_model=ProcessingJobResponse)
async def process_batch(
    request: BatchProcessRequest,
    background_tasks: BackgroundTasks,
    process_service: ProcessService = Depends(get_process_service),
    user_id: str = Depends(get_current_user_id)
):
    """
    Process multiple images asynchronously.

    Returns a job ID to poll for progress.
    """
    from datetime import datetime

    job_id = str(uuid.uuid4())

    # Queue background task
    background_tasks.add_task(
        process_batch_task,
        job_id=job_id,
        request=request,
        user_id=user_id,
        process_service=process_service
    )

    return ProcessingJobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress=0,
        processed=0,
        total=len(request.asset_ids) if request.asset_ids else 0,
        created_at=datetime.utcnow()
    )


async def process_batch_task(
    job_id: str,
    request: BatchProcessRequest,
    user_id: str,
    process_service: ProcessService
):
    """Background task for batch processing."""
    # This would be handled by Celery in production
    # Placeholder for FastAPI BackgroundTasks demo
    pass


@router.post("/webhook")
async def process_webhook(
    asset_id: str,
    image_url: str,
    process_service: ProcessService = Depends(get_process_service),
    user_id: str = Depends(get_current_user_id)
):
    """
    Webhook endpoint for real-time processing on upload.

    Called by Kizu app when a new image is uploaded.
    """
    try:
        image = load_image(image_url)

        # Quick processing (embedding, objects, faces)
        from api.schemas.requests import ProcessingOperation
        result = await process_service.process_image(
            asset_id=asset_id,
            image=image,
            operations=[
                ProcessingOperation.EMBEDDING,
                ProcessingOperation.OBJECTS,
                ProcessingOperation.FACES
            ],
            user_id=user_id,
            worker_id="api-webhook"
        )

        return {"status": "processed", "result": result}

    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.post("/reindex")
async def reindex_all_assets(
    background_tasks: BackgroundTasks,
    limit: int = 100,
    offset: int = 0,
    process_service: ProcessService = Depends(get_process_service),
    user_id: str = Depends(get_current_user_id)
):
    """
    Reindex assets that don't have embeddings.

    Call this endpoint repeatedly with increasing offset to process all assets.
    Use limit to control batch size.
    """
    from supabase import create_client
    from api.config import settings
    import logging

    logger = logging.getLogger(__name__)
    supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

    try:
        # Get assets that don't have embeddings yet
        # First get all asset IDs
        assets_result = supabase.table('assets') \
            .select('id, web_uri') \
            .eq('user_id', user_id) \
            .range(offset, offset + limit - 1) \
            .execute()

        if not assets_result.data:
            return {
                "status": "complete",
                "message": "No more assets to process",
                "processed": 0,
                "offset": offset
            }

        asset_ids = [a['id'] for a in assets_result.data]

        # Check which already have embeddings
        embeddings_result = supabase.table('image_embeddings') \
            .select('asset_id') \
            .in_('asset_id', asset_ids) \
            .execute()

        existing_ids = {e['asset_id'] for e in embeddings_result.data}
        assets_to_process = [a for a in assets_result.data if a['id'] not in existing_ids]

        if not assets_to_process:
            return {
                "status": "skipped",
                "message": f"All {len(asset_ids)} assets in this batch already processed",
                "processed": 0,
                "offset": offset,
                "next_offset": offset + limit
            }

        # Process each asset
        processed = 0
        errors = []

        for asset in assets_to_process:
            try:
                image_url = asset.get('web_uri')
                if not image_url:
                    continue

                image = load_image(image_url)
                from api.schemas.requests import ProcessingOperation

                await process_service.process_image(
                    asset_id=asset['id'],
                    image=image,
                    operations=[
                        ProcessingOperation.EMBEDDING,
                        ProcessingOperation.OBJECTS,
                        ProcessingOperation.FACES
                    ],
                    user_id=user_id,
                    worker_id="api-reindex"
                )
                processed += 1
                logger.info(f"Reindexed asset {asset['id'][:8]}... ({processed}/{len(assets_to_process)})")

            except Exception as e:
                errors.append({"asset_id": asset['id'], "error": str(e)})
                logger.error(f"Failed to reindex {asset['id']}: {e}")

        return {
            "status": "success",
            "processed": processed,
            "errors": len(errors),
            "error_details": errors[:5] if errors else None,
            "offset": offset,
            "next_offset": offset + limit,
            "message": f"Processed {processed}/{len(assets_to_process)} assets"
        }

    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
