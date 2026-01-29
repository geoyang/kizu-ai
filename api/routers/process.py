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
    import logging
    from supabase import create_client
    from api.config import settings
    from api.routers.jobs import create_job, update_job

    logger = logging.getLogger(__name__)
    supabase = create_client(
        settings.supabase_url, settings.supabase_service_role_key
    )

    # Resolve asset list
    asset_ids = request.asset_ids
    if not asset_ids:
        result = supabase.table('assets') \
            .select('id') \
            .eq('user_id', user_id) \
            .limit(request.limit or 100) \
            .execute()
        asset_ids = [a['id'] for a in (result.data or [])]

    total = len(asset_ids)
    create_job(job_id, user_id, "batch", total=total)
    update_job(job_id, status="processing")

    if not asset_ids:
        update_job(job_id, status="completed", progress=100)
        return

    # Fetch web_uri for all assets
    uri_result = supabase.table('assets') \
        .select('id, web_uri') \
        .in_('id', asset_ids) \
        .execute()
    uri_map = {a['id']: a.get('web_uri') for a in (uri_result.data or [])}

    processed = 0
    errors = 0

    for asset_id in asset_ids:
        try:
            image_url = uri_map.get(asset_id)
            if not image_url:
                logger.warning(f"[batch] No web_uri for {asset_id}")
                errors += 1
                continue

            image = load_image(image_url)

            await process_service.process_image(
                asset_id=asset_id,
                image=image,
                operations=request.operations,
                user_id=user_id,
                store_results=True,
                force_reprocess=request.force_reprocess,
                worker_id=f"batch-{job_id[:8]}"
            )
            processed += 1

        except Exception as e:
            logger.error(f"[batch] Failed {asset_id}: {e}")
            errors += 1

        progress = int(((processed + errors) / total) * 100)
        update_job(job_id, processed=processed, progress=progress)

    update_job(
        job_id,
        status="completed",
        progress=100,
        processed=processed,
    )
    logger.info(
        f"[batch] Job {job_id[:8]} done: {processed}/{total} "
        f"processed, {errors} errors"
    )


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


@router.get("/unprocessed-count")
async def get_unprocessed_count(
    user_id: str = Depends(get_current_user_id)
):
    """Get the count of assets that have not been AI-processed."""
    from supabase import create_client
    from api.config import settings

    supabase = create_client(
        settings.supabase_url, settings.supabase_service_role_key
    )

    result = supabase.table('assets') \
        .select('id', count='exact') \
        .eq('user_id', user_id) \
        .is_('ai_processed_at', 'null') \
        .execute()

    return {
        "total_unprocessed": result.count or 0,
    }


@router.get("/unprocessed")
async def get_unprocessed_assets(
    limit: int = 100,
    offset: int = 0,
    user_id: str = Depends(get_current_user_id)
):
    """Get a page of unprocessed asset IDs with their web_uri."""
    from supabase import create_client
    from api.config import settings

    supabase = create_client(
        settings.supabase_url, settings.supabase_service_role_key
    )

    result = supabase.table('assets') \
        .select('id, web_uri') \
        .eq('user_id', user_id) \
        .is_('ai_processed_at', 'null') \
        .order('created_at') \
        .range(offset, offset + limit - 1) \
        .execute()

    return {
        "assets": result.data or [],
        "count": len(result.data or []),
        "offset": offset,
        "next_offset": offset + limit,
    }


@router.post("/queue-all")
async def queue_all_unprocessed(
    batch_size: int = 100,
    offset: int = 0,
    user_id: str = Depends(get_current_user_id)
):
    """
    Queue unprocessed assets to ai_processing_jobs for the worker.

    Inserts batch_size jobs at a time. Call repeatedly with increasing
    offset to queue the entire collection. The worker processes them
    and the Upload Queue tab shows per-asset progress.
    """
    from supabase import create_client
    from api.config import settings
    import logging

    logger = logging.getLogger(__name__)
    supabase = create_client(
        settings.supabase_url, settings.supabase_service_role_key
    )

    # Get unprocessed assets for this batch (include web_uri for the worker)
    assets_result = supabase.table('assets') \
        .select('id, web_uri') \
        .eq('user_id', user_id) \
        .is_('ai_processed_at', 'null') \
        .order('created_at') \
        .range(offset, offset + batch_size - 1) \
        .execute()

    assets = assets_result.data or []

    if not assets:
        count_result = supabase.table('assets') \
            .select('id', count='exact') \
            .eq('user_id', user_id) \
            .is_('ai_processed_at', 'null') \
            .execute()

        return {
            "queued": 0,
            "skipped": 0,
            "total_remaining": count_result.count or 0,
            "next_offset": offset,
            "done": True,
            "message": "No more unprocessed assets to queue",
        }

    asset_ids = [a['id'] for a in assets]
    # Build a map of asset_id -> web_uri for inclusion in job params
    uri_map = {a['id']: a.get('web_uri') for a in assets}

    # Check which already have pending/processing jobs to avoid duplicates
    existing_result = supabase.table('ai_processing_jobs') \
        .select('input_params') \
        .eq('user_id', user_id) \
        .in_('status', ['pending', 'processing']) \
        .execute()

    existing_asset_ids = set()
    for job in (existing_result.data or []):
        params = job.get('input_params') or {}
        aid = params.get('asset_id')
        if aid:
            existing_asset_ids.add(aid)

    # Build jobs to insert
    operations = ['embedding', 'faces', 'objects', 'describe']
    jobs_to_insert = []
    skipped = 0

    for asset_id in asset_ids:
        if asset_id in existing_asset_ids:
            skipped += 1
            continue
        image_url = uri_map.get(asset_id)
        if not image_url:
            skipped += 1
            logger.warning(f"[queue-all] Skipping {asset_id}: no web_uri")
            continue
        jobs_to_insert.append({
            'user_id': user_id,
            'job_type': 'process_image',
            'status': 'pending',
            'input_params': {
                'asset_id': asset_id,
                'image_url': image_url,
                'operations': operations,
            },
            'progress': 0,
            'total': 1,
        })

    queued = 0
    if jobs_to_insert:
        insert_result = supabase.table('ai_processing_jobs') \
            .insert(jobs_to_insert) \
            .execute()
        queued = len(insert_result.data or [])

    logger.info(
        f"[queue-all] Queued {queued} jobs, skipped {skipped} "
        f"(offset {offset}, batch {batch_size})"
    )

    # Get remaining count
    count_result = supabase.table('assets') \
        .select('id', count='exact') \
        .eq('user_id', user_id) \
        .is_('ai_processed_at', 'null') \
        .execute()

    total_remaining = count_result.count or 0

    return {
        "queued": queued,
        "skipped": skipped,
        "total_remaining": total_remaining,
        "next_offset": offset + batch_size,
        "done": len(assets) < batch_size,
        "message": f"Queued {queued} assets, skipped {skipped} duplicates, "
                   f"{total_remaining} remaining",
    }


@router.post("/batch-all")
async def process_batch_all(
    batch_size: int = 100,
    offset: int = 0,
    process_service: ProcessService = Depends(get_process_service),
    user_id: str = Depends(get_current_user_id)
):
    """
    Process a batch of unprocessed assets.

    Call repeatedly with increasing offset to process all assets.
    The frontend controls the loop and can stop between batches.
    """
    from supabase import create_client
    from api.config import settings
    import logging

    logger = logging.getLogger(__name__)
    supabase = create_client(
        settings.supabase_url, settings.supabase_service_role_key
    )

    # Get unprocessed assets
    assets_result = supabase.table('assets') \
        .select('id, web_uri') \
        .eq('user_id', user_id) \
        .is_('ai_processed_at', 'null') \
        .order('created_at') \
        .range(offset, offset + batch_size - 1) \
        .execute()

    if not assets_result.data:
        # Get remaining count to confirm we're done
        count_result = supabase.table('assets') \
            .select('id', count='exact') \
            .eq('user_id', user_id) \
            .is_('ai_processed_at', 'null') \
            .execute()

        return {
            "processed": 0,
            "failed": 0,
            "total_remaining": count_result.count or 0,
            "next_offset": offset,
            "errors": [],
            "message": "No more unprocessed assets in this range",
        }

    assets = assets_result.data
    processed = 0
    failed = 0
    errors = []

    for asset in assets:
        try:
            image_url = asset.get('web_uri')
            if not image_url:
                failed += 1
                errors.append({
                    "asset_id": asset['id'],
                    "error": "No web_uri"
                })
                continue

            image = load_image(image_url)

            from api.schemas.requests import ProcessingOperation
            await process_service.process_image(
                asset_id=asset['id'],
                image=image,
                operations=[
                    ProcessingOperation.EMBEDDING,
                    ProcessingOperation.OBJECTS,
                    ProcessingOperation.FACES,
                    ProcessingOperation.DESCRIBE,
                ],
                user_id=user_id,
                store_results=True,
                worker_id="api-batch-all"
            )
            processed += 1
            logger.info(
                f"[batch-all] Processed {asset['id'][:8]}... "
                f"({processed}/{len(assets)})"
            )

        except ValueError as e:
            failed += 1
            errors.append({
                "asset_id": asset['id'],
                "error": str(e)
            })
        except Exception as e:
            failed += 1
            errors.append({
                "asset_id": asset['id'],
                "error": str(e)
            })
            logger.error(f"[batch-all] Failed {asset['id']}: {e}")

    # Get remaining unprocessed count
    count_result = supabase.table('assets') \
        .select('id', count='exact') \
        .eq('user_id', user_id) \
        .is_('ai_processed_at', 'null') \
        .execute()

    total_remaining = count_result.count or 0

    return {
        "processed": processed,
        "failed": failed,
        "total_remaining": total_remaining,
        "next_offset": offset + batch_size,
        "errors": errors[:10],
        "message": f"Processed {processed}/{len(assets)} assets, "
                   f"{total_remaining} remaining",
    }


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
