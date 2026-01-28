"""Face recognition API router."""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
import uuid

from api.schemas import (
    ClusterFacesRequest,
    AssignClusterRequest,
    MergeClustersRequest,
    ProcessingJobResponse,
    TagSyncPreviewRequest,
    ApplyTagSyncRequest,
)
from api.schemas.responses import (
    JobStatus,
    FaceClusterResponse,
    FaceClustersListResponse,
    TagSyncPreviewResponse,
    TagSyncSummary,
    AssetTagPreview,
    ManualTag,
    AIDetection,
    TagMatch,
    BoundingBox,
)
from api.services import FaceService, ClusteringService
from api.dependencies import (
    get_face_service,
    get_clustering_service,
    get_current_user_id
)


def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    x1_1, y1_1 = box1.get("x", 0), box1.get("y", 0)
    x2_1 = x1_1 + box1.get("width", 0)
    y2_1 = y1_1 + box1.get("height", 0)

    x1_2, y1_2 = box2.get("x", 0), box2.get("y", 0)
    x2_2 = x1_2 + box2.get("width", 0)
    y2_2 = y1_2 + box2.get("height", 0)

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = box1.get("width", 0) * box1.get("height", 0)
    box2_area = box2.get("width", 0) * box2.get("height", 0)
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area

router = APIRouter(prefix="/faces", tags=["faces"])


@router.post("/cluster", response_model=ProcessingJobResponse)
async def cluster_faces(
    request: ClusterFacesRequest,
    background_tasks: BackgroundTasks,
    clustering_service: ClusteringService = Depends(get_clustering_service),
    user_id: str = Depends(get_current_user_id)
):
    """
    Cluster all face embeddings for the user.

    Groups similar faces together for easier labeling.
    """
    from datetime import datetime
    from api.routers.jobs import create_job, update_job

    job_id = str(uuid.uuid4())

    # Register job so it can be polled
    create_job(job_id, user_id, "cluster_faces", total=0)

    background_tasks.add_task(
        run_clustering_task,
        job_id=job_id,
        request=request,
        user_id=user_id,
        clustering_service=clustering_service
    )

    return ProcessingJobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress=0,
        created_at=datetime.utcnow()
    )


async def run_clustering_task(
    job_id: str,
    request: ClusterFacesRequest,
    user_id: str,
    clustering_service: ClusteringService
):
    """Background task for face clustering."""
    from api.routers.jobs import update_job

    try:
        # Update job status to processing
        update_job(job_id, status="processing", progress=10)

        result = await clustering_service.cluster_faces(
            user_id=user_id,
            threshold=request.threshold,
            min_cluster_size=request.min_cluster_size
        )

        # Update job status to completed
        update_job(
            job_id,
            status="completed",
            progress=100,
            processed=result.num_clusters if result else 0,
            total=result.num_faces_clustered + result.num_noise_faces if result else 0
        )
    except Exception as e:
        # Update job status to failed
        update_job(job_id, status="failed", error_message=str(e))


@router.get("/clusters", response_model=FaceClustersListResponse)
async def list_clusters(
    include_labeled: bool = True,
    include_unlabeled: bool = True,
    clustering_service: ClusteringService = Depends(get_clustering_service),
    user_id: str = Depends(get_current_user_id)
):
    """List all face clusters for the user."""
    clusters = await clustering_service.get_clusters(user_id)

    # Filter based on params
    if not include_labeled:
        clusters = [c for c in clusters if c.knox_contact_id is None]
    if not include_unlabeled:
        clusters = [c for c in clusters if c.knox_contact_id is not None]

    total_faces = sum(c.face_count for c in clusters)
    unlabeled = sum(1 for c in clusters if c.knox_contact_id is None)

    return FaceClustersListResponse(
        clusters=clusters,
        total_faces=total_faces,
        unlabeled_clusters=unlabeled
    )


@router.get("/clusters/{cluster_id}", response_model=FaceClusterResponse)
async def get_cluster(
    cluster_id: str,
    clustering_service: ClusteringService = Depends(get_clustering_service),
    user_id: str = Depends(get_current_user_id)
):
    """Get details of a specific cluster."""
    clusters = await clustering_service.get_clusters(user_id)
    for cluster in clusters:
        if cluster.cluster_id == cluster_id:
            return cluster

    raise HTTPException(status_code=404, detail="Cluster not found")


@router.get("/clusters/{cluster_id}/faces")
async def get_cluster_faces(
    cluster_id: str,
    clustering_service: ClusteringService = Depends(get_clustering_service),
    user_id: str = Depends(get_current_user_id)
):
    """Get ALL faces in a specific cluster."""
    faces = await clustering_service.get_all_faces_for_cluster(cluster_id, user_id)
    return {"cluster_id": cluster_id, "faces": faces, "total": len(faces)}


@router.post("/clusters/{cluster_id}/assign")
async def assign_cluster(
    cluster_id: str,
    request: AssignClusterRequest,
    clustering_service: ClusteringService = Depends(get_clustering_service),
    user_id: str = Depends(get_current_user_id)
):
    """Assign a cluster to a Knox contact."""
    success = await clustering_service.assign_cluster_to_contact(
        cluster_id=cluster_id,
        contact_id=request.knox_contact_id,
        name=request.name,
        user_id=user_id,
        exclude_face_ids=request.exclude_face_ids
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to assign cluster")

    return {"status": "assigned", "cluster_id": cluster_id}


@router.post("/clusters/{cluster_id}/remove-faces")
async def remove_faces_from_cluster(
    cluster_id: str,
    face_ids: List[str],
    clustering_service: ClusteringService = Depends(get_clustering_service),
    user_id: str = Depends(get_current_user_id)
):
    """Remove specific faces from a cluster (set their cluster_id to null)."""
    success = await clustering_service.remove_faces_from_cluster(
        cluster_id=cluster_id,
        face_ids=face_ids,
        user_id=user_id
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to remove faces")

    return {"status": "removed", "count": len(face_ids)}


@router.post("/clusters/merge")
async def merge_clusters(
    request: MergeClustersRequest,
    clustering_service: ClusteringService = Depends(get_clustering_service),
    user_id: str = Depends(get_current_user_id)
):
    """Merge multiple clusters into one."""
    if len(request.cluster_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 clusters to merge"
        )

    new_cluster_id = await clustering_service.merge_clusters(
        request.cluster_ids, user_id
    )

    return {"status": "merged", "new_cluster_id": new_cluster_id}


@router.get("/contact/{contact_id}/images")
async def get_contact_images(
    contact_id: str,
    limit: int = 50,
    user_id: str = Depends(get_current_user_id)
):
    """
    Get images containing a specific contact's face.

    Returns thumbnail URLs of images where this contact has been identified.
    """
    from api.config import settings
    from supabase import create_client

    try:
        supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

        # Find face embeddings linked to this contact (via cluster or direct assignment)
        # First get clusters linked to this contact
        clusters_result = supabase.table("face_clusters") \
            .select("id") \
            .eq("user_id", user_id) \
            .eq("knox_contact_id", contact_id) \
            .execute()

        cluster_ids = [c["id"] for c in (clusters_result.data or [])]

        # Get face embeddings from these clusters
        face_asset_ids = set()

        if cluster_ids:
            faces_result = supabase.table("face_embeddings") \
                .select("asset_id, face_thumbnail_url") \
                .eq("user_id", user_id) \
                .in_("cluster_id", cluster_ids) \
                .execute()

            for face in (faces_result.data or []):
                face_asset_ids.add(face["asset_id"])

        # Also check for direct knox_contact_id assignment on face_embeddings
        direct_faces_result = supabase.table("face_embeddings") \
            .select("asset_id, face_thumbnail_url") \
            .eq("user_id", user_id) \
            .eq("knox_contact_id", contact_id) \
            .execute()

        for face in (direct_faces_result.data or []):
            face_asset_ids.add(face["asset_id"])

        if not face_asset_ids:
            return {"contact_id": contact_id, "images": [], "total": 0}

        # Get thumbnail URLs from album_assets for these asset IDs
        asset_ids_list = list(face_asset_ids)[:limit]
        assets_result = supabase.table("album_assets") \
            .select("asset_id, thumbnail_uri, asset_uri") \
            .in_("asset_id", asset_ids_list) \
            .execute()

        images = []
        for asset in (assets_result.data or []):
            thumbnail_url = asset.get("thumbnail_uri") or asset.get("asset_uri")
            if thumbnail_url and thumbnail_url.startswith("http"):
                images.append({
                    "asset_id": asset["asset_id"],
                    "thumbnail_url": thumbnail_url
                })

        return {
            "contact_id": contact_id,
            "images": images,
            "total": len(face_asset_ids)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear-clustering")
async def clear_clustering(
    clear_thumbnails: bool = False,
    clustering_service: ClusteringService = Depends(get_clustering_service),
    user_id: str = Depends(get_current_user_id)
):
    """
    Clear all clustering data for the user.

    This will:
    - Delete all face clusters
    - Clear cluster_id and knox_contact_id from all face embeddings
    - Optionally clear face thumbnails (if clear_thumbnails=True)

    After clearing, you can re-run clustering and backfill thumbnails.
    """
    from api.config import settings
    from supabase import create_client

    try:
        supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

        # Delete all face clusters for this user
        supabase.table("face_clusters") \
            .delete() \
            .eq("user_id", user_id) \
            .execute()

        # Clear cluster assignments from face embeddings
        update_data = {"cluster_id": None, "knox_contact_id": None}
        if clear_thumbnails:
            update_data["face_thumbnail_url"] = None

        supabase.table("face_embeddings") \
            .update(update_data) \
            .eq("user_id", user_id) \
            .execute()

        return {
            "status": "cleared",
            "message": "All clustering data has been cleared" + (" (including thumbnails)" if clear_thumbnails else ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backfill-thumbnails")
async def backfill_face_thumbnails(
    background_tasks: BackgroundTasks,
    clustering_service: ClusteringService = Depends(get_clustering_service),
    user_id: str = Depends(get_current_user_id)
):
    """Generate face thumbnails for existing face embeddings that don't have them."""
    from datetime import datetime
    from api.routers.jobs import create_job

    job_id = str(uuid.uuid4())
    create_job(job_id, user_id, "backfill_thumbnails", total=0)

    background_tasks.add_task(
        run_backfill_thumbnails_task,
        job_id=job_id,
        user_id=user_id,
        clustering_service=clustering_service
    )

    return {
        "status": "started",
        "job_id": job_id,
        "message": "Backfilling face thumbnails in background"
    }


async def run_backfill_thumbnails_task(
    job_id: str,
    user_id: str,
    clustering_service: ClusteringService
):
    """Background task to generate face thumbnails for existing embeddings."""
    from api.routers.jobs import update_job
    from api.config import settings
    from supabase import create_client
    from PIL import Image
    import httpx
    import io

    supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

    try:
        update_job(job_id, status="processing", progress=5)

        # Get all face embeddings without thumbnails
        result = supabase.table("face_embeddings") \
            .select("id, asset_id, face_index, bounding_box, face_thumbnail_url") \
            .eq("user_id", user_id) \
            .is_("face_thumbnail_url", "null") \
            .execute()

        faces_to_process = result.data or []
        total = len(faces_to_process)

        if total == 0:
            update_job(job_id, status="completed", progress=100, processed=0, total=0)
            return

        update_job(job_id, status="processing", progress=10, total=total)

        processed = 0
        async with httpx.AsyncClient() as client:
            for face in faces_to_process:
                try:
                    asset_id = face["asset_id"]
                    face_index = face["face_index"]
                    bounding_box = face.get("bounding_box")

                    if not bounding_box:
                        processed += 1
                        continue

                    # Get the asset URL from album_assets
                    asset_result = supabase.table("album_assets") \
                        .select("asset_uri, thumbnail_uri") \
                        .eq("asset_id", asset_id) \
                        .limit(1) \
                        .execute()

                    if not asset_result.data:
                        processed += 1
                        continue

                    image_url = asset_result.data[0].get("thumbnail_uri") or asset_result.data[0].get("asset_uri")
                    if not image_url or not image_url.startswith("http"):
                        processed += 1
                        continue

                    # Skip videos
                    lower_url = image_url.lower()
                    if any(lower_url.endswith(ext) for ext in ['.mov', '.mp4', '.avi', '.mkv', '.webm']):
                        # Mark as from video
                        supabase.table("face_embeddings") \
                            .update({"is_from_video": True}) \
                            .eq("id", face["id"]) \
                            .execute()
                        processed += 1
                        continue

                    # Download the image
                    response = await client.get(image_url, timeout=30.0)
                    if response.status_code != 200:
                        processed += 1
                        continue

                    # Load image (thumbnails from storage are already EXIF-corrected)
                    img = Image.open(io.BytesIO(response.content))

                    # Parse bounding box
                    x = bounding_box.get("x", 0)
                    y = bounding_box.get("y", 0)
                    width = bounding_box.get("width", 100)
                    height = bounding_box.get("height", 100)

                    # Add padding (30%)
                    padding = 0.3
                    pad_x = width * padding
                    pad_y = height * padding

                    left = max(0, int(x - pad_x))
                    top = max(0, int(y - pad_y))
                    right = min(img.width, int(x + width + pad_x))
                    bottom = min(img.height, int(y + height + pad_y))

                    # Crop
                    face_crop = img.crop((left, top, right, bottom))

                    # Resize to thumbnail
                    face_crop.thumbnail((150, 150), Image.Resampling.LANCZOS)

                    # Convert to JPEG bytes
                    buffer = io.BytesIO()
                    if face_crop.mode in ('RGBA', 'P'):
                        face_crop = face_crop.convert('RGB')
                    face_crop.save(buffer, format='JPEG', quality=85)
                    buffer.seek(0)
                    image_bytes = buffer.getvalue()

                    # Upload to storage
                    filename = f"{user_id}/{asset_id}_{face_index}.jpg"
                    supabase.storage.from_('face-thumbnails').upload(
                        filename,
                        image_bytes,
                        file_options={"content-type": "image/jpeg", "upsert": "true"}
                    )

                    # Get public URL and update database
                    public_url = supabase.storage.from_('face-thumbnails').get_public_url(filename)

                    supabase.table("face_embeddings") \
                        .update({"face_thumbnail_url": public_url}) \
                        .eq("id", face["id"]) \
                        .execute()

                    processed += 1

                    # Update progress every 5 faces
                    if processed % 5 == 0:
                        progress = 10 + int((processed / total) * 85)
                        update_job(job_id, status="processing", progress=progress, processed=processed, total=total)

                except Exception as e:
                    import logging
                    logging.error(f"Failed to process face {face.get('id')}: {e}")
                    processed += 1
                    continue

        update_job(job_id, status="completed", progress=100, processed=processed, total=total)

    except Exception as e:
        import logging
        logging.error(f"Backfill thumbnails failed: {e}")
        update_job(job_id, status="failed", error_message=str(e))


@router.post("/tag-sync/preview", response_model=TagSyncPreviewResponse)
async def preview_tag_sync(
    request: TagSyncPreviewRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Preview manual tag to AI detection bounding box matching.

    Calls the tag-sync-api edge function which handles the database queries.

    **Matching Algorithm:**
    - For each manual tag, finds the AI detection with highest IoU overlap
    - IoU >= threshold (default 0.3): marked as "matched"
    - IoU between 0.1 and threshold: marked as "low_confidence"
    - IoU < 0.1: no match recorded

    **Response includes:**
    - Per-asset breakdown with bounding boxes for visualization
    - Summary statistics (total tags, detections, matched, unmatched)
    """
    from api.config import settings
    import httpx
    import logging

    try:
        # Call the edge function instead of direct DB access
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.supabase_url}/functions/v1/tag-sync-api/preview",
                headers={
                    "Authorization": f"Bearer {settings.supabase_service_role_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "user_id": user_id,
                    "asset_ids": request.asset_ids,
                    "iou_threshold": request.iou_threshold,
                    "limit": request.limit,
                },
                timeout=30.0
            )

            logging.info(f"Edge function response status: {response.status_code}")

            if response.status_code != 200:
                error_detail = response.text
                logging.error(f"Edge function error: {error_detail}")
                raise HTTPException(status_code=response.status_code, detail=error_detail)

            try:
                data = response.json()
                logging.info(f"Edge function returned {len(data.get('assets', []))} assets")
            except Exception as parse_err:
                logging.error(f"Failed to parse edge function response: {parse_err}")
                logging.error(f"Response text: {response.text[:500]}")
                raise

            # Convert response to pydantic models
            assets = []
            for asset in data.get("assets", []):
                manual_tags = [
                    ManualTag(
                        tag_id=t["tag_id"],
                        contact_id=t["contact_id"],
                        contact_name=t.get("contact_name"),
                        bounding_box=BoundingBox(**t["bounding_box"]) if t.get("bounding_box") else BoundingBox(x=0, y=0, width=0, height=0)
                    ) for t in asset.get("manual_tags", [])
                ]
                ai_detections = [
                    AIDetection(
                        face_id=d["face_id"],
                        bounding_box=BoundingBox(**d["bounding_box"]) if d.get("bounding_box") else BoundingBox(x=0, y=0, width=0, height=0),
                        cluster_id=d.get("cluster_id"),
                        thumbnail_url=d.get("thumbnail_url")
                    ) for d in asset.get("ai_detections", [])
                ]
                matches = [
                    TagMatch(
                        manual_tag_id=m["manual_tag_id"],
                        ai_face_id=m["ai_face_id"],
                        iou_score=m["iou_score"],
                        status=m["status"]
                    ) for m in asset.get("matches", [])
                ]
                assets.append(AssetTagPreview(
                    asset_id=asset["asset_id"],
                    thumbnail_url=asset.get("thumbnail_url"),
                    image_width=asset.get("image_width"),
                    image_height=asset.get("image_height"),
                    metadata=asset.get("metadata"),
                    manual_tags=manual_tags,
                    ai_detections=ai_detections,
                    matches=matches
                ))

            summary = data.get("summary", {})
            return TagSyncPreviewResponse(
                assets=assets,
                summary=TagSyncSummary(
                    total_assets=summary.get("total_assets", 0),
                    total_manual_tags=summary.get("total_manual_tags", 0),
                    total_ai_faces=summary.get("total_ai_faces", 0),
                    matched=summary.get("matched", 0),
                    unmatched_manual=summary.get("unmatched_manual", 0),
                    unmatched_ai=summary.get("unmatched_ai", 0)
                )
            )

    except httpx.RequestError as e:
        logging.error(f"Tag sync preview request failed: {e}")
        raise HTTPException(status_code=503, detail=f"Edge function unavailable: {str(e)}")
    except Exception as e:
        import traceback
        logging.error(f"Tag sync preview failed: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tag-sync/apply")
async def apply_tag_sync(
    request: ApplyTagSyncRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Apply confirmed tag sync matches to link AI faces with Knox contacts.

    Calls the tag-sync-api edge function which handles the database updates.

    **What this does:**
    - For each confirmed match, links the AI face to the Knox contact
    - Updates `face_embeddings.knox_contact_id` for future recognition

    **Input:**
    - List of {manual_tag_id, ai_face_id} pairs from the preview step
    """
    from api.config import settings
    import httpx
    import logging

    try:
        # Call the edge function instead of direct DB access
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.supabase_url}/functions/v1/tag-sync-api/apply",
                headers={
                    "Authorization": f"Bearer {settings.supabase_service_role_key}",
                    "Content-Type": "application/json",
                },
                json={"user_id": user_id, "matches": request.matches},
                timeout=30.0
            )

            if response.status_code != 200:
                error_detail = response.text
                logging.error(f"Edge function error: {error_detail}")
                raise HTTPException(status_code=response.status_code, detail=error_detail)

            return response.json()

    except httpx.RequestError as e:
        logging.error(f"Tag sync apply request failed: {e}")
        raise HTTPException(status_code=503, detail=f"Edge function unavailable: {str(e)}")
    except Exception as e:
        logging.error(f"Apply tag sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
