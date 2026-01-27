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

    Compares manual face tags from the mobile app with AI-detected faces
    using IoU (Intersection over Union) to find matches.
    """
    from api.config import settings
    from supabase import create_client
    import logging

    try:
        supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

        # Get assets with manual tags
        tags_query = supabase.table("asset_tags") \
            .select("id, asset_id, contact_id, bounding_box, contacts!inner(id, display_name)") \
            .eq("user_id", user_id) \
            .not_.is_("bounding_box", "null")

        if request.asset_ids:
            tags_query = tags_query.in_("asset_id", request.asset_ids)

        tags_result = tags_query.limit(request.limit * 10).execute()
        manual_tags_data = tags_result.data or []

        if not manual_tags_data:
            return TagSyncPreviewResponse(
                assets=[],
                summary=TagSyncSummary(
                    total_assets=0,
                    total_manual_tags=0,
                    total_ai_faces=0,
                    matched=0,
                    unmatched_manual=0,
                    unmatched_ai=0
                )
            )

        # Group manual tags by asset_id
        asset_manual_tags: Dict[str, List[Dict]] = {}
        for tag in manual_tags_data:
            aid = tag["asset_id"]
            if aid not in asset_manual_tags:
                asset_manual_tags[aid] = []
            asset_manual_tags[aid].append(tag)

        asset_ids = list(asset_manual_tags.keys())[:request.limit]

        # Get AI detections for these assets
        ai_result = supabase.table("face_embeddings") \
            .select("id, asset_id, bounding_box, cluster_id, face_thumbnail_url") \
            .eq("user_id", user_id) \
            .in_("asset_id", asset_ids) \
            .not_.is_("bounding_box", "null") \
            .execute()

        ai_faces_data = ai_result.data or []

        # Group AI faces by asset_id
        asset_ai_faces: Dict[str, List[Dict]] = {}
        for face in ai_faces_data:
            aid = face["asset_id"]
            if aid not in asset_ai_faces:
                asset_ai_faces[aid] = []
            asset_ai_faces[aid].append(face)

        # Get asset thumbnails
        assets_result = supabase.table("album_assets") \
            .select("asset_id, thumbnail_uri") \
            .in_("asset_id", asset_ids) \
            .execute()

        asset_thumbnails = {a["asset_id"]: a.get("thumbnail_uri") for a in (assets_result.data or [])}

        # Build preview response
        preview_assets = []
        total_manual = 0
        total_ai = 0
        total_matched = 0

        for asset_id in asset_ids:
            manual_tags = asset_manual_tags.get(asset_id, [])
            ai_faces = asset_ai_faces.get(asset_id, [])

            # Build ManualTag objects
            manual_tag_objs = []
            for tag in manual_tags:
                bbox = tag.get("bounding_box", {})
                contact_info = tag.get("contacts", {})
                manual_tag_objs.append(ManualTag(
                    tag_id=tag["id"],
                    contact_id=tag["contact_id"],
                    contact_name=contact_info.get("display_name") if contact_info else None,
                    bounding_box=BoundingBox(
                        x=bbox.get("x", 0),
                        y=bbox.get("y", 0),
                        width=bbox.get("width", 0),
                        height=bbox.get("height", 0)
                    )
                ))

            # Build AIDetection objects
            ai_detection_objs = []
            for face in ai_faces:
                bbox = face.get("bounding_box", {})
                ai_detection_objs.append(AIDetection(
                    face_id=face["id"],
                    bounding_box=BoundingBox(
                        x=bbox.get("x", 0),
                        y=bbox.get("y", 0),
                        width=bbox.get("width", 0),
                        height=bbox.get("height", 0)
                    ),
                    cluster_id=face.get("cluster_id"),
                    thumbnail_url=face.get("face_thumbnail_url")
                ))

            # Calculate matches using IoU
            matches = []
            matched_manual_ids = set()
            matched_ai_ids = set()

            for manual_tag in manual_tags:
                best_match = None
                best_iou = 0.0
                manual_bbox = manual_tag.get("bounding_box", {})

                for ai_face in ai_faces:
                    if ai_face["id"] in matched_ai_ids:
                        continue
                    ai_bbox = ai_face.get("bounding_box", {})
                    iou = calculate_iou(manual_bbox, ai_bbox)

                    if iou > best_iou:
                        best_iou = iou
                        best_match = ai_face

                if best_match and best_iou >= request.iou_threshold:
                    matches.append(TagMatch(
                        manual_tag_id=manual_tag["id"],
                        ai_face_id=best_match["id"],
                        iou_score=round(best_iou, 3),
                        status="matched"
                    ))
                    matched_manual_ids.add(manual_tag["id"])
                    matched_ai_ids.add(best_match["id"])
                    total_matched += 1
                elif best_match and best_iou > 0.1:
                    matches.append(TagMatch(
                        manual_tag_id=manual_tag["id"],
                        ai_face_id=best_match["id"],
                        iou_score=round(best_iou, 3),
                        status="low_confidence"
                    ))

            total_manual += len(manual_tags)
            total_ai += len(ai_faces)

            preview_assets.append(AssetTagPreview(
                asset_id=asset_id,
                thumbnail_url=asset_thumbnails.get(asset_id),
                manual_tags=manual_tag_objs,
                ai_detections=ai_detection_objs,
                matches=matches
            ))

        return TagSyncPreviewResponse(
            assets=preview_assets,
            summary=TagSyncSummary(
                total_assets=len(preview_assets),
                total_manual_tags=total_manual,
                total_ai_faces=total_ai,
                matched=total_matched,
                unmatched_manual=total_manual - total_matched,
                unmatched_ai=total_ai - total_matched
            )
        )

    except Exception as e:
        logging.error(f"Tag sync preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tag-sync/apply")
async def apply_tag_sync(
    request: ApplyTagSyncRequest,
    user_id: str = Depends(get_current_user_id)
):
    """
    Apply confirmed tag sync matches to link AI faces with Knox contacts.

    This will update the knox_contact_id on matched face_embeddings.
    """
    from api.config import settings
    from supabase import create_client
    import logging

    try:
        supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

        applied = 0
        errors = []

        for match in request.matches:
            manual_tag_id = match.get("manual_tag_id")
            ai_face_id = match.get("ai_face_id")

            if not manual_tag_id or not ai_face_id:
                continue

            # Get the contact_id from the manual tag
            tag_result = supabase.table("asset_tags") \
                .select("contact_id") \
                .eq("id", manual_tag_id) \
                .eq("user_id", user_id) \
                .single() \
                .execute()

            if not tag_result.data:
                errors.append(f"Tag {manual_tag_id} not found")
                continue

            contact_id = tag_result.data["contact_id"]

            # Update the face embedding with the knox_contact_id
            update_result = supabase.table("face_embeddings") \
                .update({"knox_contact_id": contact_id}) \
                .eq("id", ai_face_id) \
                .eq("user_id", user_id) \
                .execute()

            if update_result.data:
                applied += 1
            else:
                errors.append(f"Failed to update face {ai_face_id}")

        return {
            "status": "completed",
            "applied": applied,
            "errors": errors if errors else None
        }

    except Exception as e:
        logging.error(f"Apply tag sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
