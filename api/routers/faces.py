"""Face recognition API router."""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Optional
import uuid

from api.schemas import (
    ClusterFacesRequest,
    AssignClusterRequest,
    ProcessingJobResponse,
)
from api.schemas.responses import (
    JobStatus,
    FaceClusterResponse,
    FaceClustersListResponse,
)
from api.services import FaceService, ClusteringService
from api.dependencies import (
    get_face_service,
    get_clustering_service,
    get_current_user_id
)

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
        user_id=user_id
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to assign cluster")

    return {"status": "assigned", "cluster_id": cluster_id}


@router.post("/clusters/merge")
async def merge_clusters(
    cluster_ids: List[str],
    clustering_service: ClusteringService = Depends(get_clustering_service),
    user_id: str = Depends(get_current_user_id)
):
    """Merge multiple clusters into one."""
    if len(cluster_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 clusters to merge"
        )

    new_cluster_id = await clustering_service.merge_clusters(
        cluster_ids, user_id
    )

    return {"status": "merged", "new_cluster_id": new_cluster_id}
