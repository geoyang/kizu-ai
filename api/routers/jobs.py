"""Job status API router."""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

from api.schemas.responses import ProcessingJobResponse, JobStatus
from api.dependencies import get_current_user_id

router = APIRouter(prefix="/jobs", tags=["jobs"])

# In-memory job store for demo (use Redis/DB in production)
_jobs: dict = {}


@router.get("/{job_id}", response_model=ProcessingJobResponse)
async def get_job_status(
    job_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """Get the status of a processing job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]

    if job.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    return ProcessingJobResponse(**job)


@router.delete("/{job_id}")
async def cancel_job(
    job_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """Cancel a pending or running job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]

    if job.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    if job.get("status") == JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail="Cannot cancel completed job"
        )

    # Mark as cancelled
    _jobs[job_id]["status"] = "cancelled"

    return {"status": "cancelled", "job_id": job_id}


@router.get("")
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 20,
    user_id: str = Depends(get_current_user_id)
):
    """List all jobs for the current user."""
    user_jobs = [
        j for j in _jobs.values()
        if j.get("user_id") == user_id
    ]

    if status:
        user_jobs = [j for j in user_jobs if j.get("status") == status]

    # Sort by created_at descending
    user_jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return {
        "jobs": user_jobs[:limit],
        "total": len(user_jobs)
    }


def update_job(job_id: str, **updates):
    """Helper to update job status (called from workers)."""
    if job_id in _jobs:
        _jobs[job_id].update(updates)


def create_job(job_id: str, user_id: str, job_type: str, **kwargs):
    """Helper to create a new job record."""
    from datetime import datetime

    _jobs[job_id] = {
        "job_id": job_id,
        "user_id": user_id,
        "job_type": job_type,
        "status": JobStatus.PENDING,
        "progress": 0,
        "processed": 0,
        "total": 0,
        "created_at": datetime.utcnow().isoformat(),
        **kwargs
    }


# Queue stats endpoints for Activity Dashboard
@router.get("/queue/stats")
async def get_queue_stats(
    user_id: str = Depends(get_current_user_id)
):
    """
    Get counts of jobs by status from the processing_queue table.

    Returns counts for: pending, processing, completed, failed
    """
    try:
        from supabase import create_client
        from api.config import settings

        supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

        # Get counts by status
        stats = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "total": 0
        }

        result = supabase.table('processing_queue')\
            .select('status')\
            .eq('user_id', user_id)\
            .execute()

        for row in result.data:
            status = row.get('status', 'pending')
            if status in stats:
                stats[status] += 1
            stats["total"] += 1

        return stats

    except Exception as e:
        return {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "total": 0,
            "error": str(e)
        }


@router.get("/queue/active")
async def get_active_jobs(
    limit: int = 10,
    user_id: str = Depends(get_current_user_id)
):
    """
    Get currently processing jobs from the queue.

    Returns active jobs with worker info and timing.
    """
    try:
        from supabase import create_client
        from api.config import settings

        supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

        result = supabase.table('processing_queue')\
            .select('id, asset_id, status, worker_id, started_at, created_at')\
            .eq('user_id', user_id)\
            .eq('status', 'processing')\
            .order('started_at', desc=True)\
            .limit(limit)\
            .execute()

        return {
            "jobs": result.data,
            "count": len(result.data)
        }

    except Exception as e:
        return {
            "jobs": [],
            "count": 0,
            "error": str(e)
        }


@router.get("/queue/recent")
async def get_recent_jobs(
    limit: int = 50,
    status: Optional[str] = None,
    user_id: str = Depends(get_current_user_id)
):
    """
    Get recently processed/updated jobs from the queue.

    Returns jobs with processing results and timing info.
    """
    try:
        from supabase import create_client
        from api.config import settings

        supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

        query = supabase.table('processing_queue')\
            .select('id, asset_id, status, worker_id, priority, created_at, started_at, completed_at, result')\
            .eq('user_id', user_id)\
            .order('completed_at', desc=True, nullsfirst=False)\
            .limit(limit)

        if status:
            query = query.eq('status', status)

        result = query.execute()

        return {
            "jobs": result.data,
            "count": len(result.data)
        }

    except Exception as e:
        return {
            "jobs": [],
            "count": 0,
            "error": str(e)
        }


@router.get("/queue/workers")
async def get_worker_status(
    user_id: str = Depends(get_current_user_id)
):
    """
    Get status of active workers.

    Returns list of workers with their current job info.
    """
    try:
        from supabase import create_client
        from api.config import settings

        supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

        # Get distinct workers that are currently processing
        result = supabase.table('processing_queue')\
            .select('worker_id, asset_id, started_at')\
            .eq('user_id', user_id)\
            .eq('status', 'processing')\
            .not_.is_('worker_id', 'null')\
            .execute()

        # Group by worker
        workers = {}
        for row in result.data:
            worker_id = row.get('worker_id')
            if worker_id:
                workers[worker_id] = {
                    "worker_id": worker_id,
                    "current_asset": row.get('asset_id'),
                    "started_at": row.get('started_at'),
                    "status": "active"
                }

        return {
            "workers": list(workers.values()),
            "count": len(workers)
        }

    except Exception as e:
        return {
            "workers": [],
            "count": 0,
            "error": str(e)
        }
