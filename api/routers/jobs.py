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
