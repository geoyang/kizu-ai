"""Celery application configuration."""

from celery import Celery
from api.config import settings

celery_app = Celery(
    "kizu_ai",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["api.workers.tasks"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task settings
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3000,  # 50 min soft limit

    # Worker settings
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_concurrency=1,  # Single worker (AI models need memory)

    # Result settings
    result_expires=86400,  # Results expire after 24 hours
)

# Task routing (can be expanded for multiple queues)
celery_app.conf.task_routes = {
    "api.workers.tasks.process_image_task": {"queue": "default"},
    "api.workers.tasks.batch_process_task": {"queue": "batch"},
    "api.workers.tasks.cluster_faces_task": {"queue": "batch"},
}
