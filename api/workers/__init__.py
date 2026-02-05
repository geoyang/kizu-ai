"""Workers for background processing."""

# Unified worker handles all job types:
# - ai_processing_jobs: embeddings, faces, objects, moments generation
# - image_processing_jobs: thumbnails, web versions
from .local_worker import UnifiedWorker, LocalWorker, run_worker

__all__ = [
    "UnifiedWorker",
    "LocalWorker",  # Backward compatibility alias
    "run_worker",
]

# Legacy Celery workers (optional - requires redis)
try:
    from .celery_app import celery_app
    from .tasks import process_image_task, batch_process_task, cluster_faces_task
    __all__.extend([
        "celery_app",
        "process_image_task",
        "batch_process_task",
        "cluster_faces_task",
    ])
except ImportError:
    # Celery/Redis not installed - local worker only
    pass
