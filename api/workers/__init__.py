"""Workers for background processing."""

# Local worker (recommended - uses Supabase Realtime)
from .local_worker import LocalWorker, run_worker

__all__ = [
    "LocalWorker",
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
