"""API routers."""

from .search import router as search_router
from .process import router as process_router
from .faces import router as faces_router
from .jobs import router as jobs_router
from .health import router as health_router

__all__ = [
    "search_router",
    "process_router",
    "faces_router",
    "jobs_router",
    "health_router",
]
