"""Device detection and management utilities."""

import logging
import platform
from typing import Optional

logger = logging.getLogger(__name__)

_cached_device: Optional[str] = None


def get_device() -> str:
    """
    Get the best available compute device.

    Returns:
        'cuda' for NVIDIA GPU
        'mps' for Apple Silicon
        'cpu' as fallback
    """
    global _cached_device

    if _cached_device is not None:
        return _cached_device

    import torch

    if torch.cuda.is_available():
        _cached_device = "cuda"
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        _cached_device = "mps"
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        _cached_device = "cpu"
        logger.info("Using CPU")

    return _cached_device


def get_device_info() -> dict:
    """Get detailed device information."""
    import torch

    info = {
        "device": get_device(),
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
            "gpu_count": torch.cuda.device_count(),
        })
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info.update({
            "mps_available": True,
            "processor": platform.processor(),
        })

    return info


def clear_device_cache() -> None:
    """Clear GPU memory cache."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have explicit cache clearing
        pass
