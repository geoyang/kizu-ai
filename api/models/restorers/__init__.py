"""Image restoration model implementations."""

# Compatibility shim: basicsr references torchvision.transforms.functional_tensor
# which was removed in torchvision >= 0.18. Redirect to functional.
import sys
import types
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ModuleNotFoundError:
    import torchvision.transforms.functional as _F
    _mod = types.ModuleType("torchvision.transforms.functional_tensor")
    _mod.__dict__.update({k: v for k, v in _F.__dict__.items()})
    sys.modules["torchvision.transforms.functional_tensor"] = _mod

from .realesrgan_model import RealESRGANRestorer
from .gfpgan_model import GFPGANRestorer

__all__ = ["RealESRGANRestorer", "GFPGANRestorer"]
