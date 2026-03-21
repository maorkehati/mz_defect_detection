from __future__ import annotations

from .mad_threshold import MadThresholding

__all__ = ["MadThresholding"]

try:
    from .otsu_threshold import OtsuThresholding

    __all__.append("OtsuThresholding")
except Exception:
    pass

try:
    from .fixed_threshold import FixedThresholding

    __all__.append("FixedThresholding")
except Exception:
    pass