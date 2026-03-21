from __future__ import annotations

from .absolute_difference import AbsoluteDifferenceComparator

__all__ = ["AbsoluteDifferenceComparator"]

try:
    from .gradient_difference import GradientDifferenceComparator

    __all__.append("GradientDifferenceComparator")
except Exception:
    pass

try:
    from .ssim_comparator import SsimComparator

    __all__.append("SsimComparator")
except Exception:
    # Optional dependency for this comparator may be unavailable.
    pass