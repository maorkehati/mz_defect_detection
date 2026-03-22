"""
Ground-truth point coverage vs. final defect mask.

GT coordinates ``(x, y)`` follow the inspected-image convention used in
``defects locations.txt`` and visualization: **column x, row y**, matching
``mask[y, x]`` on pipeline arrays (same shape as inspected / defect mask).

Coverage:
- **exact**: GT pixel lies on a positive (True) defect mask pixel.
- **within_radius**: Euclidean distance from GT to the nearest detected pixel
  is <= ``radius_px`` (tolerates small boundary misalignment).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


@dataclass(frozen=True)
class GTPointCoverageMetrics:
    gt_total: int
    gt_covered_exact: int
    gt_covered_within_radius: int
    coverage_fraction_exact: float
    coverage_fraction_within_radius: float
    radius_px: float


def _distance_to_nearest_defect(mask: np.ndarray) -> np.ndarray:
    """
    Per-pixel Euclidean distance to the nearest True defect pixel.
    Same shape as ``mask``; distance is 0 on defect pixels.
    """
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape[:2]
    if not np.any(m):
        return np.full((h, w), np.inf, dtype=np.float32)

    if _HAS_CV2:
        src = np.where(m, 0, 255).astype(np.uint8)
        return cv2.distanceTransform(src, cv2.DIST_L2, 5).astype(np.float32)

    # Fallback: full map via distance to nearest defect pixel (vectorized over image)
    ys, xs = np.where(m)
    if xs.size == 0:
        return np.full((h, w), np.inf, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # d[h,w] = min_k hypot(xx - xk, yy - yk)
    dmin = np.full((h, w), np.inf, dtype=np.float32)
    for xk, yk in zip(xs.astype(np.float32), ys.astype(np.float32)):
        d = np.hypot(xx - xk, yy - yk)
        dmin = np.minimum(dmin, d)
    dmin = np.where(m, 0.0, dmin)
    return dmin


def compute_gt_point_coverage_metrics(
    defect_mask: np.ndarray | None,
    gt_points_xy: Sequence[Tuple[int, int]],
    *,
    radius_px: float = 5.0,
) -> GTPointCoverageMetrics:
    """
    Evaluate how many listed GT points are covered by the binary **final** defect mask.

    Parameters
    ----------
    defect_mask:
        Boolean or 0/1 array, same height × width as inspected (``H, W``).
    gt_points_xy:
        List of ``(x, y)`` in image coordinates (column, row), same as the defects file.
    radius_px:
        Max Euclidean distance from a GT point to any positive mask pixel to count as
        "covered with radius" (default 5).
    """
    pts = list(gt_points_xy)
    n = len(pts)
    if defect_mask is None or n == 0:
        return GTPointCoverageMetrics(
            gt_total=n,
            gt_covered_exact=0,
            gt_covered_within_radius=0,
            coverage_fraction_exact=0.0,
            coverage_fraction_within_radius=0.0,
            radius_px=float(radius_px),
        )

    m = np.asarray(defect_mask).astype(bool)
    h, w = m.shape[:2]
    dist_map = _distance_to_nearest_defect(m)
    r = float(max(0.0, radius_px))

    exact = 0
    within = 0
    for gx, gy in pts:
        xi, yi = int(gx), int(gy)
        if not (0 <= yi < h and 0 <= xi < w):
            continue
        if m[yi, xi]:
            exact += 1
            within += 1
            continue
        d = float(dist_map[yi, xi])
        if d <= r:
            within += 1

    return GTPointCoverageMetrics(
        gt_total=n,
        gt_covered_exact=int(exact),
        gt_covered_within_radius=int(within),
        coverage_fraction_exact=float(exact) / float(n),
        coverage_fraction_within_radius=float(within) / float(n),
        radius_px=r,
    )
