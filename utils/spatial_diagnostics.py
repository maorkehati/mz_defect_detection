from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


def border_distance_for_bbox(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> float:
    left = float(x)
    top = float(y)
    right = float(max(0, img_w - (x + w)))
    bottom = float(max(0, img_h - (y + h)))
    return float(min(left, top, right, bottom))


def border_distance_for_point(x: int, y: int, img_w: int, img_h: int) -> float:
    left = float(x)
    top = float(y)
    right = float(max(0, img_w - 1 - x))
    bottom = float(max(0, img_h - 1 - y))
    return float(min(left, top, right, bottom))


def connected_component_stats(binary_mask: np.ndarray, border_margin_px: int = 1) -> tuple[int, int]:
    """Return (component_count, border_touching_component_count) for a binary mask."""
    bm = np.asarray(binary_mask).astype(bool)
    if bm.size == 0:
        return 0, 0
    num_labels, labels = cv2.connectedComponents(bm.astype(np.uint8))
    comp_count = int(max(0, num_labels - 1))
    if comp_count == 0:
        return 0, 0
    m = int(max(0, border_margin_px))
    border = np.zeros_like(bm, dtype=bool)
    border[: m + 1, :] = True
    border[-(m + 1) :, :] = True
    border[:, : m + 1] = True
    border[:, -(m + 1) :] = True
    touched = set(int(v) for v in np.unique(labels[border]) if int(v) > 0)
    return comp_count, int(len(touched))


def border_band_and_center_means(residual_map: np.ndarray, valid_mask: Optional[np.ndarray], band_px: int = 6) -> tuple[Optional[float], Optional[float]]:
    r = np.asarray(residual_map, dtype=np.float32)
    vm = np.ones_like(r, dtype=bool) if valid_mask is None else np.asarray(valid_mask).astype(bool)
    if vm.shape != r.shape or not np.any(vm):
        return None, None
    dist = cv2.distanceTransform(vm.astype(np.uint8), cv2.DIST_L2, 3)
    border_band = np.logical_and(vm, dist <= float(max(1, band_px)))
    center = np.logical_and(vm, dist >= float(max(2, band_px * 2)))
    border_mean = float(np.mean(np.abs(r[border_band]))) if np.any(border_band) else None
    center_mean = float(np.mean(np.abs(r[center]))) if np.any(center) else None
    return border_mean, center_mean


def nearest_gt(gt_points: List[Tuple[int, int]], x: float, y: float) -> tuple[Optional[int], Optional[float]]:
    if not gt_points:
        return None, None
    best_id: Optional[int] = None
    best_d: Optional[float] = None
    for i, (gx, gy) in enumerate(gt_points, start=1):
        d = float(np.hypot(float(gx) - float(x), float(gy) - float(y)))
        if best_d is None or d < best_d:
            best_d = d
            best_id = i
    return best_id, best_d


def local_max_around_point(img: np.ndarray, x: int, y: int, radius: int = 5) -> Optional[float]:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 2:
        return None
    h, w = arr.shape
    if not (0 <= x < w and 0 <= y < h):
        return None
    r = int(max(0, radius))
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    return float(np.max(arr[y0:y1, x0:x1]))


def percentile_in_valid_region(img: np.ndarray, value: float, valid_mask: Optional[np.ndarray]) -> Optional[float]:
    arr = np.asarray(img, dtype=np.float32)
    vals = arr.reshape(-1)
    if valid_mask is not None:
        vm = np.asarray(valid_mask).astype(bool)
        if vm.shape == arr.shape and np.any(vm):
            vals = arr[vm]
    if vals.size == 0:
        return None
    return float(100.0 * np.mean(vals <= float(value)))


def top_peaks_nms(
    anomaly_map: np.ndarray,
    top_k: int = 20,
    min_spacing_px: int = 6,
    valid_mask: Optional[np.ndarray] = None,
) -> List[Tuple[int, int, float]]:
    arr = np.asarray(anomaly_map, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0 or top_k <= 0:
        return []
    work = np.asarray(arr, dtype=np.float32).copy()
    if valid_mask is not None:
        vm = np.asarray(valid_mask).astype(bool)
        if vm.shape == work.shape:
            work[~vm] = -np.inf
    ys, xs = np.unravel_index(np.argsort(work, axis=None)[::-1], work.shape)
    peaks: List[Tuple[int, int, float]] = []
    r2 = float(max(0, min_spacing_px) ** 2)
    for y, x in zip(ys.tolist(), xs.tolist()):
        v = float(work[y, x])
        if not np.isfinite(v):
            continue
        keep = True
        for px, py, _ in peaks:
            if (float(x - px) ** 2 + float(y - py) ** 2) < r2:
                keep = False
                break
        if keep:
            peaks.append((int(x), int(y), v))
        if len(peaks) >= top_k:
            break
    return peaks

