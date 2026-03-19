from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage


def _structure_8() -> np.ndarray:
    return np.ones((3, 3), dtype=bool)


def binary_opening(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if iterations <= 0:
        return mask_bool
    return ndimage.binary_opening(mask_bool, structure=_structure_8(), iterations=iterations)


def binary_closing(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if iterations <= 0:
        return mask_bool
    return ndimage.binary_closing(mask_bool, structure=_structure_8(), iterations=iterations)


def connected_components_with_stats(mask: np.ndarray) -> Tuple[np.ndarray, int, List[Dict[str, int]]]:
    mask_bool = mask.astype(bool)
    labels, num = ndimage.label(mask_bool, structure=_structure_8())

    stats: List[Dict[str, int]] = []
    if num == 0:
        return labels, 0, stats

    objects = ndimage.find_objects(labels)
    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        ys, xs = np.where(labels[slc] == label_id)
        if ys.size == 0:
            continue

        y0 = slc[0].start + int(ys.min())
        y1 = slc[0].start + int(ys.max())
        x0 = slc[1].start + int(xs.min())
        x1 = slc[1].start + int(xs.max())
        area = int(ys.size)

        stats.append(
            {
                "label": int(label_id),
                "area": area,
                "y_min": y0,
                "y_max": y1,
                "x_min": x0,
                "x_max": x1,
            }
        )

    return labels, num, stats


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask.astype(bool)

    labels, num, stats = connected_components_with_stats(mask)
    if num == 0:
        return mask.astype(bool)

    out = np.zeros_like(mask, dtype=bool)
    for stat in stats:
        if stat["area"] >= min_area:
            out |= (labels == stat["label"])
    return out