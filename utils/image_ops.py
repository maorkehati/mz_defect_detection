from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage


def normalize_to_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32)
        info = np.iinfo(x.astype(np.int32).dtype) if False else None
        # Keep integer scale as-is; do not force [0, 1]
        return x
    return x.astype(np.float32, copy=False)


def to_grayscale(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        return x.astype(np.float32, copy=False)
    if x.ndim == 3:
        if x.shape[-1] == 1:
            return x[..., 0].astype(np.float32, copy=False)
        if x.shape[-1] in (3, 4):
            # Simple luminance-like conversion
            rgb = x[..., :3].astype(np.float32, copy=False)
            return (0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]).astype(np.float32)
    raise ValueError(f"Unsupported image shape for grayscale conversion: {x.shape}")


def gaussian_blur(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x.astype(np.float32, copy=False)
    return ndimage.gaussian_filter(x.astype(np.float32, copy=False), sigma=sigma)


def estimate_translation_phase_correlation(
    reference_image: np.ndarray,
    inspected_image: np.ndarray,
    subpixel_refinement: bool = True,
) -> Tuple[float, float, float]:
    """
    Returns:
        shift_y, shift_x, peak_response

    The returned shift is the amount that should be applied to the reference image
    so it aligns with the inspected image.
    """
    ref = reference_image.astype(np.float32, copy=False)
    ins = inspected_image.astype(np.float32, copy=False)

    ref = ref - np.mean(ref)
    ins = ins - np.mean(ins)

    eps = 1e-12
    F_ref = np.fft.fft2(ref)
    F_ins = np.fft.fft2(ins)

    cross_power = F_ins * np.conj(F_ref)
    cross_power /= np.maximum(np.abs(cross_power), eps)

    corr = np.fft.ifft2(cross_power)
    corr = np.abs(corr)

    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    peak_y, peak_x = peak_idx
    peak_response = float(corr[peak_y, peak_x])

    h, w = corr.shape

    shift_y = float(peak_y)
    shift_x = float(peak_x)

    if shift_y > h // 2:
        shift_y -= h
    if shift_x > w // 2:
        shift_x -= w

    if subpixel_refinement:
        shift_y += _quadratic_subpixel_offset_1d(corr[:, peak_x], peak_y)
        shift_x += _quadratic_subpixel_offset_1d(corr[peak_y, :], peak_x)

    return shift_y, shift_x, peak_response


def _quadratic_subpixel_offset_1d(arr: np.ndarray, peak_idx: int) -> float:
    n = arr.shape[0]
    left_idx = (peak_idx - 1) % n
    right_idx = (peak_idx + 1) % n

    y1 = float(arr[left_idx])
    y2 = float(arr[peak_idx])
    y3 = float(arr[right_idx])

    denom = (y1 - 2.0 * y2 + y3)
    if abs(denom) < 1e-12:
        return 0.0

    offset = 0.5 * (y1 - y3) / denom
    offset = float(np.clip(offset, -1.0, 1.0))
    return offset


def apply_shift(
    x: np.ndarray,
    shift_y: float,
    shift_x: float,
    order: int = 1,
    mode: str = "reflect",
) -> np.ndarray:
    return ndimage.shift(
        x.astype(np.float32, copy=False),
        shift=(shift_y, shift_x),
        order=order,
        mode=mode,
        prefilter=(order > 1),
    )