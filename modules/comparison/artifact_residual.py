"""
Artifact residual comparator: signed residual (inspected - reference), bright/dark split,
morphological white top-hat for compact-artifact enhancement, optional strong-edge suppression,
then robust percentile normalization to [0, 1] for thresholding.

Internals use float32; residual sign is preserved until the final non-negative combined map.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from modules.base import ComparatorBase


def _robust_normalize_01(
    img: np.ndarray,
    low_pct: float,
    high_pct: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Map to [0, 1] via percentiles on valid region (for thresholding only; not applied to signed residual)."""
    arr = np.asarray(img, dtype=np.float32)
    if valid_mask is not None and valid_mask.shape == arr.shape and np.any(valid_mask):
        vals = arr[valid_mask]
    else:
        vals = arr.reshape(-1)
    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(vals, low_pct))
    hi = float(np.percentile(vals, high_pct))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32, copy=False)


def _mean_over_valid(arr: np.ndarray, vm: np.ndarray | None) -> float:
    a = np.asarray(arr, dtype=np.float32)
    if vm is not None and vm.shape == a.shape and np.any(vm):
        return float(np.mean(a[vm]))
    return float(np.mean(a))


def _odd_kernel_size(k: int) -> int:
    kk = max(3, int(k))
    if kk % 2 == 0:
        kk += 1
    return kk


def _white_tophat_ellipse(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """White top-hat: I - opening(I)."""
    x = np.asarray(img, dtype=np.float32)
    ks = _odd_kernel_size(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    return cv2.morphologyEx(x, cv2.MORPH_TOPHAT, kernel).astype(np.float32, copy=False)


def _white_tophat_repeat(
    img: np.ndarray,
    kernel_size: int,
    iterations: int,
) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    it = max(1, int(iterations))
    for _ in range(it):
        x = _white_tophat_ellipse(x, kernel_size)
    return x


def _gradient_magnitude(img: np.ndarray, ksize: int) -> np.ndarray:
    k = int(ksize) if int(ksize) % 2 == 1 else int(ksize) + 1
    k = max(3, k)
    gx = cv2.Sobel(np.asarray(img, dtype=np.float32), cv2.CV_32F, 1, 0, ksize=k)
    gy = cv2.Sobel(np.asarray(img, dtype=np.float32), cv2.CV_32F, 0, 1, ksize=k)
    return cv2.magnitude(gx, gy).astype(np.float32, copy=False)


def _dilate_mask(mask_bool: np.ndarray, kernel_px: int, iterations: int) -> np.ndarray:
    """Binary dilate on boolean mask."""
    m = np.asarray(mask_bool, dtype=np.uint8) * 255
    ks = _odd_kernel_size(kernel_px)
    k_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    it = max(0, int(iterations))
    for _ in range(it):
        m = cv2.dilate(m, k_el, iterations=1)
    return (m > 0).astype(bool)


class ArtifactResidualComparator(ComparatorBase):
    """
    Signed residual → bright/dark → optional Gaussian pre-smooth → white top-hat (× iterations)
    → max combine → optional edge suppression (off / soft / hard) → percentile norm to [0,1].
    """

    def __init__(self) -> None:
        super().__init__(name="artifact_residual")

    def run(self, reference_image, inspected_image, cfg) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.validate_config(cfg)
        self.validate_array(reference_image, "reference_image")
        self.validate_array(inspected_image, "inspected_image")
        self.validate_same_shape(reference_image, inspected_image, "reference_image", "inspected_image")

        ref = np.asarray(reference_image, dtype=np.float32)
        ins = np.asarray(inspected_image, dtype=np.float32)
        if ref.ndim != 2 or ins.ndim != 2:
            raise ValueError("artifact_residual requires single-channel 2D images.")

        pre_blur_sigma = float(self.get_param(cfg, "pre_blur_sigma", 1.0))
        top_hat_kernel_size = int(
            self.get_param(cfg, "top_hat_kernel_size", self.get_param(cfg, "tophat_kernel_size", 9))
        )
        top_hat_iterations = int(self.get_param(cfg, "top_hat_iterations", 1))
        combine_mode = str(self.get_param(cfg, "combine_mode", "max")).strip().lower()
        p_low = float(self.get_param(cfg, "norm_percentile_low", 1.0))
        p_high = float(self.get_param(cfg, "norm_percentile_high", 99.0))
        use_valid_mask = bool(self.get_param(cfg, "use_valid_mask", True))
        valid_mask = self.get_param(cfg, "valid_mask", None)

        edge_mode = str(self.get_param(cfg, "edge_mode", "hard")).strip().lower()
        edge_percentile = float(self.get_param(cfg, "edge_percentile", 90.0))
        edge_dilate_kernel = int(self.get_param(cfg, "edge_dilate_kernel", 5))
        edge_dilate_iterations = int(self.get_param(cfg, "edge_dilate_iterations", 1))
        edge_weight_on_edges = float(self.get_param(cfg, "edge_weight_on_edges", 0.25))
        edge_gradient_ksize = int(self.get_param(cfg, "edge_gradient_ksize", 3))
        edge_source = str(self.get_param(cfg, "edge_source", "inspected")).strip().lower()
        min_valid_fraction = float(self.get_param(cfg, "min_valid_fraction", 0.0))
        debug_save_intermediates = bool(self.get_param(cfg, "debug_save_intermediates", True))

        vm: np.ndarray | None = None
        if use_valid_mask and valid_mask is not None:
            cand = np.asarray(valid_mask).astype(bool)
            if cand.shape == ref.shape:
                vm = cand

        valid_frac = float(np.mean(vm.astype(np.float32))) if vm is not None else 1.0
        if vm is not None and min_valid_fraction > 0 and valid_frac < min_valid_fraction:
            # Safety: still run, but metadata flags low overlap
            low_valid_warning = True
        else:
            low_valid_warning = False

        # A. Signed residual (float; preserves sign)
        residual_signed = (ins - ref).astype(np.float32, copy=False)

        # B. Split channels (non-negative)
        r_pos = np.maximum(residual_signed, 0.0)
        r_neg = np.maximum(-residual_signed, 0.0)

        # Optional mild smoothing before top-hat (does not change sign of split channels)
        if pre_blur_sigma > 0:
            r_pos = cv2.GaussianBlur(r_pos, (0, 0), pre_blur_sigma)
            r_neg = cv2.GaussianBlur(r_neg, (0, 0), pre_blur_sigma)

        # C. White top-hat enhancement
        enhanced_pos = _white_tophat_repeat(r_pos, top_hat_kernel_size, top_hat_iterations)
        enhanced_neg = _white_tophat_repeat(r_neg, top_hat_kernel_size, top_hat_iterations)

        if combine_mode == "max":
            combined = np.maximum(enhanced_pos, enhanced_neg)
        else:
            raise ValueError(
                f"artifact_residual: unsupported combine_mode={combine_mode!r}. Expected 'max'."
            )

        combined_before_edge = combined.copy()

        # D. Strong-edge mask from inspected (default) or reference gradient magnitude
        edge_threshold_value = float("nan")
        strong_edge_fraction = 0.0
        edge_mask_after_dilate = np.zeros(ref.shape, dtype=bool)
        grad_mag = np.zeros(ref.shape, dtype=np.float32)

        if edge_mode != "off":
            grad_img = ins if edge_source != "reference" else ref
            grad_mag = _gradient_magnitude(grad_img, edge_gradient_ksize)
            if vm is not None and np.any(vm):
                gvals = grad_mag[vm]
            else:
                gvals = grad_mag.reshape(-1)
            if gvals.size > 0:
                edge_threshold_value = float(np.percentile(gvals, edge_percentile))
                strong_edge = grad_mag >= edge_threshold_value
                strong_edge_fraction = float(np.mean(strong_edge.astype(np.float32)))
                edge_mask_after_dilate = _dilate_mask(strong_edge, edge_dilate_kernel, edge_dilate_iterations)
                if vm is not None:
                    edge_mask_after_dilate = edge_mask_after_dilate & vm
            else:
                edge_mode = "off"

        combined_after_edge = combined.astype(np.float32, copy=True)
        fraction_anomaly_suppressed = 0.0

        if edge_mode == "soft" and np.any(edge_mask_after_dilate):
            w = float(np.clip(edge_weight_on_edges, 0.0, 1.0))
            combined_after_edge = np.where(edge_mask_after_dilate, combined * w, combined)
            if vm is not None:
                denom = float(np.count_nonzero(vm))
                num = float(np.count_nonzero(vm & edge_mask_after_dilate))
            else:
                denom = float(combined.size)
                num = float(np.count_nonzero(edge_mask_after_dilate))
            fraction_anomaly_suppressed = num / max(1.0, denom)
        elif edge_mode == "hard" and np.any(edge_mask_after_dilate):
            combined_after_edge = np.where(edge_mask_after_dilate, 0.0, combined)
            if vm is not None:
                denom = float(np.count_nonzero(vm))
                num = float(np.count_nonzero(vm & edge_mask_after_dilate))
            else:
                denom = float(combined.size)
                num = float(np.count_nonzero(edge_mask_after_dilate))
            fraction_anomaly_suppressed = num / max(1.0, denom)

        # E. Output for thresholding: robust [0,1] on valid region only (combined is non-negative)
        anomaly = _robust_normalize_01(combined_after_edge, p_low, p_high, vm)
        if vm is not None:
            anomaly = anomaly.copy()
            anomaly[~vm] = 0.0

        bright_artifact_mean = _mean_over_valid(enhanced_pos, vm)
        dark_artifact_mean = _mean_over_valid(enhanced_neg, vm)
        anomaly_mean = _mean_over_valid(anomaly, vm)

        metadata: Dict[str, Any] = {
            "method": self.name,
            "pre_blur_sigma": pre_blur_sigma,
            "top_hat_kernel_size": int(_odd_kernel_size(top_hat_kernel_size)),
            "top_hat_iterations": int(max(1, top_hat_iterations)),
            "combine_mode": combine_mode,
            "norm_percentile_low": p_low,
            "norm_percentile_high": p_high,
            "final_normalization": "percentile_robust_01",
            "used_valid_mask": vm is not None,
            "valid_overlap_fraction": valid_frac,
            "min_valid_fraction_threshold": min_valid_fraction,
            "low_valid_overlap_warning": low_valid_warning,
            "bright_artifact_mean": float(bright_artifact_mean),
            "dark_artifact_mean": float(dark_artifact_mean),
            "anomaly_mean_after_norm": float(anomaly_mean),
            "edge_mode": edge_mode,
            "edge_percentile": edge_percentile,
            "edge_threshold_value": edge_threshold_value,
            "strong_edge_fraction": strong_edge_fraction,
            "edge_dilate_kernel": int(_odd_kernel_size(edge_dilate_kernel)),
            "edge_dilate_iterations": int(max(0, edge_dilate_iterations)),
            "edge_weight_on_edges": edge_weight_on_edges,
            "edge_gradient_ksize": edge_gradient_ksize,
            "edge_source": edge_source,
            "fraction_anomaly_touched_by_edge_mask": float(fraction_anomaly_suppressed),
            "debug_save_intermediates": debug_save_intermediates,
        }

        if debug_save_intermediates:
            dbg: Dict[str, np.ndarray] = {
                "residual_signed": residual_signed.copy(),
                "residual_positive": r_pos.copy(),
                "residual_negative": r_neg.copy(),
                "enhanced_positive": enhanced_pos.copy(),
                "enhanced_negative": enhanced_neg.copy(),
                "combined_before_edge": combined_before_edge.copy(),
                "combined_after_edge": combined_after_edge.copy(),
                "edge_mask_dilated": edge_mask_after_dilate.astype(np.float32),
                "gradient_magnitude_inspected_or_source": grad_mag.copy(),
            }
            metadata["artifact_residual_debug_maps"] = dbg

        return anomaly, metadata
