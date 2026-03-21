from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from modules.base import ComparatorBase


def _robust_normalize(
    img: np.ndarray,
    low_pct: float,
    high_pct: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if valid_mask is not None and valid_mask.shape == arr.shape and np.any(valid_mask):
        vals = arr[valid_mask]
    else:
        vals = arr.reshape(-1)
    lo = float(np.percentile(vals, low_pct))
    hi = float(np.percentile(vals, high_pct))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32, copy=False)


def _gradient_magnitude(img: np.ndarray, ksize: int) -> np.ndarray:
    gx = cv2.Sobel(np.asarray(img, dtype=np.float32), cv2.CV_32F, 1, 0, ksize=int(ksize))
    gy = cv2.Sobel(np.asarray(img, dtype=np.float32), cv2.CV_32F, 0, 1, ksize=int(ksize))
    return cv2.magnitude(gx, gy).astype(np.float32, copy=False)


def _mean_over_region(arr: np.ndarray, vm: np.ndarray | None) -> float:
    """Mean over valid_mask if provided and non-empty; else full array."""
    a = np.asarray(arr, dtype=np.float32)
    if vm is not None and vm.shape == a.shape and np.any(vm):
        return float(np.mean(a[vm]))
    return float(np.mean(a))


class GradientDifferenceComparator(ComparatorBase):
    def __init__(self) -> None:
        super().__init__(name="gradient_difference")

    def run(self, reference_image, inspected_image, cfg):
        self.validate_config(cfg)
        self.validate_array(reference_image, "reference_image")
        self.validate_array(inspected_image, "inspected_image")
        self.validate_same_shape(reference_image, inspected_image, "reference_image", "inspected_image")

        ref = np.asarray(reference_image, dtype=np.float32)
        ins = np.asarray(inspected_image, dtype=np.float32)
        if ref.ndim != 2 or ins.ndim != 2:
            raise ValueError("gradient_difference requires single-channel 2D images.")

        pre_blur_sigma = float(self.get_param(cfg, "pre_blur_sigma", 1.0))
        post_blur_sigma = float(self.get_param(cfg, "post_blur_sigma", 1.0))
        gradient_ksize = int(self.get_param(cfg, "gradient_ksize", 3))
        p_low = float(self.get_param(cfg, "norm_percentile_low", 1.0))
        p_high = float(self.get_param(cfg, "norm_percentile_high", 99.0))
        use_valid_mask = bool(self.get_param(cfg, "use_valid_mask", True))
        edge_suppression_enabled = bool(self.get_param(cfg, "edge_suppression_enabled", False))
        edge_percentile = float(self.get_param(cfg, "edge_percentile", 85.0))
        edge_weight_on_edges = float(self.get_param(cfg, "edge_weight_on_edges", 0.35))
        valid_mask = self.get_param(cfg, "valid_mask", None)

        vm = None
        if use_valid_mask and valid_mask is not None:
            cand = np.asarray(valid_mask).astype(bool)
            if cand.shape == ref.shape:
                vm = cand

        if pre_blur_sigma > 0:
            ref_f = cv2.GaussianBlur(ref, (0, 0), pre_blur_sigma)
            ins_f = cv2.GaussianBlur(ins, (0, 0), pre_blur_sigma)
        else:
            ref_f = ref
            ins_f = ins

        grad_ref = _gradient_magnitude(ref_f, gradient_ksize)
        grad_ins = _gradient_magnitude(ins_f, gradient_ksize)

        if post_blur_sigma > 0:
            grad_ref = cv2.GaussianBlur(grad_ref, (0, 0), post_blur_sigma)
            grad_ins = cv2.GaussianBlur(grad_ins, (0, 0), post_blur_sigma)

        grad_ref_norm = _robust_normalize(grad_ref, p_low, p_high, vm)
        grad_ins_norm = _robust_normalize(grad_ins, p_low, p_high, vm)

        anomaly = np.abs(grad_ref_norm - grad_ins_norm).astype(np.float32, copy=False)

        strong_edge_fraction = 0.0
        strong_edge_pixel_count = 0
        anomaly_mean_before_edge_suppression = _mean_over_region(anomaly, vm)
        anomaly_mean_after_edge_suppression = anomaly_mean_before_edge_suppression
        anomaly_mean_delta_edge_suppression = 0.0

        if edge_suppression_enabled:
            if vm is not None and np.any(vm):
                edge_vals = grad_ins[vm]
            else:
                edge_vals = grad_ins.reshape(-1)
            if edge_vals.size > 0:
                edge_thr = float(np.percentile(edge_vals, edge_percentile))
                strong_edge_mask = grad_ins >= edge_thr
                strong_edge_pixel_count = int(np.count_nonzero(strong_edge_mask))
                strong_edge_fraction = float(np.mean(strong_edge_mask.astype(np.float32)))
                anomaly_mean_before_edge_suppression = _mean_over_region(anomaly, vm)
                anomaly = anomaly * np.where(strong_edge_mask, edge_weight_on_edges, 1.0).astype(np.float32)
                anomaly_mean_after_edge_suppression = _mean_over_region(anomaly, vm)
                anomaly_mean_delta_edge_suppression = float(
                    anomaly_mean_after_edge_suppression - anomaly_mean_before_edge_suppression
                )
        else:
            strong_edge_pixel_count = 0
            strong_edge_fraction = 0.0
            anomaly_mean_after_edge_suppression = anomaly_mean_before_edge_suppression
            anomaly_mean_delta_edge_suppression = 0.0

        if vm is not None:
            anomaly = anomaly.copy()
            anomaly[~vm] = 0.0

        metadata: Dict[str, Any] = {
            "method": self.name,
            "pre_blur_sigma": pre_blur_sigma,
            "post_blur_sigma": post_blur_sigma,
            "gradient_ksize": gradient_ksize,
            "norm_percentile_low": p_low,
            "norm_percentile_high": p_high,
            "used_valid_mask": vm is not None,
            "edge_suppression_enabled": edge_suppression_enabled,
            "edge_percentile": float(edge_percentile),
            "edge_weight_on_edges": float(edge_weight_on_edges),
            "strong_edge_fraction": float(strong_edge_fraction),
            "strong_edge_pixel_count": int(strong_edge_pixel_count),
            "anomaly_mean_before_edge_suppression": float(anomaly_mean_before_edge_suppression),
            "anomaly_mean_after_edge_suppression": float(anomaly_mean_after_edge_suppression),
            "anomaly_mean_delta_edge_suppression": float(anomaly_mean_delta_edge_suppression),
        }
        return anomaly, metadata

