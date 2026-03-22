"""
Diagnostics for the exact numeric array passed to MAD / fixed thresholding (not display-normalized).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def summarize_threshold_input(
    arr: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    *,
    threshold_value: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Per-case statistics on the real threshold input (finite values, optional valid_mask intersection).
    """
    a = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(a)
    vm = (
        np.ones(a.shape, dtype=bool)
        if valid_mask is None
        else (np.asarray(valid_mask, dtype=bool) & finite)
    )
    if vm.shape != a.shape:
        vm = finite
    vals = a[vm]
    n_vm = int(np.count_nonzero(vm))
    n_fin = int(np.count_nonzero(finite))
    out: Dict[str, Any] = {
        "shape": list(a.shape),
        "finite_pixel_count": n_fin,
        "valid_masked_pixel_count": n_vm,
        "strictly_positive_count": int(np.count_nonzero(vals > 0)) if vals.size else 0,
        "nan_count": int(np.count_nonzero(~finite)),
        "inf_count": int(np.count_nonzero(np.isinf(a))),
    }
    if vals.size == 0:
        out.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "mad": None,
                "p90": None,
                "p95": None,
                "p99": None,
            }
        )
    else:
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        out.update(
            {
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "mean": float(np.mean(vals)),
                "median": med,
                "mad": mad,
                "p90": float(np.percentile(vals, 90.0)),
                "p95": float(np.percentile(vals, 95.0)),
                "p99": float(np.percentile(vals, 99.0)),
            }
        )
    if threshold_value is not None and np.isfinite(threshold_value):
        thr = float(threshold_value)
        t = thr
        m = vm if np.any(vm) else finite
        am = a[m]
        out["threshold_value_used"] = t
        out["count_gt_0"] = int(np.count_nonzero(am > 0))
        out["count_ge_thr"] = int(np.count_nonzero(am >= t))
        out["count_gt_thr"] = int(np.count_nonzero(am > t))
        out["count_eq_thr"] = int(np.count_nonzero(am == t))
    return out


def explain_threshold_collapse(
    *,
    median: float,
    mad: float,
    k_mad: float,
    min_threshold: float,
    thr_before_cap: float,
    thr_after_cap: Optional[float],
    max_val: float,
    min_val: float,
) -> str:
    """Human-readable note when threshold hits 0 or 1 for [0,1] maps."""
    parts = []
    if max_val <= 1.000001 and min_val >= -1e-6:
        if thr_after_cap is not None and thr_after_cap >= 0.999 and mad < 1e-12:
            parts.append(
                "Anomaly appears in [0,1] with very small MAD on core; effective threshold may cap at 1.0 — "
                "mask can be empty if no pixels reach exactly 1.0 (floating point)."
            )
        if abs(thr_after_cap or 0.0) < 1e-12 and abs(median) < 1e-12 and mad < 1e-12:
            parts.append(
                "Median and MAD are ~0 on core (degenerate or all-zero core stats); threshold stays at min_threshold."
            )
    return " ".join(parts) if parts else ""
