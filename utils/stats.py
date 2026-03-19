from __future__ import annotations

import numpy as np


def mad(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float32)
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)) + eps)


def robust_threshold_value(
    anomaly_map: np.ndarray,
    k_mad: float,
    min_threshold: float = 0.0,
) -> float:
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    med = float(np.median(anomaly_map))
    m = mad(anomaly_map)
    # 1.4826 converts MAD to sigma-like scale for Gaussian noise
    sigma = 1.4826 * m
    thr = med + k_mad * sigma
    return float(max(thr, min_threshold))