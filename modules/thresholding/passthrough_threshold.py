"""Pass binary mask from comparator (noise-calibrated path) without MAD/Otsu."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from modules.base import ThresholdingBase


class PassthroughThresholding(ThresholdingBase):
    def __init__(self) -> None:
        super().__init__(name="passthrough_threshold")

    def run(self, anomaly_map, cfg) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        self.validate_config(cfg)
        self.validate_array(anomaly_map, "anomaly_map")
        mask = self.get_param(cfg, "passthrough_binary_mask", None)
        if mask is None:
            raise ValueError(
                "passthrough_threshold requires runtime param passthrough_binary_mask "
                "(set by pipeline from noise_calibrated_residual comparator)."
            )
        arr = np.asarray(anomaly_map, dtype=np.float32)
        m = np.asarray(mask).astype(bool)
        if m.shape != arr.shape:
            raise ValueError(f"passthrough_binary_mask shape {m.shape} != anomaly_map {arr.shape}")
        meta: Dict[str, Any] = {
            "method": self.name,
            "threshold_value": float("nan"),
            "passthrough": True,
            "positive_pixel_count": int(np.count_nonzero(m)),
            "positive_fraction": float(np.mean(m.astype(np.float32))),
        }
        return m, np.full_like(arr, np.nan, dtype=np.float32), meta
