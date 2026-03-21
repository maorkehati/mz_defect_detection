from __future__ import annotations

import numpy as np

from modules.base import ThresholdingBase


class FixedThresholding(ThresholdingBase):
    def __init__(self) -> None:
        super().__init__(name="fixed_threshold")

    def run(self, anomaly_map, cfg):
        self.validate_config(cfg)
        self.validate_array(anomaly_map, "anomaly_map")
        arr = np.asarray(anomaly_map, dtype=np.float32)

        thr = float(self.get_param(cfg, "threshold_value", 0.15))
        use_valid_mask = bool(self.get_param(cfg, "use_valid_mask", True))
        valid_mask = self.get_param(cfg, "valid_mask", None)

        binary = arr > thr
        used_valid_mask = False
        if use_valid_mask and valid_mask is not None:
            v = np.asarray(valid_mask).astype(bool)
            if v.shape == arr.shape:
                binary = np.logical_and(binary, v)
                used_valid_mask = True

        threshold_map = np.full_like(arr, fill_value=thr, dtype=np.float32)
        metadata = {
            "method": self.name,
            "threshold_value": float(thr),
            "used_valid_mask": bool(used_valid_mask),
        }
        return binary.astype(bool), threshold_map, metadata

