from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu

from modules.base import ThresholdingBase


class OtsuThresholding(ThresholdingBase):
    def __init__(self) -> None:
        super().__init__(name="otsu_threshold")

    def run(self, anomaly_map, cfg):
        self.validate_config(cfg)
        self.validate_array(anomaly_map, "anomaly_map")
        arr = np.asarray(anomaly_map, dtype=np.float32)

        use_valid_mask = bool(self.get_param(cfg, "use_valid_mask", True))
        valid_mask = self.get_param(cfg, "valid_mask", None)
        used_valid_mask = False

        values = arr.reshape(-1)
        if use_valid_mask and valid_mask is not None:
            v = np.asarray(valid_mask).astype(bool)
            if v.shape == arr.shape and np.any(v):
                values = arr[v]
                used_valid_mask = True

        if values.size == 0:
            thr = float(arr.min())
        elif float(np.max(values)) == float(np.min(values)):
            thr = float(values.flat[0])
        else:
            thr = float(threshold_otsu(values))

        binary = arr > thr
        if used_valid_mask:
            binary = np.logical_and(binary, np.asarray(valid_mask).astype(bool))

        threshold_map = np.full_like(arr, fill_value=thr, dtype=np.float32)
        metadata = {
            "method": self.name,
            "threshold_value": float(thr),
            "used_valid_mask": bool(used_valid_mask),
        }
        return binary.astype(bool), threshold_map, metadata

