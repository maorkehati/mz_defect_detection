from __future__ import annotations

import numpy as np

from modules.base import ThresholdingBase
from utils.stats import robust_threshold_value


class MadThresholding(ThresholdingBase):
    def __init__(self) -> None:
        super().__init__(name="mad_threshold")

    def run(self, anomaly_map, cfg):
        self.validate_config(cfg)
        self.validate_array(anomaly_map, "anomaly_map")

        k_mad = float(self.get_param(cfg, "k_mad", 4.0))
        min_threshold = float(self.get_param(cfg, "min_threshold", 0.0))

        thr = robust_threshold_value(
            anomaly_map=anomaly_map,
            k_mad=k_mad,
            min_threshold=min_threshold,
        )

        binary_mask_raw = anomaly_map > thr
        threshold_map = np.full_like(
            anomaly_map, fill_value=thr, dtype=np.float32
        )

        return binary_mask_raw.astype(bool), threshold_map

