from __future__ import annotations

import cv2
import numpy as np

from modules.base import ThresholdingBase


class MadThresholding(ThresholdingBase):
    def __init__(self) -> None:
        super().__init__(name="mad_threshold")

    def run(self, anomaly_map, cfg):
        self.validate_config(cfg)
        self.validate_array(anomaly_map, "anomaly_map")

        arr = np.asarray(anomaly_map, dtype=np.float32)
        k_mad = float(self.get_param(cfg, "k_mad", 4.0))
        min_threshold = float(self.get_param(cfg, "min_threshold", 0.0))
        use_valid_mask = bool(self.get_param(cfg, "use_valid_mask", True))
        use_core_mask = bool(self.get_param(cfg, "use_core_mask", True))
        core_erode_iterations = int(self.get_param(cfg, "core_erode_iterations", 1))
        valid_mask = self.get_param(cfg, "valid_mask", None)

        used_valid_mask = False
        used_core_mask = False
        if use_valid_mask and valid_mask is not None:
            vm = np.asarray(valid_mask).astype(bool)
            if vm.shape == arr.shape and np.any(vm):
                used_valid_mask = True
            else:
                vm = np.ones(arr.shape, dtype=bool)
        else:
            vm = np.ones(arr.shape, dtype=bool)

        core_mask = vm
        if use_core_mask and np.any(vm):
            core_u8 = cv2.erode(
                vm.astype(np.uint8),
                np.ones((3, 3), np.uint8),
                iterations=max(1, core_erode_iterations),
            )
            if np.any(core_u8):
                core_mask = core_u8 > 0
                used_core_mask = True

        values = arr[core_mask] if np.any(core_mask) else arr.reshape(-1)
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        thr = max(float(min_threshold), float(med + k_mad * mad))

        binary_mask_raw = arr >= thr
        if used_valid_mask:
            binary_mask_raw = np.logical_and(binary_mask_raw, vm)

        threshold_map = np.full_like(arr, fill_value=thr, dtype=np.float32)
        positive_count = int(np.count_nonzero(binary_mask_raw))
        total_count = int(binary_mask_raw.size)
        metadata = {
            "method": self.name,
            "threshold_value": float(thr),
            "median": float(med),
            "mad": float(mad),
            "k_mad": float(k_mad),
            "min_threshold": float(min_threshold),
            "valid_pixel_count": int(np.count_nonzero(core_mask)),
            "used_valid_mask": bool(used_valid_mask),
            "used_core_mask": bool(used_core_mask),
            "positive_pixel_count": positive_count,
            "positive_fraction": float((positive_count / total_count) if total_count else 0.0),
        }
        return binary_mask_raw.astype(bool), threshold_map, metadata

