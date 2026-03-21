from __future__ import annotations

from typing import Any, Dict

import numpy as np
from skimage.metrics import structural_similarity

from modules.base import ComparatorBase


class SsimComparator(ComparatorBase):
    def __init__(self) -> None:
        super().__init__(name="ssim_comparator")

    def run(self, reference_image, inspected_image, cfg):
        self.validate_config(cfg)
        self.validate_array(reference_image, "reference_image")
        self.validate_array(inspected_image, "inspected_image")
        self.validate_same_shape(
            reference_image, inspected_image, "reference_image", "inspected_image"
        )

        ref = np.asarray(reference_image, dtype=np.float32)
        ins = np.asarray(inspected_image, dtype=np.float32)
        if ref.ndim != 2 or ins.ndim != 2:
            raise ValueError("ssim_comparator requires single-channel 2D images.")

        use_valid_mask = bool(self.get_param(cfg, "use_valid_mask", True))
        valid_mask = self.get_param(cfg, "valid_mask", None)
        win_size = int(self.get_param(cfg, "win_size", 7))
        win_size = self._resolve_win_size(win_size, min(ref.shape))

        data_min = min(float(ref.min()), float(ins.min()))
        data_max = max(float(ref.max()), float(ins.max()))
        data_range = data_max - data_min
        if data_range <= 0:
            data_range = 1.0

        global_ssim, ssim_map = structural_similarity(
            ref,
            ins,
            data_range=data_range,
            win_size=win_size,
            full=True,
        )
        anomaly = (1.0 - ssim_map).astype(np.float32, copy=False)

        used_valid_mask = False
        if use_valid_mask and valid_mask is not None:
            v = np.asarray(valid_mask).astype(bool)
            if v.shape == anomaly.shape:
                anomaly = anomaly.copy()
                anomaly[~v] = 0.0
                used_valid_mask = True

        metadata: Dict[str, Any] = {
            "method": self.name,
            "global_ssim_score": float(global_ssim),
            "used_valid_mask": used_valid_mask,
            "data_range": float(data_range),
            "ssim_map": ssim_map.astype(np.float32, copy=False),
        }
        return anomaly, metadata

    def _resolve_win_size(self, requested: int, min_dim: int) -> int:
        if min_dim < 3:
            raise ValueError(f"Image too small for SSIM: min_dim={min_dim}.")
        win = max(3, requested)
        if win % 2 == 0:
            win += 1
        if win > min_dim:
            win = min_dim if min_dim % 2 == 1 else (min_dim - 1)
        if win < 3:
            raise ValueError(f"Cannot choose valid SSIM window for min_dim={min_dim}.")
        return int(win)

