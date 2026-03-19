from __future__ import annotations

import numpy as np
from scipy import ndimage

from modules.base import ComparatorBase


class AbsoluteDifferenceComparator(ComparatorBase):
    def __init__(self) -> None:
        super().__init__(name="absolute_difference")

    def run(self, reference_image, inspected_image, cfg):
        self.validate_config(cfg)
        self.validate_array(reference_image, "reference_image")
        self.validate_array(inspected_image, "inspected_image")
        self.validate_same_shape(
            reference_image, inspected_image, "reference_image", "inspected_image"
        )

        gradient_weight = float(self.get_param(cfg, "gradient_weight", 0.0))
        coarse_sigma = float(self.get_param(cfg, "coarse_sigma", 0.0))
        coarse_weight = float(self.get_param(cfg, "coarse_weight", 0.0))

        ref = reference_image.astype(np.float32, copy=False)
        ins = inspected_image.astype(np.float32, copy=False)

        anomaly = np.abs(ins - ref)

        if gradient_weight > 0:
            ref_gx = ndimage.sobel(ref, axis=1)
            ref_gy = ndimage.sobel(ref, axis=0)
            ins_gx = ndimage.sobel(ins, axis=1)
            ins_gy = ndimage.sobel(ins, axis=0)

            grad_ref = np.hypot(ref_gx, ref_gy)
            grad_ins = np.hypot(ins_gx, ins_gy)
            grad_diff = np.abs(grad_ins - grad_ref)

            anomaly = anomaly + gradient_weight * grad_diff

        if coarse_weight > 0 and coarse_sigma > 0:
            ref_coarse = ndimage.gaussian_filter(ref, sigma=coarse_sigma)
            ins_coarse = ndimage.gaussian_filter(ins, sigma=coarse_sigma)
            coarse_diff = np.abs(ins_coarse - ref_coarse)
            anomaly = anomaly + coarse_weight * coarse_diff

        return anomaly.astype(np.float32)

