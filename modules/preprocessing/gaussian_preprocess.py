from __future__ import annotations

import numpy as np

from modules.base import PreprocessorBase
from utils.image_ops import gaussian_blur, normalize_to_float32, to_grayscale


class GaussianPreprocessor(PreprocessorBase):
    def __init__(self) -> None:
        super().__init__(name="gaussian_preprocess")

    def run(self, reference_image, inspected_image, cfg):
        self.validate_config(cfg)
        self.validate_array(reference_image, "reference_image")
        self.validate_array(inspected_image, "inspected_image")
        self.validate_same_shape(
            reference_image, inspected_image, "reference_image", "inspected_image"
        )

        convert_to_grayscale = bool(self.get_param(cfg, "convert_to_grayscale", True))
        gaussian_sigma = float(self.get_param(cfg, "gaussian_sigma", 0.8))

        ref = normalize_to_float32(reference_image)
        ins = normalize_to_float32(inspected_image)

        if convert_to_grayscale:
            ref = to_grayscale(ref)
            ins = to_grayscale(ins)

        ref = gaussian_blur(ref, sigma=gaussian_sigma)
        ins = gaussian_blur(ins, sigma=gaussian_sigma)

        return ref.astype(np.float32), ins.astype(np.float32)

