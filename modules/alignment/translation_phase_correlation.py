from __future__ import annotations

from typing import Any, Dict

import numpy as np

from modules.base import AlignerBase
from utils.image_ops import apply_shift, estimate_translation_phase_correlation


class TranslationPhaseCorrelationAligner(AlignerBase):
    def __init__(self) -> None:
        super().__init__(name="translation_phase_correlation")

    def run(self, reference_image, inspected_image, cfg):
        self.validate_config(cfg)
        self.validate_array(reference_image, "reference_image")
        self.validate_array(inspected_image, "inspected_image")
        self.validate_same_shape(
            reference_image, inspected_image, "reference_image", "inspected_image"
        )

        subpixel_refinement = bool(
            self.get_param(cfg, "subpixel_refinement", True)
        )
        max_shift = self.get_param(cfg, "max_shift", None)
        interpolation_order = int(self.get_param(cfg, "interpolation_order", 1))

        shift_y, shift_x, peak_response = estimate_translation_phase_correlation(
            reference_image=reference_image,
            inspected_image=inspected_image,
            subpixel_refinement=subpixel_refinement,
        )

        if max_shift is not None:
            max_shift_f = float(max_shift)
            shift_y = float(np.clip(shift_y, -max_shift_f, max_shift_f))
            shift_x = float(np.clip(shift_x, -max_shift_f, max_shift_f))

        aligned_ref = apply_shift(
            reference_image,
            shift_y=shift_y,
            shift_x=shift_x,
            order=interpolation_order,
        ).astype(np.float32, copy=False)

        ins_out = np.asarray(inspected_image, dtype=np.float32)

        metadata: Dict[str, Any] = {
            "shift_y": float(shift_y),
            "shift_x": float(shift_x),
            "peak_response": float(peak_response),
            "method": self.name,
        }

        return aligned_ref, ins_out, metadata

