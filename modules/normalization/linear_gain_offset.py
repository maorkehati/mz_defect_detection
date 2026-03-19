from __future__ import annotations

from typing import Tuple

import numpy as np

from modules.base import NormalizerBase


class LinearGainOffsetNormalizer(NormalizerBase):
    def __init__(self) -> None:
        super().__init__(name="linear_gain_offset")

    def run(self, reference_image, inspected_image, cfg):
        self.validate_config(cfg)
        self.validate_array(reference_image, "reference_image")
        self.validate_array(inspected_image, "inspected_image")
        self.validate_same_shape(
            reference_image, inspected_image, "reference_image", "inspected_image"
        )

        fit_gain = bool(self.get_param(cfg, "fit_gain", True))
        fit_offset = bool(self.get_param(cfg, "fit_offset", True))
        robust = bool(self.get_param(cfg, "robust", True))
        clip_output = bool(self.get_param(cfg, "clip_output", False))

        ref = reference_image.astype(np.float32, copy=False)
        ins = inspected_image.astype(np.float32, copy=False)

        gain, offset = self._estimate_gain_offset(
            reference=ref,
            inspected=ins,
            fit_gain=fit_gain,
            fit_offset=fit_offset,
            robust=robust,
        )

        ref_norm = gain * ref + offset
        ins_norm = ins

        if clip_output:
            lo = min(float(ins.min()), float(ref_norm.min()))
            hi = max(float(ins.max()), float(ref_norm.max()))
            ref_norm = np.clip(ref_norm, lo, hi)

        metadata = {"gain": float(gain), "offset": float(offset), "method": self.name}
        return ref_norm.astype(np.float32), ins_norm.astype(np.float32), metadata

    def _estimate_gain_offset(
        self,
        reference: np.ndarray,
        inspected: np.ndarray,
        fit_gain: bool,
        fit_offset: bool,
        robust: bool,
    ) -> Tuple[float, float]:
        x = reference.reshape(-1).astype(np.float64)
        y = inspected.reshape(-1).astype(np.float64)

        if robust:
            x_med = np.median(x)
            y_med = np.median(y)
            x = np.clip(x, np.percentile(x, 1), np.percentile(x, 99))
            y = np.clip(y, np.percentile(y, 1), np.percentile(y, 99))
        else:
            x_med = np.mean(x)
            y_med = np.mean(y)

        if fit_gain and fit_offset:
            A = np.stack([x, np.ones_like(x)], axis=1)
            sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            gain = float(sol[0])
            offset = float(sol[1])
            return gain, offset

        if fit_gain and not fit_offset:
            denom = float(np.dot(x, x))
            gain = 1.0 if denom < 1e-12 else float(np.dot(x, y) / denom)
            return gain, 0.0

        if (not fit_gain) and fit_offset:
            offset = float(y_med - x_med)
            return 1.0, offset

        return 1.0, 0.0

