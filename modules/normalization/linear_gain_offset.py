from __future__ import annotations

from typing import Tuple

import cv2
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
        valid_mask = self.get_param(cfg, "valid_mask", None)
        min_fit_pixels = int(self.get_param(cfg, "min_fit_pixels", 1000))

        ref = reference_image.astype(np.float32, copy=False)
        ins = inspected_image.astype(np.float32, copy=False)

        gain, offset, fit_metadata = self._estimate_gain_offset(
            reference=ref,
            inspected=ins,
            fit_gain=fit_gain,
            fit_offset=fit_offset,
            robust=robust,
            valid_mask=valid_mask,
            min_fit_pixels=min_fit_pixels,
        )

        ref_norm = gain * ref + offset
        ins_norm = ins

        if clip_output:
            lo = min(float(ins.min()), float(ref_norm.min()))
            hi = max(float(ins.max()), float(ref_norm.max()))
            ref_norm = np.clip(ref_norm, lo, hi)

        metadata = {"gain": float(gain), "offset": float(offset), "method": self.name, **fit_metadata}
        return ref_norm.astype(np.float32), ins_norm.astype(np.float32), metadata

    def _estimate_gain_offset(
        self,
        reference: np.ndarray,
        inspected: np.ndarray,
        fit_gain: bool,
        fit_offset: bool,
        robust: bool,
        valid_mask,
        min_fit_pixels: int,
    ) -> Tuple[float, float, dict]:
        vm = None
        if valid_mask is not None:
            cand = np.asarray(valid_mask).astype(bool)
            if cand.shape == reference.shape:
                vm = cand
        if vm is None:
            vm = np.ones(reference.shape, dtype=bool)

        # Core overlap for robust fit: erode valid region once with 3x3 kernel.
        core = cv2.erode(vm.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
        overlap_fraction = float(np.mean(vm.astype(np.float32)))
        core_fraction = float(np.mean(core.astype(np.float32)))
        n_used = int(np.count_nonzero(core))

        fallback_reason = None
        if n_used < int(min_fit_pixels):
            fallback_reason = f"too_few_pixels:{n_used}<{int(min_fit_pixels)}"
            return 1.0, 0.0, {
                "fit_pixels_used": n_used,
                "overlap_fraction": overlap_fraction,
                "core_overlap_fraction": core_fraction,
                "fallback_used": True,
                "fallback_reason": fallback_reason,
            }

        x = reference[core].reshape(-1).astype(np.float64)
        y = inspected[core].reshape(-1).astype(np.float64)

        if robust:
            x_med = np.median(x)
            y_med = np.median(y)
            x = np.clip(x, np.percentile(x, 1), np.percentile(x, 99))
            y = np.clip(y, np.percentile(y, 1), np.percentile(y, 99))
        else:
            x_med = np.mean(x)
            y_med = np.mean(y)

        try:
            if fit_gain and fit_offset:
                A = np.stack([x, np.ones_like(x)], axis=1)
                sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                gain = float(sol[0])
                offset = float(sol[1])
            elif fit_gain and not fit_offset:
                denom = float(np.dot(x, x))
                gain = 1.0 if denom < 1e-12 else float(np.dot(x, y) / denom)
                offset = 0.0
            elif (not fit_gain) and fit_offset:
                gain = 1.0
                offset = float(y_med - x_med)
            else:
                gain = 1.0
                offset = 0.0
        except Exception as exc:
            fallback_reason = f"lstsq_error:{exc}"
            return 1.0, 0.0, {
                "fit_pixels_used": n_used,
                "overlap_fraction": overlap_fraction,
                "core_overlap_fraction": core_fraction,
                "fallback_used": True,
                "fallback_reason": fallback_reason,
            }

        if (not np.isfinite(gain)) or (not np.isfinite(offset)):
            fallback_reason = "non_finite_solution"
            return 1.0, 0.0, {
                "fit_pixels_used": n_used,
                "overlap_fraction": overlap_fraction,
                "core_overlap_fraction": core_fraction,
                "fallback_used": True,
                "fallback_reason": fallback_reason,
            }

        return gain, offset, {
            "fit_pixels_used": n_used,
            "overlap_fraction": overlap_fraction,
            "core_overlap_fraction": core_fraction,
            "fallback_used": False,
            "fallback_reason": None,
        }

