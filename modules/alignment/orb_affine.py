from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from modules.base import AlignerBase


class OrbAffineAligner(AlignerBase):
    def __init__(self) -> None:
        super().__init__(name="orb_affine")

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
            raise ValueError("orb_affine requires single-channel 2D images.")

        nfeatures = int(self.get_param(cfg, "nfeatures", 2000))
        top_matches = int(self.get_param(cfg, "top_matches", 300))
        min_matches = int(self.get_param(cfg, "min_matches_for_estimation", 12))
        ransac_thr = float(self.get_param(cfg, "ransac_reproj_threshold", 3.0))
        allow_fallback = bool(self.get_param(cfg, "allow_fallback_to_identity", True))

        ref_u8 = self._to_u8(ref)
        ins_u8 = self._to_u8(ins)
        orb = cv2.ORB_create(nfeatures=nfeatures)
        kp_ref, des_ref = orb.detectAndCompute(ref_u8, None)
        kp_ins, des_ins = orb.detectAndCompute(ins_u8, None)

        num_keypoints_ref = 0 if kp_ref is None else len(kp_ref)
        num_keypoints_ins = 0 if kp_ins is None else len(kp_ins)

        metadata: Dict[str, Any] = {
            "method": self.name,
            "num_keypoints_ref": int(num_keypoints_ref),
            "num_keypoints_inspected": int(num_keypoints_ins),
            "num_matches_total": 0,
            "num_matches_used": 0,
            "affine_matrix": None,
            "inlier_count": 0,
            "fallback_used": False,
        }

        if des_ref is None or des_ins is None:
            return self._fallback_or_raise(ref, ins, allow_fallback, metadata, "ORB descriptors not found.")

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des_ref, des_ins)
        matches = sorted(matches, key=lambda m: m.distance)
        metadata["num_matches_total"] = int(len(matches))

        if len(matches) < min_matches:
            return self._fallback_or_raise(
                ref, ins, allow_fallback, metadata, f"Insufficient matches: {len(matches)} < {min_matches}."
            )

        used = matches[: min(len(matches), top_matches)]
        metadata["num_matches_used"] = int(len(used))

        src = np.float32([kp_ref[m.queryIdx].pt for m in used]).reshape(-1, 1, 2)
        dst = np.float32([kp_ins[m.trainIdx].pt for m in used]).reshape(-1, 1, 2)

        M, inlier_mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thr,
        )
        if M is None:
            return self._fallback_or_raise(ref, ins, allow_fallback, metadata, "Affine estimation failed.")

        h, w = ins.shape
        aligned_ref = cv2.warpAffine(
            ref,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.float32, copy=False)

        ones_mask = np.ones_like(ref, dtype=np.uint8) * 255
        warped_valid = cv2.warpAffine(
            ones_mask,
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_mask = warped_valid > 0

        metadata["affine_matrix"] = M.tolist()
        metadata["inlier_count"] = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
        metadata["valid_mask"] = valid_mask.astype(bool)

        return aligned_ref, ins.astype(np.float32, copy=False), metadata

    def _fallback_or_raise(self, ref, ins, allow_fallback, metadata, reason):
        if not allow_fallback:
            raise RuntimeError(f"orb_affine failed: {reason}")
        metadata["fallback_used"] = True
        metadata["reason"] = reason
        metadata["valid_mask"] = np.ones_like(ins, dtype=bool)
        return ref.astype(np.float32, copy=False), ins.astype(np.float32, copy=False), metadata

    def _to_u8(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi <= lo:
            return np.zeros_like(x, dtype=np.uint8)
        x01 = (x - lo) / (hi - lo)
        return np.clip(x01 * 255.0, 0.0, 255.0).astype(np.uint8)

