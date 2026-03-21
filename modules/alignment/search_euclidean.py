from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from modules.base import AlignerBase
from utils.image_ops import estimate_translation_phase_correlation


def _robust_normalize(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    p1, p99 = np.percentile(arr, [1, 99])
    if float(p99) <= float(p1):
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - float(p1)) / (float(p99) - float(p1)), 0.0, 1.0).astype(np.float32)


def _gradient_magnitude(img: np.ndarray) -> np.ndarray:
    x = cv2.GaussianBlur(np.asarray(img, dtype=np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.magnitude(gx, gy)
    return cv2.GaussianBlur(g, (0, 0), 1.0).astype(np.float32, copy=False)


def _hanning2d(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def _affine_about_center(shape: tuple[int, int], angle_deg: float, tx: float, ty: float) -> np.ndarray:
    h, w = shape
    center = (w / 2.0, h / 2.0)
    rot = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0).astype(np.float32)
    rot[:, 2] += [float(tx), float(ty)]
    return rot


def _warp_image(img: np.ndarray, M: np.ndarray, shape: tuple[int, int], interp: int) -> np.ndarray:
    h, w = shape
    return cv2.warpAffine(
        np.asarray(img, dtype=np.float32),
        np.asarray(M, dtype=np.float32),
        (int(w), int(h)),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32, copy=False)


def _estimate_shift_with_masks(
    inspected_feat: np.ndarray,
    rotated_ref_feat: np.ndarray,
    rotated_ref_mask: np.ndarray,
    upsample_factor: int,
) -> tuple[float, float]:
    try:
        from skimage.registration import phase_cross_correlation

        shift_rc, _, _ = phase_cross_correlation(
            reference_image=np.asarray(inspected_feat, dtype=np.float32),
            moving_image=np.asarray(rotated_ref_feat, dtype=np.float32),
            reference_mask=np.ones_like(inspected_feat, dtype=bool),
            moving_mask=np.asarray(rotated_ref_mask, dtype=np.float32) > 0.5,
            overlap_ratio=0.3,
            upsample_factor=int(upsample_factor),
            normalization=None,
        )
        dy, dx = float(shift_rc[0]), float(shift_rc[1])
        return float(dx), float(dy)
    except Exception:
        # Environment fallback (keeps aligner runnable if skimage binary is unavailable).
        dy, dx, _ = estimate_translation_phase_correlation(
            reference_image=np.asarray(rotated_ref_feat, dtype=np.float32),
            inspected_image=np.asarray(inspected_feat, dtype=np.float32),
            subpixel_refinement=True,
        )
        return float(dx), float(dy)


def _score_candidate(
    ref_img: np.ndarray,
    ins_img: np.ndarray,
    ref_feat: np.ndarray,
    ins_feat: np.ndarray,
    angle_deg: float,
    overlap_threshold: float,
    upsample_factor: int,
) -> dict[str, Any]:
    shape = ref_img.shape
    M_rot = _affine_about_center(shape, angle_deg, 0.0, 0.0)
    rotated_ref_feat = _warp_image(ref_feat, M_rot, shape, cv2.INTER_LINEAR)
    rotated_ref_mask = _warp_image(np.ones(shape, np.float32), M_rot, shape, cv2.INTER_NEAREST)

    tx, ty = _estimate_shift_with_masks(
        inspected_feat=ins_feat,
        rotated_ref_feat=rotated_ref_feat,
        rotated_ref_mask=rotated_ref_mask,
        upsample_factor=upsample_factor,
    )

    M = _affine_about_center(shape, angle_deg, tx, ty)
    aligned_ref_feat = _warp_image(ref_feat, M, shape, cv2.INTER_LINEAR)
    aligned_ref_img = _warp_image(ref_img, M, shape, cv2.INTER_LINEAR)
    valid_mask = _warp_image(np.ones(shape, np.float32), M, shape, cv2.INTER_NEAREST) > 0.5

    overlap = float(valid_mask.mean())
    if overlap < float(overlap_threshold):
        score = float("inf")
        pixels_scored = 0
    else:
        diff = np.abs(aligned_ref_feat - ins_feat)
        core = cv2.erode(valid_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
        if np.any(core):
            score = float(np.median(diff[core]))
            pixels_scored = int(np.count_nonzero(core))
        else:
            score = float("inf")
            pixels_scored = 0

    return {
        "angle_deg": float(angle_deg),
        "theta_deg": float(angle_deg),
        "tx": float(tx),
        "ty": float(ty),
        "score": float(score),
        "overlap": float(overlap),
        "overlap_fraction": float(overlap),
        "valid": bool(np.isfinite(score)),
        "pixels_scored": int(pixels_scored),
        "matrix": np.asarray(M, dtype=np.float32),
        "aligned_ref_img": aligned_ref_img,
        "valid_mask": np.asarray(valid_mask, dtype=bool),
    }


class SearchEuclideanAligner(AlignerBase):
    def __init__(self) -> None:
        super().__init__(name="search_euclidean")

    def run(self, reference_image, inspected_image, cfg):
        self.validate_config(cfg)
        self.validate_array(reference_image, "reference_image")
        self.validate_array(inspected_image, "inspected_image")
        self.validate_same_shape(reference_image, inspected_image, "reference_image", "inspected_image")

        ref_in = np.asarray(reference_image, dtype=np.float32)
        ins_in = np.asarray(inspected_image, dtype=np.float32)
        if ref_in.ndim != 2 or ins_in.ndim != 2:
            raise ValueError("search_euclidean requires single-channel 2D images.")

        coarse_angle_min = float(self.get_param(cfg, "coarse_angle_min", -4.0))
        coarse_angle_max = float(self.get_param(cfg, "coarse_angle_max", 4.0))
        coarse_steps = int(self.get_param(cfg, "coarse_steps", 17))
        refine_half_width = float(self.get_param(cfg, "refine_half_width", 0.75))
        refine_steps = int(self.get_param(cfg, "refine_steps", 15))
        overlap_threshold = float(self.get_param(cfg, "overlap_threshold", 0.92))
        upsample_factor = int(self.get_param(cfg, "upsample_factor", 20))

        # Alignment-only internal normalization + feature construction.
        ref_n = _robust_normalize(ref_in)
        ins_n = _robust_normalize(ins_in)
        window = _hanning2d(ref_n.shape)
        ref_feat = _gradient_magnitude(ref_n) * window
        ins_feat = _gradient_magnitude(ins_n) * window

        coarse_angles = np.linspace(coarse_angle_min, coarse_angle_max, coarse_steps, dtype=np.float32)
        coarse_scores = [
            _score_candidate(ref_n, ins_n, ref_feat, ins_feat, float(a), overlap_threshold, upsample_factor)
            for a in coarse_angles
        ]
        coarse_best = min(coarse_scores, key=lambda d: float(d.get("score", float("inf"))))

        refined_angles = np.linspace(
            float(coarse_best["angle_deg"]) - refine_half_width,
            float(coarse_best["angle_deg"]) + refine_half_width,
            refine_steps,
            dtype=np.float32,
        )
        refined_scores = [
            _score_candidate(ref_n, ins_n, ref_feat, ins_feat, float(a), overlap_threshold, upsample_factor)
            for a in refined_angles
        ]
        best = min(refined_scores, key=lambda d: float(d.get("score", float("inf"))))

        best_matrix = np.asarray(best["matrix"], dtype=np.float32)
        # Apply final transform to the original inputs (pipeline-facing behavior).
        aligned_ref = _warp_image(ref_in, best_matrix, ref_in.shape, cv2.INTER_LINEAR)
        valid_mask = _warp_image(np.ones(ref_in.shape, np.float32), best_matrix, ref_in.shape, cv2.INTER_NEAREST) > 0.5
        valid_fraction = float(np.mean(valid_mask.astype(np.float32)))

        coarse_records: List[Dict[str, Any]] = []
        for d in coarse_scores:
            rec = {
                "iteration": 0,
                "theta_deg": float(d["theta_deg"]),
                "tx": float(d["tx"]),
                "ty": float(d["ty"]),
                "score": float(d["score"]),
                "overlap_fraction": float(d["overlap_fraction"]),
                "valid": bool(d["valid"]),
                "pixels_scored": int(d.get("pixels_scored", 0)),
            }
            coarse_records.append(rec)

        refined_records: List[Dict[str, Any]] = []
        for d in refined_scores:
            rec = {
                "iteration": 1,
                "theta_deg": float(d["theta_deg"]),
                "tx": float(d["tx"]),
                "ty": float(d["ty"]),
                "score": float(d["score"]),
                "overlap_fraction": float(d["overlap_fraction"]),
                "valid": bool(d["valid"]),
                "pixels_scored": int(d.get("pixels_scored", 0)),
            }
            refined_records.append(rec)

        metadata: Dict[str, Any] = {
            "method": self.name,
            "score_mode": "median_abs_grad_diff_core",
            "scoring_method": "median_abs_grad_diff_core",
            "coarse_angle_min": float(coarse_angle_min),
            "coarse_angle_max": float(coarse_angle_max),
            "coarse_steps": int(coarse_steps),
            "refine_half_width": float(refine_half_width),
            "refine_steps": int(refine_steps),
            "overlap_threshold": float(overlap_threshold),
            "upsample_factor": int(upsample_factor),
            "best_theta_deg": float(best["theta_deg"]),
            "best_tx": float(best["tx"]),
            "best_ty": float(best["ty"]),
            "final_theta_deg": float(best["theta_deg"]),
            "final_tx": float(best["tx"]),
            "final_ty": float(best["ty"]),
            "best_score": float(best["score"]),
            "final_score": float(best["score"]),
            "best_stage": "refined",
            "overlap_fraction": float(best["overlap_fraction"]),
            "valid_pixel_fraction": float(valid_fraction),
            "warp_matrix": np.asarray(best_matrix, dtype=np.float32).tolist(),
            "valid_mask": np.asarray(valid_mask, dtype=bool),
            "coarse_scores": coarse_scores,
            "refined_scores": refined_scores,
            "coarse_candidate_records": coarse_records,
            "refined_candidate_records": refined_records,
            "candidate_records": coarse_records + refined_records,
            "coarse_candidates_total": int(len(coarse_records)),
            "coarse_candidates_valid": int(sum(1 for x in coarse_records if bool(x.get("valid", False)))),
            "refined_candidates_total": int(len(refined_records)),
            "refined_candidates_valid": int(sum(1 for x in refined_records if bool(x.get("valid", False)))),
            "iterations_used": 2,
            "converged": True,
            "iteration_summaries": [
                {
                    "iteration": 0,
                    "theta_search_min": float(coarse_angle_min),
                    "theta_search_max": float(coarse_angle_max),
                    "chosen_theta_deg": float(coarse_best["theta_deg"]),
                    "chosen_tx": float(coarse_best["tx"]),
                    "chosen_ty": float(coarse_best["ty"]),
                    "best_score": float(coarse_best["score"]),
                    "overlap_fraction": float(coarse_best["overlap_fraction"]),
                    "valid": bool(coarse_best["valid"]),
                },
                {
                    "iteration": 1,
                    "theta_search_min": float(refined_angles.min()),
                    "theta_search_max": float(refined_angles.max()),
                    "chosen_theta_deg": float(best["theta_deg"]),
                    "chosen_tx": float(best["tx"]),
                    "chosen_ty": float(best["ty"]),
                    "best_score": float(best["score"]),
                    "overlap_fraction": float(best["overlap_fraction"]),
                    "valid": bool(best["valid"]),
                },
            ],
        }
        return aligned_ref, ins_in.astype(np.float32, copy=False), metadata

