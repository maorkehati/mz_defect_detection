from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from modules.base import PostprocessorBase

# Rejection reason labels (used in metadata and diagnostics).
REJECT_KEYS = (
    "min_area",
    "max_area",
    "aspect_ratio",
    "fill_ratio",
    "border_touch",
    "degenerate_bbox",
    "sign_consistency",
)


def _odd_kernel_size(k: int) -> int:
    """OpenCV morphological kernels are typically odd-sized; ensure at least 1."""
    kk = max(1, int(k))
    if kk % 2 == 0:
        kk += 1
    return kk


def apply_pre_contour_morphology(
    mask_u8: np.ndarray,
    *,
    morph_open_kernel: int,
    morph_open_iterations: int,
    morph_close_kernel: int,
    morph_close_iterations: int,
) -> np.ndarray:
    """
    Optional opening (denoise) then closing (bridge) on the binary mask before findContours.
    Skips a step if kernel size < 1 or iterations < 1.
    """
    work = np.asarray(mask_u8, dtype=np.uint8, order="C")
    if work.size == 0:
        return work

    oi = max(0, int(morph_open_iterations))
    ok = max(0, int(morph_open_kernel))
    if oi > 0 and ok >= 1:
        ks = _odd_kernel_size(ok)
        k_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        for _ in range(oi):
            work = cv2.morphologyEx(work, cv2.MORPH_OPEN, k_el)

    ci = max(0, int(morph_close_iterations))
    ck = max(0, int(morph_close_kernel))
    if ci > 0 and ck >= 1:
        ks = _odd_kernel_size(ck)
        k_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        for _ in range(ci):
            work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, k_el)

    return work


def _p95_anomaly_inside_contour(
    vals: np.ndarray,
    mean_anomaly: float,
    max_anomaly: float,
) -> float:
    """
    95th percentile of anomaly values inside the contour mask.
    Safe for very small regions: 0 pixels -> mean; 1 pixel -> that value;
    non-finite percentile -> fall back to max then mean.
    """
    v = np.asarray(vals, dtype=np.float64).ravel()
    n = int(v.size)
    if n <= 0:
        return float(mean_anomaly)
    if n == 1:
        return float(v.flat[0])
    try:
        p = float(np.percentile(v, 95.0))
    except Exception:
        p = float(max_anomaly)
    if not np.isfinite(p):
        p = float(max_anomaly) if np.isfinite(max_anomaly) else float(mean_anomaly)
    return p


def _compute_sign_consistency(
    signed_residual: np.ndarray,
    inside: np.ndarray,
) -> Tuple[float, float, float, float, float, str]:
    """
    Compute signed-residual statistics inside the contour.
    Returns (positive_sum, negative_sum, positive_mean, negative_mean, sign_consistency, dominant_sign).
    sign_consistency = max(pos_sum, neg_sum) / (pos_sum + neg_sum + eps); 1.0 = perfectly one-sign.
    dominant_sign: "positive" if pos_sum >= neg_sum else "negative".
    """
    rvals = np.asarray(signed_residual[inside], dtype=np.float64).ravel()
    if rvals.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 1.0, "neutral"
    pos = np.maximum(rvals, 0.0)
    neg = np.maximum(-rvals, 0.0)
    pos_sum = float(np.sum(pos))
    neg_sum = float(np.sum(neg))
    pos_mean = float(np.mean(pos)) if pos.size > 0 else 0.0
    neg_mean = float(np.mean(neg)) if neg.size > 0 else 0.0
    eps = 1e-12
    total = pos_sum + neg_sum + eps
    sign_consistency = float(max(pos_sum, neg_sum) / total)
    dominant_sign = "positive" if pos_sum >= neg_sum else "negative"
    return pos_sum, neg_sum, pos_mean, neg_mean, sign_consistency, dominant_sign


def _ring_mean_and_p95(
    mask_inside_u8: np.ndarray,
    anom: np.ndarray,
    ring_radius_px: int,
    valid_mask: Optional[np.ndarray],
) -> Tuple[float, float]:
    """
    Dilate filled contour mask by ring_radius_px, subtract interior -> background ring.
    Intersect with valid_mask if provided. Returns (ring_mean, ring_p95); (nan, nan) if ring empty.
    """
    mask_inside_u8 = np.asarray(mask_inside_u8, dtype=np.uint8)
    h, w = int(mask_inside_u8.shape[0]), int(mask_inside_u8.shape[1])
    rr = int(max(1, ring_radius_px))
    ks = 2 * rr + 1
    k_el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    dilated = cv2.dilate(mask_inside_u8, k_el, iterations=1)
    inside = mask_inside_u8 > 0
    ring = (dilated > 0) & ~inside
    if valid_mask is not None:
        vm = np.asarray(valid_mask)
        if vm.shape == (h, w):
            ring = ring & vm.astype(bool)
    if not np.any(ring):
        return float("nan"), float("nan")
    rvals = np.asarray(anom[ring], dtype=np.float64).ravel()
    ring_mean = float(np.mean(rvals))
    if rvals.size <= 1:
        ring_p95 = ring_mean
    else:
        ring_p95 = float(np.percentile(rvals, 95.0))
    if not math.isfinite(ring_mean):
        ring_mean = float("nan")
    if not math.isfinite(ring_p95):
        ring_p95 = float("nan")
    return ring_mean, ring_p95


def compute_ranking_score(
    mode: str,
    area: float,
    mean_anomaly: float,
    max_anomaly: float,
    p95_anomaly: Optional[float] = None,
    *,
    ring_mean_anomaly: Optional[float] = None,
    sign_consistency: Optional[float] = None,
) -> float:
    """
    Dominance score for ordering contours after geometric filtering.

    Modes:
    - integrated_anomaly: area * mean (legacy; favors large regions); p95 / ring unused
    - intensity_size_balanced: p95 * sqrt(area); ring unused
    - intensity_peak_balanced: max * sqrt(area); ring unused
    - local_contrast_balanced: max(0, p95_inside - ring_mean) * sqrt(area); local contrast over size
    - artifact_consistent_local_contrast: local_contrast * sqrt(area) * sign_consistency; favors one-sign artifacts

    If ``p95_anomaly`` is omitted, intensity modes use ``max_anomaly`` as fallback.
    For ``local_contrast_balanced``, if ``ring_mean_anomaly`` is missing/non-finite, uses ``mean_anomaly``
    (zero local contrast vs interior mean).
    For ``artifact_consistent_local_contrast``, sign_consistency is required; if None, treated as 1.0.
    """
    a = float(max(area, 0.0))
    sqrt_a = float(np.sqrt(a))
    p95 = float(p95_anomaly) if p95_anomaly is not None else float(max_anomaly)

    if mode == "integrated_anomaly":
        return float(a * mean_anomaly)
    if mode == "intensity_size_balanced":
        return float(p95 * sqrt_a)
    if mode == "intensity_peak_balanced":
        return float(max_anomaly * sqrt_a)
    if mode == "local_contrast_balanced":
        rm = ring_mean_anomaly
        if rm is None or not math.isfinite(float(rm)):
            rm = float(mean_anomaly)
        else:
            rm = float(rm)
        contrast = max(0.0, float(p95) - rm)
        return float(contrast * sqrt_a)
    if mode == "artifact_consistent_local_contrast":
        rm = ring_mean_anomaly
        if rm is None or not math.isfinite(float(rm)):
            rm = float(mean_anomaly)
        else:
            rm = float(rm)
        local_contrast = max(0.0, float(p95) - rm)
        sc = float(sign_consistency) if sign_consistency is not None else 1.0
        sc = max(0.0, min(1.0, sc))
        return float(local_contrast * sqrt_a * sc)
    raise ValueError(f"Unknown ranking_mode: {mode!r}")


def contour_keep_decision(
    cnt: np.ndarray,
    img_h: int,
    img_w: int,
    *,
    min_area: float,
    max_area_f: Optional[float],
    max_aspect_ratio: Optional[float],
    min_fill_ratio: Optional[float],
    exclude_border_touching: bool,
    border_margin_px: int,
) -> Tuple[bool, Optional[str]]:
    """
    Returns (keep, reject_reason_or_none). Criteria are evaluated in order;
    reject_reason is the first failing criterion.
    """
    area = float(cv2.contourArea(cnt))
    if area < min_area:
        return False, "min_area"
    if max_area_f is not None and area > max_area_f:
        return False, "max_area"

    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 0 or h <= 0:
        return False, "degenerate_bbox"

    wh = float(w) / float(h)
    hw = float(h) / float(w)
    aspect = max(wh, hw)
    if max_aspect_ratio is not None and aspect > float(max_aspect_ratio):
        return False, "aspect_ratio"

    bbox_area = float(w * h)
    fill = area / bbox_area if bbox_area > 0 else 0.0
    if min_fill_ratio is not None and fill < float(min_fill_ratio):
        return False, "fill_ratio"

    if exclude_border_touching:
        m = int(max(0, border_margin_px))
        touches = (
            x <= m
            or y <= m
            or (x + w) >= (img_w - m)
            or (y + h) >= (img_h - m)
        )
        if touches:
            return False, "border_touch"

    return True, None


class ContourFilterPostprocessor(PostprocessorBase):
    def __init__(self) -> None:
        super().__init__(name="contour_filter_postprocess")

    def run(self, binary_mask_raw, anomaly_map, cfg):
        self.validate_config(cfg)
        self.validate_array(binary_mask_raw, "binary_mask_raw")
        self.validate_array(anomaly_map, "anomaly_map")
        self.validate_same_shape(
            binary_mask_raw, anomaly_map, "binary_mask_raw", "anomaly_map"
        )

        min_area = float(self.get_param(cfg, "min_area", 1.0))
        max_area = self.get_param(cfg, "max_area", None)
        max_area_f = float(max_area) if max_area is not None else None

        max_aspect_ratio = self.get_param(cfg, "max_aspect_ratio", None)
        max_aspect_ratio_f = float(max_aspect_ratio) if max_aspect_ratio is not None else None

        min_fill_ratio = self.get_param(cfg, "min_fill_ratio", None)
        min_fill_ratio_f = float(min_fill_ratio) if min_fill_ratio is not None else None

        exclude_border_touching = bool(self.get_param(cfg, "exclude_border_touching", False))
        border_margin_px = int(self.get_param(cfg, "border_margin_px", 2))

        top_k_raw = self.get_param(cfg, "top_k_keep", None)
        if top_k_raw is None:
            top_k_keep: Optional[int] = None
        else:
            try:
                top_k_keep = int(top_k_raw)
            except (TypeError, ValueError):
                top_k_keep = None
        ranking_mode = str(self.get_param(cfg, "ranking_mode", "integrated_anomaly"))

        min_contour_score_raw = self.get_param(cfg, "min_contour_score", None)
        min_contour_score_f: Optional[float] = None
        if min_contour_score_raw is not None:
            try:
                min_contour_score_f = float(min_contour_score_raw)
            except (TypeError, ValueError):
                min_contour_score_f = None
        contour_score_threshold_mode = str(
            self.get_param(cfg, "contour_score_threshold_mode", "absolute")
        )

        ring_radius_px = int(self.get_param(cfg, "ring_radius_px", 0))
        valid_mask = self.get_param(cfg, "valid_mask", None)
        signed_residual = self.get_param(cfg, "signed_residual", None)
        min_sign_consistency_raw = self.get_param(cfg, "min_sign_consistency", None)
        min_sign_consistency_f: Optional[float] = None
        if min_sign_consistency_raw is not None:
            try:
                min_sign_consistency_f = float(min_sign_consistency_raw)
            except (TypeError, ValueError):
                min_sign_consistency_f = None

        morph_open_kernel = int(self.get_param(cfg, "morph_open_kernel", 0))
        morph_open_iterations = int(self.get_param(cfg, "morph_open_iterations", 0))
        morph_close_kernel = int(self.get_param(cfg, "morph_close_kernel", 0))
        morph_close_iterations = int(self.get_param(cfg, "morph_close_iterations", 0))

        mask_u8 = (np.asarray(binary_mask_raw).astype(bool).astype(np.uint8) * 255)
        positive_pixels_before_morph = int(np.count_nonzero(mask_u8))
        mask_u8 = apply_pre_contour_morphology(
            mask_u8,
            morph_open_kernel=morph_open_kernel,
            morph_open_iterations=morph_open_iterations,
            morph_close_kernel=morph_close_kernel,
            morph_close_iterations=morph_close_iterations,
        )
        positive_pixels_after_morph = int(np.count_nonzero(mask_u8))

        img_h, img_w = int(mask_u8.shape[0]), int(mask_u8.shape[1])
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        reject_counts: Dict[str, int] = {k: 0 for k in REJECT_KEYS}

        areas_all: List[float] = []
        geo_ok: List[np.ndarray] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            areas_all.append(area)

            ok, reason = contour_keep_decision(
                cnt,
                img_h,
                img_w,
                min_area=min_area,
                max_area_f=max_area_f,
                max_aspect_ratio=max_aspect_ratio_f,
                min_fill_ratio=min_fill_ratio_f,
                exclude_border_touching=exclude_border_touching,
                border_margin_px=border_margin_px,
            )
            if not ok:
                if reason in reject_counts:
                    reject_counts[reason] += 1
                continue
            geo_ok.append(cnt)

        anom = np.asarray(anomaly_map, dtype=np.float32)
        sres: Optional[np.ndarray] = None
        if signed_residual is not None:
            sres = np.asarray(signed_residual, dtype=np.float32)
            if sres.shape != anom.shape:
                sres = None
        scored: List[Dict[str, Any]] = []
        _candidate_id = 0
        for cnt in geo_ok:
            c_area = float(cv2.contourArea(cnt))
            if c_area <= 0:
                continue
            mask_c = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.drawContours(mask_c, [cnt], -1, 255, thickness=cv2.FILLED)
            inside = mask_c > 0
            if not np.any(inside):
                continue
            if sres is not None:
                pos_sum, neg_sum, pos_mean, neg_mean, sign_consistency, dominant_sign = _compute_sign_consistency(
                    sres, inside
                )
            else:
                sign_consistency, dominant_sign = 1.0, "neutral"
                pos_sum, neg_sum, pos_mean, neg_mean = 0.0, 0.0, 0.0, 0.0
            if min_sign_consistency_f is not None and sign_consistency < min_sign_consistency_f:
                reject_counts["sign_consistency"] += 1
                continue
            vals = anom[inside]
            mean_a = float(np.mean(vals))
            max_a = float(np.max(vals))
            p95_a = _p95_anomaly_inside_contour(vals, mean_a, max_a)
            if ring_radius_px > 0:
                rm, rp = _ring_mean_and_p95(mask_c, anom, ring_radius_px, valid_mask)
                if not math.isfinite(rm):
                    rm = mean_a
                if not math.isfinite(rp):
                    rp = mean_a
            else:
                rm, rp = mean_a, mean_a

            rm_for_score = float(rm) if math.isfinite(rm) else mean_a
            score = compute_ranking_score(
                ranking_mode,
                c_area,
                mean_a,
                max_a,
                p95_a,
                ring_mean_anomaly=rm_for_score,
                sign_consistency=sign_consistency,
            )
            x, y, w, h = cv2.boundingRect(cnt)
            wh = float(w) / float(h) if h else 0.0
            hw = float(h) / float(w) if w else 0.0
            aspect = max(wh, hw)
            bbox_area = float(max(1, w * h))
            fill_ratio = float(c_area / bbox_area)
            _candidate_id += 1
            scored.append({
                "candidate_id": int(_candidate_id),
                "cnt": cnt,
                "area": float(c_area),
                "mean_anomaly": mean_a,
                "max_anomaly": max_a,
                "p95_anomaly": p95_a,
                "ring_mean_anomaly": float(rm) if math.isfinite(rm) else float(mean_a),
                "ring_p95_anomaly": float(rp) if math.isfinite(rp) else float(mean_a),
                "ranking_score": float(score),
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "aspect_ratio": float(aspect),
                "fill_ratio": float(fill_ratio),
            })

        scored.sort(key=lambda d: d["ranking_score"], reverse=True)
        num_contours_after_geom_filters = int(len(geo_ok))
        num_scored = int(len(scored))

        score_threshold_used: Optional[float] = None
        if min_contour_score_f is None:
            after_threshold = scored
        elif contour_score_threshold_mode == "absolute":
            score_threshold_used = float(min_contour_score_f)
            after_threshold = [
                d for d in scored if float(d["ranking_score"]) >= float(min_contour_score_f)
            ]
        else:
            raise ValueError(
                f"Unsupported contour_score_threshold_mode: {contour_score_threshold_mode!r}. "
                "Expected 'absolute'."
            )

        num_contours_after_score_threshold = int(len(after_threshold))

        # top_k_keep is a maximum cap only (0..K detections), not a quota.
        if top_k_keep is None or top_k_keep <= 0:
            selected = after_threshold
        else:
            selected = after_threshold[: min(int(top_k_keep), len(after_threshold))]

        num_contours_after_topk = int(len(selected))

        after_threshold_ids = {int(d["candidate_id"]) for d in after_threshold}
        selected_ids = {int(d["candidate_id"]) for d in selected}

        contour_audit_rows: List[Dict[str, Any]] = []
        contour_audit_specs: List[Dict[str, Any]] = []
        for d in scored:
            cid = int(d["candidate_id"])
            if cid not in after_threshold_ids:
                st = "score_threshold"
                rr = "score_threshold"
            elif cid not in selected_ids:
                st = "top_k_cap"
                rr = "top_k_cap"
            else:
                st = "kept"
                rr = ""
            contour_audit_rows.append({
                "candidate_id": cid,
                "area": float(d["area"]),
                "ranking_score": float(d["ranking_score"]),
                "mean_inside": float(d["mean_anomaly"]),
                "p95_inside": float(d["p95_anomaly"]),
                "ring_mean": float(d.get("ring_mean_anomaly", d["mean_anomaly"])),
                "sign_consistency": float(d.get("sign_consistency", 1.0)),
                "dominant_sign": d.get("dominant_sign", "neutral"),
                "kept_final": bool(cid in selected_ids),
                "reject_reason": rr,
            })
            contour_audit_specs.append({
                "candidate_id": cid,
                "cnt": d["cnt"],
                "status": st,
            })

        out = np.zeros_like(mask_u8, dtype=np.uint8)
        boxes: List[Dict[str, Any]] = []
        areas_kept: List[float] = []
        for item in selected:
            cnt = item["cnt"]
            cv2.drawContours(out, [cnt], contourIdx=-1, color=255, thickness=cv2.FILLED)
            boxes.append({
                "x": item["x"],
                "y": item["y"],
                "w": item["w"],
                "h": item["h"],
                "area": float(item["area"]),
                "aspect_ratio": float(item["aspect_ratio"]),
                "fill_ratio": float(item["fill_ratio"]),
                "mean_anomaly": float(item["mean_anomaly"]),
                "max_anomaly": float(item["max_anomaly"]),
                "p95_anomaly": float(item["p95_anomaly"]),
                "ring_mean_anomaly": float(item.get("ring_mean_anomaly", item["mean_anomaly"])),
                "ring_p95_anomaly": float(item.get("ring_p95_anomaly", item["mean_anomaly"])),
                "ranking_score": float(item["ranking_score"]),
                "sign_consistency": float(item.get("sign_consistency", 1.0)),
                "dominant_sign": item.get("dominant_sign", "neutral"),
            })
            areas_kept.append(float(item["area"]))

        areas_sorted = sorted([float(a) for a in areas_kept], reverse=True)
        scores_sorted = [float(item["ranking_score"]) for item in selected]
        mean_ranked = [float(item["mean_anomaly"]) for item in selected]
        p95_ranked = [float(item["p95_anomaly"]) for item in selected]
        areas_ranked = [float(item["area"]) for item in selected]
        ring_mean_ranked = [float(item.get("ring_mean_anomaly", item["mean_anomaly"])) for item in selected]
        sign_consistency_ranked = [float(item.get("sign_consistency", 1.0)) for item in selected]
        dominant_sign_ranked = [str(item.get("dominant_sign", "neutral")) for item in selected]

        metadata = {
            "method": self.name,
            "morph_open_kernel": int(morph_open_kernel),
            "morph_open_iterations": int(morph_open_iterations),
            "morph_close_kernel": int(morph_close_kernel),
            "morph_close_iterations": int(morph_close_iterations),
            "positive_pixels_before_morph": int(positive_pixels_before_morph),
            "positive_pixels_after_morph": int(positive_pixels_after_morph),
            "num_contours_total": int(len(contours)),
            "num_contours_scored": int(num_scored),
            "num_contours_after_geom_filters": int(num_contours_after_geom_filters),
            "num_contours_after_score_threshold": int(num_contours_after_score_threshold),
            "num_contours_after_topk": int(num_contours_after_topk),
            "num_contours_kept": int(num_contours_after_topk),
            "num_kept_contours": int(num_contours_after_topk),
            "num_kept_contours_before_topk": int(num_contours_after_score_threshold),
            "num_kept_contours_after_topk": int(num_contours_after_topk),
            "final_num_contours": int(num_contours_after_topk),
            "final_num_centers": int(num_contours_after_topk),
            "min_contour_score": None if min_contour_score_f is None else float(min_contour_score_f),
            "contour_score_threshold_mode": contour_score_threshold_mode,
            "score_threshold_used": score_threshold_used,
            "top_k_keep": None if top_k_keep is None or top_k_keep <= 0 else int(top_k_keep),
            "ranking_mode": ranking_mode,
            "ring_radius_px": int(ring_radius_px),
            "min_area": float(min_area),
            "max_area": None if max_area_f is None else float(max_area_f),
            "max_aspect_ratio": None if max_aspect_ratio_f is None else float(max_aspect_ratio_f),
            "min_fill_ratio": None if min_fill_ratio_f is None else float(min_fill_ratio_f),
            "exclude_border_touching": bool(exclude_border_touching),
            "border_margin_px": int(border_margin_px),
            "reject_counts": dict(reject_counts),
            "contour_areas_all": areas_all,
            "contour_areas_kept": areas_kept,
            "contour_areas_sorted_desc": areas_sorted,
            "top_contour_areas": areas_sorted[:10],
            "ranking_scores_sorted_desc": scores_sorted,
            "top_contour_scores": scores_sorted[:10],
            # Final survivors (score order: highest first).
            "top_scores_ranked": scores_sorted[:10],
            "top_areas_ranked": areas_ranked[:10],
            "top_mean_anomalies_ranked": mean_ranked[:10],
            "top_p95_anomalies_ranked": p95_ranked[:10],
            "top_mean_inside_ranked": mean_ranked[:10],
            "top_p95_inside_ranked": p95_ranked[:10],
            "top_ring_mean_ranked": ring_mean_ranked[:10],
            "top_sign_consistency_ranked": sign_consistency_ranked[:10],
            "top_dominant_sign_ranked": dominant_sign_ranked[:10],
            "min_sign_consistency": min_sign_consistency_f,
            "total_kept_area": float(sum(areas_kept)),
            "num_centers_drawn": int(num_contours_after_topk),
            "components": boxes,
            "bounding_boxes": boxes,
            "contour_audit_rows": contour_audit_rows,
            "contour_audit_specs": contour_audit_specs,
        }
        return out.astype(bool), metadata
