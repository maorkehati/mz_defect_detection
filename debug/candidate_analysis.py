from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from debug.report_types import CandidateAuditRecord


def _nearest_gt(gt_points: List[Tuple[int, int]], cx: float, cy: float) -> Tuple[Optional[int], Optional[float]]:
    if not gt_points:
        return None, None
    best_i = None
    best_d = None
    for i, (x, y) in enumerate(gt_points, start=1):
        d = float(np.hypot(float(x) - cx, float(y) - cy))
        if best_d is None or d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d


def build_candidate_audit_records(
    gt_points: List[Tuple[int, int]],
    contour_audit_rows: Iterable[Dict[str, Any]],
    *,
    gt_match_radius_px: int,
    edge_mask: Optional[np.ndarray] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    contour_specs: Optional[Iterable[Dict[str, Any]]] = None,
    noise_calibrated_candidates: Optional[Iterable[Dict[str, Any]]] = None,
    morphology_changed: Optional[bool] = None,
    morph_added_mask: Optional[np.ndarray] = None,
) -> List[CandidateAuditRecord]:
    def _normalize_reason(raw: str, kept: bool) -> str:
        if kept:
            return ""
        v = str(raw or "").strip()
        if v == "min_area":
            return "min_area"
        if v == "max_area":
            return "max_area"
        if v in ("aspect_ratio", "max_aspect_ratio"):
            return "max_aspect_ratio"
        if v in ("fill_ratio", "min_fill_ratio"):
            return "min_fill_ratio"
        if v in ("border_touch", "border_touching"):
            return "border_touching"
        if v == "sign_consistency":
            return "sign_consistency"
        if v in ("RANKED_OUT_TOPK", "ranked_out_topk", "top_k_cap"):
            return "ranked_out_topk"
        return "unknown"

    def _candidate_mask(candidate_id: int, row: Dict[str, Any]) -> Optional[np.ndarray]:
        if contour_specs is not None and image_shape is not None:
            for s in contour_specs:
                if int(s.get("candidate_id", -1)) != int(candidate_id):
                    continue
                cnt = s.get("cnt")
                if cnt is None:
                    continue
                m = np.zeros(image_shape, dtype=np.uint8)
                cv2.drawContours(m, [np.asarray(cnt)], -1, 1, thickness=cv2.FILLED)
                return m.astype(bool)
        if noise_calibrated_candidates is not None:
            for c in noise_calibrated_candidates:
                if int(c.get("candidate_id", -1)) == int(candidate_id) and c.get("mask") is not None:
                    return np.asarray(c.get("mask")).astype(bool)
        bx, by, bw, bh = row.get("bbox_x"), row.get("bbox_y"), row.get("bbox_w"), row.get("bbox_h")
        if image_shape is not None and None not in (bx, by, bw, bh):
            m = np.zeros(image_shape, dtype=bool)
            x0, y0 = int(bx), int(by)
            x1, y1 = x0 + int(bw), y0 + int(bh)
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(image_shape[1], x1)
            y1 = min(image_shape[0], y1)
            if x0 < x1 and y0 < y1:
                m[y0:y1, x0:x1] = True
                return m
        return None

    def _origin_heuristic(
        *,
        border_distance: Optional[float],
        border_overlap_fraction: Optional[float],
        edge_overlap_fraction: Optional[float],
        morphology_changed_local: bool,
    ) -> str:
        if border_distance is not None and border_distance <= 2.0:
            if border_overlap_fraction is not None and border_overlap_fraction >= 0.6:
                return "crop boundary mismatch"
            return "border artifact"
        if edge_overlap_fraction is not None and edge_overlap_fraction >= 0.55:
            return "structural mismatch"
        if morphology_changed_local:
            return "morphology merge artifact"
        return "unknown"

    out: List[CandidateAuditRecord] = []
    ranked = sorted(
        list(contour_audit_rows),
        key=lambda r: float(r.get("score", r.get("ranking_score", 0.0)) or 0.0),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        cx = float(row.get("centroid_x", 0.0))
        cy = float(row.get("centroid_y", 0.0))
        ng_id, ng_d = _nearest_gt(gt_points, cx, cy)
        gt_match = bool(ng_d is not None and ng_d <= float(gt_match_radius_px))
        ring = row.get("ring_mean", row.get("ring_mean_anomaly"))
        p95 = row.get("p95_inside", row.get("p95_anomaly"))
        local_contrast = None
        if ring is not None and p95 is not None:
            local_contrast = float(p95) - float(ring)
        kept = bool(row.get("kept_final", False))
        reason = _normalize_reason(str(row.get("rejection_reason", row.get("reject_reason", ""))), kept)
        c_mask = _candidate_mask(int(row.get("candidate_id", rank)), row)
        edge_overlap_fraction = None
        border_overlap_fraction = None
        likely_origin = None
        if c_mask is not None and np.any(c_mask):
            if edge_mask is not None:
                em = np.asarray(edge_mask).astype(bool)
                if em.shape == c_mask.shape:
                    edge_overlap_fraction = float(np.count_nonzero(np.logical_and(c_mask, em)) / np.count_nonzero(c_mask))
            if image_shape is not None:
                border = np.zeros(image_shape, dtype=bool)
                border[:3, :] = True
                border[-3:, :] = True
                border[:, :3] = True
                border[:, -3:] = True
                border_overlap_fraction = float(np.count_nonzero(np.logical_and(c_mask, border)) / np.count_nonzero(c_mask))
        local_morph_merge = bool(morphology_changed)
        if c_mask is not None and morph_added_mask is not None:
            mam = np.asarray(morph_added_mask).astype(bool)
            if mam.shape == c_mask.shape:
                local_morph_merge = bool(np.any(np.logical_and(c_mask, mam)))
        if not gt_points and kept:
            likely_origin = _origin_heuristic(
                border_distance=row.get("border_distance"),
                border_overlap_fraction=border_overlap_fraction,
                edge_overlap_fraction=edge_overlap_fraction,
                morphology_changed_local=local_morph_merge,
            )
        out.append(
            CandidateAuditRecord(
                candidate_id=int(row.get("candidate_id", rank)),
                centroid_x=cx,
                centroid_y=cy,
                area=float(row.get("area", 0.0)),
                bbox_x=row.get("bbox_x"),
                bbox_y=row.get("bbox_y"),
                bbox_w=row.get("bbox_w"),
                bbox_h=row.get("bbox_h"),
                aspect_ratio=row.get("aspect_ratio"),
                fill_ratio=row.get("fill_ratio"),
                border_touching=row.get("border_touching"),
                border_distance=row.get("border_distance"),
                nearest_gt_id=ng_id,
                nearest_gt_distance=ng_d,
                gt_match=gt_match,
                mean_anomaly=row.get("mean_inside", row.get("mean_anomaly")),
                p95_anomaly=row.get("p95_inside", row.get("p95_anomaly")),
                ring_mean=ring,
                local_contrast=local_contrast,
                sign_consistency=row.get("sign_consistency"),
                dominant_sign=row.get("dominant_sign"),
                score=row.get("score", row.get("ranking_score")),
                rank=rank,
                pass_area=row.get("pass_area"),
                pass_aspect_ratio=row.get("pass_aspect_ratio"),
                pass_fill_ratio=row.get("pass_fill_ratio"),
                pass_border=row.get("pass_border", row.get("pass_border_touching")),
                pass_sign_consistency=row.get("pass_sign_consistency"),
                kept_final=kept,
                rejection_reason=reason,
                edge_overlap_fraction=edge_overlap_fraction,
                border_overlap_fraction=border_overlap_fraction,
                likely_origin=likely_origin,
            )
        )
    return out

