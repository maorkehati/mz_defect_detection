from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from debug.report_types import GTFateRecord
from utils.spatial_diagnostics import local_max_around_point, percentile_in_valid_region


def _distance_to_invalid_border(valid_mask: np.ndarray, x: int, y: int) -> Optional[float]:
    vm = np.asarray(valid_mask).astype(bool)
    if vm.ndim != 2 or not (0 <= y < vm.shape[0] and 0 <= x < vm.shape[1]):
        return None
    inv = (~vm).astype(np.uint8)
    dist = cv2.distanceTransform(1 - inv, cv2.DIST_L2, 3)
    return float(dist[y, x])


def _point_inside_bbox(x: int, y: int, bx: Optional[int], by: Optional[int], bw: Optional[int], bh: Optional[int]) -> bool:
    if bx is None or by is None or bw is None or bh is None:
        return False
    return bool(int(bx) <= int(x) < int(bx) + int(bw) and int(by) <= int(y) < int(by) + int(bh))


def _nearest_candidate(candidate_rows: Iterable[Dict[str, Any]], x: int, y: int) -> Tuple[Optional[int], Optional[float], Optional[Dict[str, Any]]]:
    best = None
    best_id = None
    best_d = None
    for c in candidate_rows:
        cx = c.get("centroid_x")
        cy = c.get("centroid_y")
        if cx is None or cy is None:
            continue
        d = float(np.hypot(float(cx) - float(x), float(cy) - float(y)))
        if best_d is None or d < best_d:
            best = c
            best_d = d
            best_id = int(c.get("candidate_id", 0))
    return best_id, best_d, best


def _row_score(row: Dict[str, Any]) -> float:
    s = row.get("score", row.get("ranking_score", 0.0))
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _candidate_ids_containing_point(
    contour_audit_specs: Iterable[Dict[str, Any]],
    x: int,
    y: int,
) -> List[int]:
    ids: List[int] = []
    pt = (float(x), float(y))
    for spec in contour_audit_specs:
        cid = spec.get("candidate_id")
        cnt = spec.get("cnt")
        if cid is None or cnt is None:
            continue
        try:
            inside = cv2.pointPolygonTest(np.asarray(cnt), pt, False)
        except Exception:
            inside = -1.0
        if float(inside) >= 0.0:
            ids.append(int(cid))
    return ids


def _connected_component_labels(mask: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Dict[int, int]]:
    if mask is None:
        return None, {}
    bm = np.asarray(mask).astype(bool)
    if bm.ndim != 2 or bm.size == 0:
        return None, {}
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bm.astype(np.uint8))
    areas = {int(i): int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, int(n))}
    return labels, areas


def _component_id_within_radius(labels: Optional[np.ndarray], x: int, y: int, r: int) -> Optional[int]:
    if labels is None:
        return None
    h, w = labels.shape
    if not (0 <= x < w and 0 <= y < h):
        return None
    if int(labels[y, x]) > 0:
        return int(labels[y, x])
    rr = int(max(0, r))
    x0, x1 = max(0, x - rr), min(w, x + rr + 1)
    y0, y1 = max(0, y - rr), min(h, y + rr + 1)
    patch = labels[y0:y1, x0:x1]
    ys, xs = np.where(patch > 0)
    if ys.size == 0:
        return None
    dx = xs.astype(np.float32) + float(x0 - x)
    dy = ys.astype(np.float32) + float(y0 - y)
    d2 = dx * dx + dy * dy
    idx = int(np.argmin(d2))
    return int(patch[ys[idx], xs[idx]])


def infer_failure_stage(
    *,
    inside_valid_overlap: bool,
    anomaly_percentile: Optional[float],
    local_alignment_residual_mean_abs: Optional[float],
    global_alignment_residual_p95_abs: Optional[float],
    any_threshold_pixel_within_r5: Optional[bool],
    threshold_support_r5: Optional[bool],
    survived_morph: Optional[bool],
    nearest_candidate: Optional[Dict[str, Any]],
    kept_final: bool,
) -> tuple[str, str]:
    """Deterministic earliest-failure GT taxonomy with explainable short details."""
    if kept_final:
        return "NONE", "GT is matched by final detection."
    if not inside_valid_overlap:
        return "OUTSIDE_VALID_OVERLAP", "GT lies outside valid overlap."
    if (
        local_alignment_residual_mean_abs is not None
        and global_alignment_residual_p95_abs is not None
        and local_alignment_residual_mean_abs > (1.35 * global_alignment_residual_p95_abs)
    ):
        return "ALIGNMENT_WEAK_SIGNAL", "Local alignment residual is unusually high."
    if anomaly_percentile is not None and anomaly_percentile < 70.0 and any_threshold_pixel_within_r5 is False:
        return "COMPARATOR_WEAK_SIGNAL", "Comparator response is weak near GT."
    if threshold_support_r5 is False:
        return "THRESHOLD", "Threshold stage removed GT support."
    if survived_morph is False:
        return "MORPHOLOGY", "Morphology removed GT support."
    if nearest_candidate is None:
        return "NO_CONTOUR_FORMED", "No candidate region formed near GT."
    reason = str(nearest_candidate.get("rejection_reason", nearest_candidate.get("reject_reason", "")) or "")
    if reason == "min_area":
        return "FILTER_AREA", "Candidate failed min-area filter."
    if reason == "aspect_ratio":
        return "FILTER_ASPECT_RATIO", "Candidate failed aspect-ratio filter."
    if reason == "fill_ratio":
        return "FILTER_FILL_RATIO", "Candidate failed fill-ratio filter."
    if reason == "border_touch":
        return "FILTER_BORDER", "Candidate failed border filter."
    if reason == "sign_consistency":
        return "FILTER_SIGN_CONSISTENCY", "Candidate failed sign-consistency filter."
    if reason in ("top_k_cap", "RANKED_OUT_TOPK"):
        return "RANKED_OUT_TOPK", "Candidate survived hard filters but lost top-k ranking."
    if reason in ("score_threshold", "final_mask_mismatch"):
        return "FINAL_MASK_MISMATCH", "Candidate exists but final mask has no GT match."
    return "UNKNOWN", "Unable to determine a more specific failure stage."


def build_gt_fate_records(
    gt_points: List[Tuple[int, int]],
    artifacts: Any,
    decision_metadata: Dict[str, Any],
    *,
    gt_match_radius_px: int,
) -> List[GTFateRecord]:
    anomaly = np.asarray(artifacts.anomaly_map, dtype=np.float32) if artifacts.anomaly_map is not None else None
    threshold_map = np.asarray(artifacts.threshold_map, dtype=np.float32) if artifacts.threshold_map is not None else None
    valid_mask = np.asarray(artifacts.valid_mask).astype(bool) if artifacts.valid_mask is not None else None
    raw_mask = np.asarray(artifacts.binary_mask_raw).astype(bool) if artifacts.binary_mask_raw is not None else None
    final_mask = np.asarray(artifacts.binary_mask_final).astype(bool) if artifacts.binary_mask_final is not None else None
    post_morph = decision_metadata.get("mask_after_morph")
    post_morph_mask = np.asarray(post_morph).astype(bool) if post_morph is not None else raw_mask
    rows = list(
        decision_metadata.get(
            "candidate_audit_records",
            decision_metadata.get("candidate_records", []),
        )
        or []
    )
    contour_audit_specs = list(decision_metadata.get("contour_audit_specs", []) or [])
    row_by_id = {
        int(r.get("candidate_id", 0)): r
        for r in rows
        if r.get("candidate_id") is not None
    }
    ranked = sorted(rows, key=lambda r: float(r.get("score", r.get("ranking_score", 0.0)) or 0.0), reverse=True)
    rank_by_id = {int(r.get("candidate_id", 0)): i + 1 for i, r in enumerate(ranked)}
    thr_labels, thr_areas = _connected_component_labels(raw_mask)
    morph_labels, _ = _connected_component_labels(post_morph_mask)
    final_labels, _ = _connected_component_labels(final_mask)
    align_residual = np.asarray(artifacts.alignment_residual_map, dtype=np.float32) if getattr(artifacts, "alignment_residual_map", None) is not None else None
    align_abs = np.abs(align_residual) if align_residual is not None else None
    global_align_p95 = float(np.percentile(align_abs, 95.0)) if align_abs is not None and align_abs.size > 0 else None

    records: List[GTFateRecord] = []
    for i, (x, y) in enumerate(gt_points, start=1):
        inside_valid = bool(valid_mask[y, x]) if valid_mask is not None and 0 <= y < valid_mask.shape[0] and 0 <= x < valid_mask.shape[1] else True
        d_border = _distance_to_invalid_border(valid_mask, x, y) if valid_mask is not None else None
        anomaly_at = float(anomaly[y, x]) if anomaly is not None and 0 <= y < anomaly.shape[0] and 0 <= x < anomaly.shape[1] else None
        local_max = local_max_around_point(anomaly, x, y, radius=5) if anomaly is not None else None
        percentile = percentile_in_valid_region(anomaly, anomaly_at, valid_mask) if (anomaly is not None and anomaly_at is not None) else None
        threshold_value = float(np.mean(threshold_map)) if threshold_map is not None else None
        above_thr = (anomaly_at >= threshold_value) if (anomaly_at is not None and threshold_value is not None) else None
        thr_support = (local_max_around_point(raw_mask.astype(np.uint8), x, y, radius=5) or 0.0) > 0 if raw_mask is not None else None
        morph_support = (local_max_around_point(post_morph_mask.astype(np.uint8), x, y, radius=5) or 0.0) > 0 if post_morph_mask is not None else None
        threshold_component_id = _component_id_within_radius(thr_labels, x, y, 5)
        threshold_component_area = int(thr_areas.get(int(threshold_component_id), 0)) if threshold_component_id is not None else None
        morph_component_id = _component_id_within_radius(morph_labels, x, y, 5)
        final_component_id = _component_id_within_radius(final_labels, x, y, 5)
        containing_ids = _candidate_ids_containing_point(contour_audit_specs, x, y)
        nearest_id, nearest_d, nearest = _nearest_candidate(rows, x, y)
        contour_contains = False
        if containing_ids:
            containing_rows = [row_by_id[cid] for cid in containing_ids if cid in row_by_id]
            if containing_rows:
                containing_rows.sort(
                    key=lambda r: (
                        bool(r.get("kept_final", False)),
                        _row_score(r),
                    ),
                    reverse=True,
                )
                nearest = containing_rows[0]
                nearest_id = int(nearest.get("candidate_id", 0))
                cx = nearest.get("centroid_x")
                cy = nearest.get("centroid_y")
                if cx is not None and cy is not None:
                    nearest_d = float(np.hypot(float(cx) - float(x), float(cy) - float(y)))
                contour_contains = True
        if nearest is not None and not contour_contains:
            contour_contains = _point_inside_bbox(
                x, y,
                nearest.get("bbox_x"),
                nearest.get("bbox_y"),
                nearest.get("bbox_w"),
                nearest.get("bbox_h"),
            )
        is_match = bool((nearest_d is not None and nearest_d <= float(gt_match_radius_px)) or contour_contains)
        candidate_present = bool(nearest is not None and morph_support is True)
        kept = bool(is_match and nearest is not None and bool(nearest.get("kept_final", False)))
        local_align_mean = None
        if align_abs is not None:
            y0, y1 = max(0, y - 5), min(align_abs.shape[0], y + 6)
            x0, x1 = max(0, x - 5), min(align_abs.shape[1], x + 6)
            patch = align_abs[y0:y1, x0:x1]
            if patch.size > 0:
                local_align_mean = float(np.mean(patch))
        stage, short_details = infer_failure_stage(
            inside_valid_overlap=inside_valid,
            anomaly_percentile=percentile,
            local_alignment_residual_mean_abs=local_align_mean,
            global_alignment_residual_p95_abs=global_align_p95,
            any_threshold_pixel_within_r5=thr_support,
            threshold_support_r5=thr_support,
            survived_morph=morph_support,
            nearest_candidate=nearest if (is_match or candidate_present) else None,
            kept_final=kept,
        )
        records.append(
            GTFateRecord(
                gt_id=i,
                x=int(x),
                y=int(y),
                inside_valid_overlap=inside_valid,
                distance_to_invalid_border_px=d_border,
                anomaly_at_gt=anomaly_at,
                anomaly_local_max_r5=local_max,
                anomaly_percentile=percentile,
                above_threshold=above_thr,
                threshold_support_r5=thr_support,
                threshold_component_id=threshold_component_id,
                threshold_component_area=threshold_component_area,
                survived_morph=morph_support,
                morph_component_id=morph_component_id,
                candidate_id=nearest_id if (is_match or candidate_present) else None,
                contour_contains_gt=bool(contour_contains or (nearest_d is not None and nearest_d <= float(gt_match_radius_px))),
                contour_centroid_distance_to_gt_px=nearest_d,
                candidate_score=(float(nearest.get("score")) if nearest is not None and nearest.get("score") is not None else float(nearest.get("ranking_score")) if nearest is not None and nearest.get("ranking_score") is not None else None),
                candidate_rank=rank_by_id.get(nearest_id) if nearest_id is not None and (is_match or candidate_present) else None,
                kept_final=kept,
                final_component_id=final_component_id,
                failure_stage=stage,
                rejection_reason=str((nearest or {}).get("rejection_reason", (nearest or {}).get("reject_reason", "")) if is_match else ""),
                short_details=short_details,
                extras={
                    "threshold_value": threshold_value,
                    "nearest_gt_distance": nearest_d,
                    "local_alignment_residual_mean_abs": local_align_mean,
                    "global_alignment_residual_p95_abs": global_align_p95,
                },
            )
        )
    return records

