from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

from debug.auto_diagnosis import build_auto_diagnosis
from debug.candidate_analysis import build_candidate_audit_records
from debug.gt_analysis import build_gt_fate_records
from debug.stage_analysis import build_stage_summary_rows
from utils.spatial_diagnostics import border_distance_for_point, nearest_gt, top_peaks_nms


def _nearest_candidate_for_point(candidates: List[Any], x: int, y: int, radius_px: float) -> tuple[int | None, float | None, bool | None]:
    best_id = None
    best_d = None
    best_kept = None
    for c in candidates:
        d = float(np.hypot(float(c.centroid_x) - float(x), float(c.centroid_y) - float(y)))
        if best_d is None or d < best_d:
            best_d = d
            best_id = int(c.candidate_id)
            best_kept = bool(c.kept_final)
    if best_d is None or best_d > float(radius_px):
        return None, best_d, None
    return best_id, best_d, best_kept


def _build_top_peak_records(
    peaks_xyz: List[Tuple[int, int, float]],
    *,
    gt_points: List[Tuple[int, int]],
    candidates: List[Any],
    threshold_mask: np.ndarray | None,
    morph_mask: np.ndarray | None,
    edge_mask: np.ndarray | None,
    candidate_match_radius_px: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if threshold_mask is not None:
        h, w = threshold_mask.shape
    elif morph_mask is not None:
        h, w = morph_mask.shape
    else:
        h = w = 0
    for i, (x, y, v) in enumerate(peaks_xyz, start=1):
        bd = border_distance_for_point(int(x), int(y), int(w), int(h)) if w > 0 and h > 0 else None
        ng_id, ng_d = nearest_gt(gt_points, float(x), float(y))
        thr = bool(threshold_mask[y, x]) if threshold_mask is not None and 0 <= y < threshold_mask.shape[0] and 0 <= x < threshold_mask.shape[1] else None
        mor = bool(morph_mask[y, x]) if morph_mask is not None and 0 <= y < morph_mask.shape[0] and 0 <= x < morph_mask.shape[1] else None
        cid, cd, kept = _nearest_candidate_for_point(candidates, int(x), int(y), radius_px=float(candidate_match_radius_px))
        rows.append(
            {
                "peak_rank": i,
                "x": int(x),
                "y": int(y),
                "value": float(v),
                "border_distance": bd,
                "nearest_gt_id": ng_id,
                "nearest_gt_distance": ng_d,
                "thresholded": thr,
                "survived_morph": mor,
                "candidate_id_if_any": cid,
                "kept_final": kept,
                "edge_flag": (
                    bool(edge_mask[y, x])
                    if edge_mask is not None and 0 <= y < edge_mask.shape[0] and 0 <= x < edge_mask.shape[1]
                    else None
                ),
                "near_border": bool(bd is not None and bd <= 3.0),
            }
        )
    return rows

def build_case_report_payload(
    *,
    pair_id: str,
    cfg: Any,
    artifacts: Any,
    gt_points: List[Tuple[int, int]],
) -> Dict[str, Any]:
    dm = artifacts.decision_metadata or {}
    gt_fates = build_gt_fate_records(
        gt_points,
        artifacts,
        dm,
        gt_match_radius_px=int(getattr(cfg.debug_report, "gt_match_radius_px", 5)),
    )
    candidates = build_candidate_audit_records(
        gt_points,
        dm.get("candidate_audit_records", dm.get("contour_audit_rows", [])),
        gt_match_radius_px=int(getattr(cfg.debug_report, "gt_match_radius_px", 5)),
        edge_mask=getattr(artifacts, "edge_mask", None),
        image_shape=tuple(np.asarray(artifacts.anomaly_map).shape[:2]) if artifacts.anomaly_map is not None else None,
        contour_specs=dm.get("contour_audit_specs"),
        noise_calibrated_candidates=dm.get("noise_calibrated_candidates"),
        morphology_changed=(dm.get("cc_after_close") != dm.get("cc_before_morph")) if dm.get("cc_after_close") is not None and dm.get("cc_before_morph") is not None else None,
        morph_added_mask=(
            np.logical_and(
                np.asarray(dm.get("mask_after_close")).astype(bool),
                np.logical_not(np.asarray(dm.get("mask_after_open")).astype(bool)),
            )
            if dm.get("mask_after_close") is not None and dm.get("mask_after_open") is not None
            else None
        ),
    )
    stage_rows = build_stage_summary_rows(artifacts, dm)
    anomaly = np.asarray(artifacts.anomaly_map, dtype=np.float32) if artifacts.anomaly_map is not None else None
    top_k = int(getattr(cfg.debug_report, "top_peak_count", 20))
    peaks = (
        top_peaks_nms(
            anomaly,
            top_k=min(top_k, int(anomaly.size)),
            min_spacing_px=6,
            valid_mask=(np.asarray(artifacts.valid_mask).astype(bool) if getattr(artifacts, "valid_mask", None) is not None else None),
        )
        if anomaly is not None and anomaly.size > 0
        else []
    )
    thr_mask = np.asarray(artifacts.binary_mask_raw).astype(bool) if artifacts.binary_mask_raw is not None else None
    morph_mask = np.asarray(dm.get("mask_after_morph")).astype(bool) if dm.get("mask_after_morph") is not None else thr_mask
    edge_mask = np.asarray(getattr(artifacts, "edge_mask")).astype(bool) if getattr(artifacts, "edge_mask", None) is not None else None
    peak_rows = _build_top_peak_records(
        peaks,
        gt_points=gt_points,
        candidates=candidates,
        threshold_mask=thr_mask,
        morph_mask=morph_mask,
        edge_mask=edge_mask,
        candidate_match_radius_px=float(getattr(cfg.debug_report, "gt_match_radius_px", 5)),
    )
    final_count = int(dm.get("final_num_contours", dm.get("num_kept_contours", 0)))
    gt_hit_final = int(sum(1 for g in gt_fates if g.kept_final))
    gt_hit_thr = int(sum(1 for g in gt_fates if g.threshold_support_r5))
    gt_hit_morph = int(sum(1 for g in gt_fates if g.survived_morph))
    header = {
        "pair_id": pair_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_name": getattr(cfg, "config_name", None),
        "stages": {
            "alignment": cfg.choices.alignment,
            "normalization": cfg.choices.normalization,
            "comparison": cfg.choices.comparison,
            "thresholding": cfg.choices.thresholding,
            "postprocessing": cfg.choices.postprocessing,
        },
        "objective": {
            "gt_points_expected": len(gt_points),
            "final_detection_count": final_count,
            "gt_points_hit_by_final_mask": gt_hit_final,
            "gt_points_hit_by_threshold_mask": gt_hit_thr,
            "gt_points_hit_by_post_morph_mask": gt_hit_morph,
            "final_status": "PASS" if (len(gt_points) == 0 and final_count == 0) or (len(gt_points) > 0 and gt_hit_final == len(gt_points)) else "FAIL",
        },
    }
    diagnosis = build_auto_diagnosis(
        gt_fates,
        final_count=final_count,
        gt_expected=len(gt_points),
        top_peaks=peak_rows,
        candidates=candidates,
    )
    max_rej = int(getattr(cfg.debug_report, "max_rejected_candidates_to_print", 15))
    kept = [c for c in candidates if c.kept_final]
    gt_matched = [c for c in candidates if c.gt_match]
    rejected = sorted([c for c in candidates if not c.kept_final], key=lambda c: float(c.score or 0.0), reverse=True)
    shown_ids = set()
    shown_candidates = []
    for bucket in (kept, gt_matched, rejected[:max_rej]):
        for c in bucket:
            if c.candidate_id in shown_ids:
                continue
            shown_ids.add(c.candidate_id)
            shown_candidates.append(c)
    fp_autopsy = [c for c in kept if len(gt_points) == 0]
    gt_percentiles = [float(g.anomaly_percentile) for g in gt_fates if g.anomaly_percentile is not None]
    non_gt_border_peaks = sum(1 for p in peak_rows if p["nearest_gt_id"] is None and bool(p.get("near_border", False)))
    border_dominated = bool(peak_rows) and (non_gt_border_peaks / max(1, len(peak_rows))) >= 0.5
    return {
        "header": header,
        "gt_fates": gt_fates,
        "candidates": candidates,
        "shown_candidates": shown_candidates,
        "false_positive_autopsy": fp_autopsy,
        "stage_rows": stage_rows,
        "top_peaks": peak_rows,
        "comparator_summary": {
            "gt_signal_quality_mean_percentile": (float(np.mean(np.asarray(gt_percentiles, dtype=np.float32))) if gt_percentiles else None),
            "gt_signal_quality_min_percentile": (float(np.min(np.asarray(gt_percentiles, dtype=np.float32))) if gt_percentiles else None),
            "non_gt_top_peaks_near_border_count": int(non_gt_border_peaks),
            "top_peak_count": int(len(peak_rows)),
            "border_dominated": bool(border_dominated),
        },
        "auto_diagnosis": diagnosis,
    }


def render_case_report_text(payload: Dict[str, Any]) -> str:
    h = payload["header"]
    obj = h["objective"]
    gt_fates = payload["gt_fates"]
    candidates = payload["shown_candidates"]
    stage_rows = payload["stage_rows"]
    peaks = payload["top_peaks"]
    comp = payload.get("comparator_summary", {})
    lines: List[str] = []
    lines.append("A. HEADER / RUN IDENTITY")
    lines.append(f"pair_id: {h['pair_id']}")
    lines.append(f"timestamp: {h['timestamp']}")
    lines.append(f"config_name: {h.get('config_name')}")
    lines.append(f"modules: {h['stages']}")
    lines.append(f"objective: {obj}")
    lines.append("")
    lines.append("B. GT DEFECT FATE SUMMARY")
    if not gt_fates:
        lines.append("No GT points for this case.")
    for g in gt_fates:
        lines.append(
            f"gt_id={g.gt_id} xy=({g.x},{g.y}) inside_valid_overlap={g.inside_valid_overlap} "
            f"anomaly_at_gt={g.anomaly_at_gt} anomaly_local_max_r5={g.anomaly_local_max_r5} "
            f"anomaly_percentile={g.anomaly_percentile} above_threshold_at_gt={g.above_threshold} "
            f"threshold_support_r5={g.threshold_support_r5} threshold_component_id={g.threshold_component_id} "
            f"threshold_component_area={g.threshold_component_area} survived_morphology={g.survived_morph} "
            f"morph_component_id={g.morph_component_id} candidate_id={g.candidate_id} "
            f"contour_contains_gt={g.contour_contains_gt} contour_centroid_distance_to_gt_px={g.contour_centroid_distance_to_gt_px} "
            f"candidate_score={g.candidate_score} contour_rank={g.candidate_rank} contour_kept_final={g.kept_final} "
            f"final_component_id={g.final_component_id} failure_stage={g.failure_stage} "
            f"rejection_reason={g.rejection_reason} short_details={g.short_details}"
        )
    lines.append("")
    lines.append("C. STAGE HEALTH SUMMARY")
    for stage, metric, value in stage_rows:
        lines.append(f"{stage}: {metric}={value}")
    lines.append("")
    lines.append("D. ALIGNMENT DIAGNOSIS")
    lines.append("Derived from alignment metadata and stage summary metrics.")
    lines.append("")
    lines.append("E. COMPARATOR DIAGNOSIS")
    for p in peaks:
        lines.append(
            f"peak_rank={p['peak_rank']} x={p['x']} y={p['y']} value={p['value']} "
            f"border_distance={p['border_distance']} nearest_gt_id={p['nearest_gt_id']} nearest_gt_distance={p['nearest_gt_distance']} "
            f"thresholded={p['thresholded']} survived_morph={p['survived_morph']} "
            f"candidate_id_if_any={p['candidate_id_if_any']} kept_final={p['kept_final']} edge_flag={p.get('edge_flag')}"
        )
    lines.append(
        f"gt_signal_quality_mean_percentile={comp.get('gt_signal_quality_mean_percentile')} "
        f"gt_signal_quality_min_percentile={comp.get('gt_signal_quality_min_percentile')} "
        f"non_gt_top_peaks_near_border_count={comp.get('non_gt_top_peaks_near_border_count')}/{comp.get('top_peak_count')} "
        f"border_dominated={comp.get('border_dominated')}"
    )
    lines.append("")
    lines.append("F. THRESHOLD DIAGNOSIS")
    lines.append("Threshold effects are inferred from GT threshold coverage and threshold-stage components.")
    lines.append("")
    lines.append("G. MORPHOLOGY DIAGNOSIS")
    lines.append("Morphology effects are inferred from before/after mask component and GT support transitions.")
    lines.append("")
    lines.append("H. CANDIDATE AUDIT")
    for c in candidates:
        lines.append(
            f"candidate_id={c.candidate_id} centroid=({c.centroid_x:.2f},{c.centroid_y:.2f}) area={c.area:.2f} "
            f"bbox=({c.bbox_x},{c.bbox_y},{c.bbox_w},{c.bbox_h}) aspect_ratio={c.aspect_ratio} fill_ratio={c.fill_ratio} "
            f"border_touching={c.border_touching} "
            f"border_distance={c.border_distance} nearest_gt_id={c.nearest_gt_id} nearest_gt_distance={c.nearest_gt_distance} "
            f"gt_match={c.gt_match} mean_anomaly={c.mean_anomaly} p95_anomaly={c.p95_anomaly} ring_mean={c.ring_mean} "
            f"local_contrast={c.local_contrast} sign_consistency={c.sign_consistency} dominant_sign={c.dominant_sign} "
            f"score={c.score} rank={c.rank} pass_area={c.pass_area} pass_aspect_ratio={c.pass_aspect_ratio} "
            f"pass_fill_ratio={c.pass_fill_ratio} pass_border={c.pass_border} "
            f"pass_sign_consistency={c.pass_sign_consistency} final_status={'KEPT' if c.kept_final else 'REJECTED'} "
            f"rejection_reason={c.rejection_reason}"
        )
    if obj["gt_points_expected"] == 0:
        lines.append("")
        lines.append("FALSE POSITIVE AUTOPSY")
        for c in payload.get("false_positive_autopsy", []):
            lines.append(
                f"candidate_id={c.candidate_id} centroid=({c.centroid_x:.2f},{c.centroid_y:.2f}) area={c.area:.2f} "
                f"border_distance={c.border_distance} border_touching={c.border_touching} "
                f"mean_anomaly={c.mean_anomaly} p95_anomaly={c.p95_anomaly} ring_mean={c.ring_mean} "
                f"local_contrast={c.local_contrast} sign_consistency={c.sign_consistency} dominant_sign={c.dominant_sign} "
                f"edge_overlap_fraction={c.edge_overlap_fraction} border_overlap_fraction={c.border_overlap_fraction} "
                f"likely_origin={c.likely_origin}"
            )
    lines.append("")
    lines.append("I. FINAL TARGET ASSESSMENT")
    lines.append(
        f"expected_gt_count={obj['gt_points_expected']} final_detection_count={obj['final_detection_count']} "
        f"gt_points_hit={obj['gt_points_hit_by_final_mask']} matching_rule=within_radius_5px"
    )
    lines.append("")
    lines.append("J. AUTO DIAGNOSIS")
    lines.append(payload["auto_diagnosis"])
    return "\n".join(lines) + "\n"

