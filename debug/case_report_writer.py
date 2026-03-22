from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from debug.case_report_builder import build_case_report_payload, render_case_report_text
from utils.ground_truth_defects import get_ground_truth_points_for_pair, pair_id_to_case_key


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_case_report_bundle(pair_id: str, cfg: Any, artifacts: Any, repo_root: Path) -> Path:
    gt_points = get_ground_truth_points_for_pair(pair_id)
    payload = build_case_report_payload(pair_id=pair_id, cfg=cfg, artifacts=artifacts, gt_points=gt_points)
    out_root = Path(getattr(cfg.debug_report, "debug_report_output_dir", None) or (repo_root / "outs" / "debug_reports"))
    case_dir = out_root / (pair_id_to_case_key(pair_id) or pair_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    report_path = case_dir / "pipeline_case_report.txt"
    report_path.write_text(render_case_report_text(payload), encoding="utf-8")

    gt_rows = [
        {
            "gt_id": r.gt_id,
            "x": r.x,
            "y": r.y,
            "inside_valid_overlap": r.inside_valid_overlap,
            "distance_to_invalid_border_px": r.distance_to_invalid_border_px,
            "anomaly_at_gt": r.anomaly_at_gt,
            "anomaly_local_max_r5": r.anomaly_local_max_r5,
            "anomaly_percentile": r.anomaly_percentile,
            "above_threshold_at_gt": r.above_threshold,
            "any_threshold_pixel_within_r5": r.threshold_support_r5,
            "threshold_component_id": r.threshold_component_id,
            "threshold_component_area": r.threshold_component_area,
            "survived_morph": r.survived_morph,
            "morph_component_id": r.morph_component_id,
            "candidate_id": r.candidate_id,
            "contour_contains_gt": r.contour_contains_gt,
            "contour_centroid_distance_to_gt_px": r.contour_centroid_distance_to_gt_px,
            "candidate_score": r.candidate_score,
            "candidate_rank": r.candidate_rank,
            "contour_kept_final": r.kept_final,
            "final_component_id": r.final_component_id,
            "failure_stage": r.failure_stage,
            "rejection_reason": r.rejection_reason,
            "short_details": r.short_details,
        }
        for r in payload["gt_fates"]
    ]
    _write_csv(
        case_dir / "gt_fate.csv",
        gt_rows,
        [
            "gt_id", "x", "y", "inside_valid_overlap", "distance_to_invalid_border_px",
            "anomaly_at_gt", "anomaly_local_max_r5", "anomaly_percentile", "above_threshold_at_gt",
            "any_threshold_pixel_within_r5", "threshold_component_id", "threshold_component_area",
            "survived_morph", "morph_component_id", "candidate_id", "contour_contains_gt",
            "contour_centroid_distance_to_gt_px", "candidate_score", "candidate_rank",
            "contour_kept_final", "final_component_id", "failure_stage", "rejection_reason", "short_details",
        ],
    )

    cand_rows = [
        {
            "candidate_id": c.candidate_id,
            "centroid_x": c.centroid_x,
            "centroid_y": c.centroid_y,
            "area": c.area,
            "bbox_x": c.bbox_x,
            "bbox_y": c.bbox_y,
            "bbox_w": c.bbox_w,
            "bbox_h": c.bbox_h,
            "aspect_ratio": c.aspect_ratio,
            "fill_ratio": c.fill_ratio,
            "border_touching": c.border_touching,
            "border_distance": c.border_distance,
            "nearest_gt_id": c.nearest_gt_id,
            "nearest_gt_distance": c.nearest_gt_distance,
            "gt_match": c.gt_match,
            "mean_anomaly": c.mean_anomaly,
            "p95_anomaly": c.p95_anomaly,
            "ring_mean": c.ring_mean,
            "local_contrast": c.local_contrast,
            "sign_consistency": c.sign_consistency,
            "dominant_sign": c.dominant_sign,
            "score": c.score,
            "rank": c.rank,
            "pass_area": c.pass_area,
            "pass_aspect_ratio": c.pass_aspect_ratio,
            "pass_fill_ratio": c.pass_fill_ratio,
            "pass_border": c.pass_border,
            "pass_sign_consistency": c.pass_sign_consistency,
            "kept_final": c.kept_final,
            "rejection_reason": c.rejection_reason,
            "edge_overlap_fraction": c.edge_overlap_fraction,
            "border_overlap_fraction": c.border_overlap_fraction,
            "likely_origin": c.likely_origin,
        }
        for c in payload["candidates"]
    ]
    _write_csv(
        case_dir / "candidate_audit.csv",
        cand_rows,
        [
            "candidate_id", "centroid_x", "centroid_y", "area", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "aspect_ratio", "fill_ratio",
            "border_touching", "border_distance", "nearest_gt_id", "nearest_gt_distance", "gt_match",
            "mean_anomaly", "p95_anomaly", "ring_mean", "local_contrast", "sign_consistency", "dominant_sign",
            "score", "rank", "pass_area", "pass_aspect_ratio", "pass_fill_ratio", "pass_border",
            "pass_sign_consistency", "kept_final", "rejection_reason", "edge_overlap_fraction",
            "border_overlap_fraction", "likely_origin",
        ],
    )

    _write_csv(
        case_dir / "stage_summary.csv",
        [{"stage": s, "metric": m, "value": v} for s, m, v in payload["stage_rows"]],
        ["stage", "metric", "value"],
    )

    _write_csv(
        case_dir / "top_peaks.csv",
        [
            {
                "peak_rank": p.get("peak_rank"),
                "x": p.get("x"),
                "y": p.get("y"),
                "value": p.get("value"),
                "border_distance": p.get("border_distance"),
                "nearest_gt_id": p.get("nearest_gt_id"),
                "nearest_gt_distance": p.get("nearest_gt_distance"),
                "thresholded": p.get("thresholded"),
                "survived_morph": p.get("survived_morph"),
                "candidate_id_if_any": p.get("candidate_id_if_any"),
                "kept_final": p.get("kept_final"),
                "edge_flag": p.get("edge_flag"),
            }
            for p in payload["top_peaks"]
        ],
        [
            "peak_rank", "x", "y", "value", "border_distance", "nearest_gt_id", "nearest_gt_distance",
            "thresholded", "survived_morph", "candidate_id_if_any", "kept_final", "edge_flag",
        ],
    )
    return report_path

