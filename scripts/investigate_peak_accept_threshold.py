"""
Dump per-candidate pre-threshold statistics for the focused peak-NMS path (three pairs).

Writes CSVs and summary JSON under ``outs/threshold_investigation/`` (repo-relative).

Run from repo root::

    python scripts/investigate_peak_accept_threshold.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from config import build_search_euclidean_artifact_residual_mad_config
from data import load_sample_pairs
from modules.postprocessing.peak_nms_postprocess import (
    _merge_peak_params,
    compute_dynamic_accept_threshold,
)
from pipeline import DefectDetectionPipeline
from utils.ground_truth_defects import pair_id_to_case_key


def _summarize(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    v = [float(x) for x in vals if math.isfinite(float(x))]
    if not v:
        return {"min": None, "max": None, "mean": None, "median": None, "stdev": None}
    return {
        "min": float(min(v)),
        "max": float(max(v)),
        "mean": float(statistics.mean(v)),
        "median": float(statistics.median(v)),
        "stdev": float(statistics.pstdev(v)) if len(v) > 1 else 0.0,
    }


def main() -> None:
    repo = _REPO
    out_dir = repo / "outs" / "threshold_investigation"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_search_euclidean_artifact_residual_mad_config()
    cfg.debug.enable_debug_visualization = False
    cfg.debug.save_debug_images = False

    p = DefectDetectionPipeline(cfg)
    samples = [
        s
        for s in load_sample_pairs(root_pattern=str(repo.parent / "*"), sort_results=True)
        if pair_id_to_case_key(s.pair_id) in ("case1", "case2", "case3")
    ]
    cached = {}
    for s in samples:
        a, _ = p.run_through_normalization(s, silent=True)
        cached[s.pair_id] = a

    summary: Dict[str, Any] = {"pairs": {}}

    for s in samples:
        case_key = pair_id_to_case_key(s.pair_id)
        pk = cfg.peak_nms_postprocess
        merged = _merge_peak_params(pk, case_key)
        base_thr = float(merged.get("accept_score_threshold", 0.0))
        use_stat = bool(merged.get("use_stat_derived_accept_threshold", False))

        r = p.run_from_normalized(s, cached[s.pair_id], silent=True)
        dm = r.artifacts.decision_metadata or {}
        pipe = dm.get("peak_nms_pipeline") or {}
        rows_in = list(dm.get("peak_nms_candidate_rows") or [])
        scores = [float(x["final_score"]) for x in rows_in]
        scores_sorted = sorted(scores, reverse=True)
        peakness = [float(x["peakness"]) for x in rows_in]

        # Effective threshold is whatever the postprocessor applied (stat-derived or config).
        eff_thr = float(pipe.get("accept_score_threshold", base_thr))

        ranked = sorted(range(len(rows_in)), key=lambda i: scores[i], reverse=True)
        rank_by_score = {int(i): r + 1 for r, i in enumerate(ranked)}

        table: List[Dict[str, Any]] = []
        for j, row in enumerate(rows_in):
            sc = float(row["final_score"])
            table.append({
                "pair_id": row.get("pair_id", s.pair_id),
                "case_key": case_key,
                "x": row.get("x"),
                "y": row.get("y"),
                "center_value": row.get("center_value"),
                "peakness": row.get("peakness"),
                "score": sc,
                "on_edge": row.get("on_edge"),
                "valid_support": row.get("valid_ok"),
                "rejected_pre_threshold_reason": "",
                "would_pass_current_threshold": bool(sc >= eff_thr),
                "is_kept_final": bool(row.get("kept")),
                "rank_by_score": rank_by_score.get(j),
                "nearest_gt_dist_px": row.get("nearest_gt_dist_px"),
            })

        safe = case_key
        csv_path = out_dir / f"candidates_pre_threshold_{safe}.csv"
        keys = list(table[0].keys()) if table else [
            "pair_id", "case_key", "x", "y", "center_value", "peakness", "score",
            "on_edge", "valid_support", "rejected_pre_threshold_reason",
            "would_pass_current_threshold", "is_kept_final", "rank_by_score",
            "nearest_gt_dist_px",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for row in table:
                w.writerow(row)

        n_above = sum(1 for sc in scores if sc >= eff_thr)
        dyn = compute_dynamic_accept_threshold(scores, base_fallback=base_thr) if use_stat else None
        summary["pairs"][case_key] = {
            "pair_id": s.pair_id,
            "pipeline_mode": pipe.get("mode"),
            "n_candidates_before_threshold": len(scores),
            "score_stats": _summarize(scores),
            "peakness_stats": _summarize(peakness),
            "top_10_scores": scores_sorted[:10],
            "top_10_peakness": sorted(peakness, reverse=True)[:10],
            "merged_config_accept_score_threshold_base": base_thr,
            "use_stat_derived_accept_threshold": use_stat,
            "effective_accept_threshold_used": eff_thr,
            "compute_dynamic_accept_threshold_if_enabled": dyn,
            "n_candidates_above_threshold": n_above,
            "n_after_nms": pipe.get("after_nms"),
            "final_kept_peaks": pipe.get("final_kept"),
            "final_detection_count": dm.get("final_num_contours"),
            "mask_true_pixels": int(np.count_nonzero(r.defect_mask)),
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir} (CSVs + summary.json)")


if __name__ == "__main__":
    main()
