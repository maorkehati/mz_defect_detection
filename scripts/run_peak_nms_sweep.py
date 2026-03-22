"""
Sweep peak_nms_postprocess parameters with **no** per-case overrides.

Pipeline: search_euclidean → linear_gain_offset → artifact_residual → mad_threshold → peak_nms_postprocess.

Caches preprocessing/alignment/normalization once per sample, then runs comparator→threshold→postprocess
for each grid point (``run_from_normalized``, ``save_outputs=False``).

Grid (tuned for global threshold vs. case3 suppression; see ACCEPT_THRESHOLDS below).

Usage:
  python scripts/run_peak_nms_sweep.py

Outputs:
  outs/sweeps/peak_nms_global/summary.csv
"""

from __future__ import annotations

import copy
import csv
import math
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import build_search_euclidean_artifact_residual_mad_config
from data import load_sample_pairs
from dd_types import SamplePair
from pipeline import DefectDetectionPipeline
from utils.ground_truth_defects import get_ground_truth_points_for_pair
from utils.gt_coverage import compute_gt_point_coverage_metrics

ROOT_PATTERN = str(REPO_ROOT.parent / "*")
INSPECTED_PATTERN = "case*_inspected_image.tif"
REFERENCE_PATTERN = "case*_reference_image.tif"

OUT_DIR = REPO_ROOT / "outs" / "sweeps" / "peak_nms_global"
SUMMARY_CSV = OUT_DIR / "summary.csv"

# Second sweep: meaningful regime vs. prior case3 override (~2.45)
ACCEPT_THRESHOLDS = [
    0.40,
    0.60,
    0.80,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.20,
    2.35,
    2.45,
    2.55,
    2.70,
]
MIN_PEAKNESS = [0.008, 0.015, 0.020]
NMS_RADII = [8.0, 10.0]
GT_RADIUS_PX = 5.0
GT_MATCH_FOR_SCORE_PX = 5.0


def _pair_to_case_key(pair_id: str) -> str:
    m = re.search(r"case\s*(\d+)", pair_id, re.I)
    return f"case{int(m.group(1))}" if m else pair_id


def _filter_three_cases(samples: List[SamplePair]) -> List[SamplePair]:
    want = {"case1", "case2", "case3"}
    out: List[SamplePair] = []
    for s in samples:
        ck = _pair_to_case_key(s.pair_id)
        if ck in want:
            out.append(s)
    out.sort(key=lambda sp: _pair_to_case_key(sp.pair_id))
    return out


def _base_peak_nms_config() -> Any:
    cfg = build_search_euclidean_artifact_residual_mad_config()
    cfg.peak_nms_postprocess.case_overrides = {}
    cfg.debug.enable_debug_visualization = False
    cfg.debug.save_debug_images = False
    cfg.output.return_artifacts = True
    return cfg


def _apply_grid(cfg: Any, *, acc: float, min_pk: float, nms_r: float) -> None:
    p = cfg.peak_nms_postprocess
    p.accept_score_threshold = float(acc)
    p.min_peakness = float(min_pk)
    p.post_accept_nms_radius_px = float(nms_r)


def _detection_count(defect_mask: np.ndarray, decision_metadata: Dict[str, Any]) -> int:
    m = np.asarray(defect_mask).astype(bool)
    if not np.any(m):
        return 0
    dm = decision_metadata or {}
    v = dm.get("final_num_contours", dm.get("num_centers_drawn", dm.get("num_peaks_selected", 0)))
    try:
        return int(v)
    except (TypeError, ValueError):
        return 1


def _max_pre_accept_score(decision_metadata: Dict[str, Any]) -> Optional[float]:
    """
    Max ``final_score`` among peak candidates that reached the scored stage (edge/peakness OK),
    before ``accept_score_threshold`` — from ``peak_nms_candidate_rows``.
    """
    rows = decision_metadata.get("peak_nms_candidate_rows") or []
    scores: List[float] = []
    for r in rows:
        fs = r.get("final_score")
        if fs is None:
            continue
        try:
            scores.append(float(fs))
        except (TypeError, ValueError):
            pass
    return max(scores) if scores else None


def _min_gt_matched_final_score(
    decision_metadata: Dict[str, Any],
    gt_points_xy: List[Tuple[int, int]],
) -> Optional[float]:
    """
    Among **final** kept peaks (``components``), for each GT point take the nearest peak center
    within ``GT_MATCH_FOR_SCORE_PX``; return the minimum ``peak_score`` among those matches.
    None if no GT point gets a matching final peak.
    """
    if not gt_points_xy:
        return None
    comps = decision_metadata.get("components") or []
    if not comps:
        return None
    matched: List[float] = []
    for gx, gy in gt_points_xy:
        best_d = float("inf")
        best_sc: Optional[float] = None
        for c in comps:
            cx = float(c.get("centroid_x", 0.0))
            cy = float(c.get("centroid_y", 0.0))
            d = math.hypot(cx - float(gx), cy - float(gy))
            if d <= GT_MATCH_FOR_SCORE_PX and d < best_d:
                best_d = d
                try:
                    best_sc = float(c.get("peak_score", 0.0))
                except (TypeError, ValueError):
                    best_sc = None
        if best_sc is not None:
            matched.append(best_sc)
    return min(matched) if matched else None


def _csv_val(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float):
        return v
    return v


def main() -> None:
    samples = load_sample_pairs(
        root_pattern=ROOT_PATTERN,
        inspected_pattern=INSPECTED_PATTERN,
        reference_pattern=REFERENCE_PATTERN,
        recursive=False,
        sort_results=True,
    )
    samples = _filter_three_cases(samples)
    if len(samples) != 3:
        print(
            f"[peak_nms_sweep] Expected 3 samples (case1–3), got {len(samples)}. "
            f"Check ROOT_PATTERN={ROOT_PATTERN!r}",
            flush=True,
        )
        sys.exit(1)

    base = _base_peak_nms_config()
    probe = DefectDetectionPipeline(base)
    cached: Dict[str, Any] = {}
    for s in samples:
        art, _ = probe.run_through_normalization(s, silent=True)
        cached[s.pair_id] = art

    grid = list(product(ACCEPT_THRESHOLDS, MIN_PEAKNESS, NMS_RADII))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    perfect_lines: List[str] = []

    for acc, min_pk, nms_r in grid:
        cfg = copy.deepcopy(base)
        _apply_grid(cfg, acc=acc, min_pk=min_pk, nms_r=nms_r)
        pipe = DefectDetectionPipeline(cfg)

        case1_hits = 0
        case2_hits = 0
        case3_det = 0
        case3_max_pre: Optional[float] = None
        case1_min_gt: Optional[float] = None
        case2_min_gt: Optional[float] = None

        for s in samples:
            ck = _pair_to_case_key(s.pair_id)
            result = pipe.run_from_normalized(s, cached[s.pair_id], silent=True)
            mask = result.defect_mask
            dm = result.artifacts.decision_metadata or {}
            pts = get_ground_truth_points_for_pair(s.pair_id)

            if ck == "case3":
                case3_det = _detection_count(mask, dm)
                case3_max_pre = _max_pre_accept_score(dm)
                continue

            if pts:
                met = compute_gt_point_coverage_metrics(mask, pts, radius_px=GT_RADIUS_PX)
                hits = int(met.gt_covered_within_radius)
            else:
                hits = 0
            mg = _min_gt_matched_final_score(dm, list(pts))
            if ck == "case1":
                case1_hits = hits
                case1_min_gt = mg
            elif ck == "case2":
                case2_hits = hits
                case2_min_gt = mg

        perfect = 1 if (case1_hits == 3 and case2_hits == 3 and case3_det == 0) else 0
        row = {
            "accept_score_threshold": acc,
            "min_peakness": min_pk,
            "post_accept_nms_radius_px": nms_r,
            "case1_hits": case1_hits,
            "case2_hits": case2_hits,
            "case3_detections": case3_det,
            "case3_max_pre_accept_score": _csv_val(case3_max_pre),
            "case1_min_gt_matched_score": _csv_val(case1_min_gt),
            "case2_min_gt_matched_score": _csv_val(case2_min_gt),
            "perfect": perfect,
        }
        rows.append(row)
        if perfect:
            perfect_lines.append(
                f"  PERFECT  accept={acc}  min_peakness={min_pk}  nms_r={nms_r}  "
                f"hits c1={case1_hits} c2={case2_hits}  case3_det={case3_det}  "
                f"case3_max_pre={case3_max_pre}  c1_min_gt={case1_min_gt}  c2_min_gt={case2_min_gt}"
            )

    fieldnames = [
        "accept_score_threshold",
        "min_peakness",
        "post_accept_nms_radius_px",
        "case1_hits",
        "case2_hits",
        "case3_detections",
        "case3_max_pre_accept_score",
        "case1_min_gt_matched_score",
        "case2_min_gt_matched_score",
        "perfect",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[peak_nms_sweep] wrote {SUMMARY_CSV}  rows={len(rows)}  grid={len(ACCEPT_THRESHOLDS)}x{len(MIN_PEAKNESS)}x{len(NMS_RADII)}", flush=True)
    perf = [r for r in rows if r["perfect"] == 1]
    if perf:
        print(f"[peak_nms_sweep] perfect configs ({len(perf)}):", flush=True)
        for line in perfect_lines:
            print(line, flush=True)
    else:
        print("[peak_nms_sweep] no perfect configs (3/3 GT @5px + case3 0 detections)", flush=True)


if __name__ == "__main__":
    main()
