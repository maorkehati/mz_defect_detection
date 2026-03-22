"""
Two-pass downstream screening sweep for the artifact_residual master path.

Pass 1: postprocess screening (fixed k_mad=4.0).
Pass 2: k_mad refinement on the best Pass 1 configs only.

Caches preprocessing / alignment / normalization once per sample, then sweeps
thresholding + contour_filter_postprocess only.

Outputs under ``outs/sweeps/downstream_artifact_residual_screening/``:
  - pass1_summary.csv, pass2_summary.csv, combined_summary.csv, top_configs_pass1.csv
  - pass1/run_XXX/, pass2/run_XXX/ — summary.txt + per-case candidate CSVs

Example:
  python scripts/run_downstream_sweep.py
  python scripts/run_downstream_sweep.py --pass2-top-n 12
"""

from __future__ import annotations

import argparse
import copy
import csv
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from config import PipelineConfig, build_search_euclidean_artifact_residual_mad_config
from data import load_sample_pairs
from pipeline import DefectDetectionPipeline
from utils.ground_truth_defects import get_ground_truth_points_for_pair, pair_id_to_case_key
from utils.gt_coverage import compute_gt_point_coverage_metrics

# --- Pass 1 (108 combos): k_mad fixed at 4.0 ---
PASS1_K_MAD = 4.0
PASS1_TOP_K = [3, 5, 8]
PASS1_MIN_AREA = [0, 8, 20]
# Three plausible modes for local defect emphasis (supported by contour_filter_postprocess).
PASS1_RANKING_MODES = [
    "artifact_consistent_local_contrast",
    "local_contrast_balanced",
    "intensity_size_balanced",
]
PASS1_MIN_CONTOUR_SCORE = [0.0, 4.0, 6.0, 8.0]
PASS1_GRID_N = len(PASS1_TOP_K) * len(PASS1_MIN_AREA) * len(PASS1_RANKING_MODES) * len(PASS1_MIN_CONTOUR_SCORE)

# --- Pass 2: threshold refinement on shortlisted Pass 1 configs ---
PASS2_K_MAD = [3.5, 4.0, 4.5, 5.0]

DEFAULT_ROOT_PATTERN = str(REPO_ROOT.parent / "*")
INSPECTED_PATTERN = "case*_inspected_image.tif"
REFERENCE_PATTERN = "case*_reference_image.tif"

OUT_ROOT_SCREENING = REPO_ROOT / "outs" / "sweeps" / "downstream_artifact_residual_screening"


def _ensure_parent(path: Path) -> None:
    """Ensure a file path's parent directory exists before writing."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _case_key(pair_id: str) -> str:
    return pair_id_to_case_key(pair_id) or pair_id


def _align_snippet(am: Dict[str, Any]) -> str:
    if not am:
        return "align(theta=na, tx=na, ty=na, overlap=na)"
    th = am.get("final_theta_deg", am.get("best_theta_deg", "na"))
    tx = am.get("final_tx", am.get("best_tx", "na"))
    ty = am.get("final_ty", am.get("best_ty", "na"))
    ov = am.get("overlap_fraction", am.get("overlap", "na"))
    return f"align(theta={th}, tx={tx}, ty={ty}, overlap={ov})"


def _compare_snippet(cm: Dict[str, Any], amap_stats: Dict[str, Any], nm: Dict[str, Any]) -> str:
    fe = cm.get("fraction_anomaly_touched_by_edge_mask", "na")
    gg = nm.get("gain", "na")
    p50 = amap_stats.get("p50", "na")
    p95 = amap_stats.get("p95", "na")
    p99 = amap_stats.get("p99", "na")
    return f"compare(frac_edge={fe}, norm_gain={gg}, anomaly_p50={p50}, anomaly_p95={p95}, anomaly_p99={p99})"


def _threshold_snippet(tm: Dict[str, Any], thr_stats: Dict[str, Any]) -> str:
    k = tm.get("k_mad", "na")
    t = tm.get("threshold_value", thr_stats.get("threshold", "na"))
    fg = thr_stats.get("positive_count", "na")
    ncc = tm.get("threshold_component_count", "na")
    return f"threshold(k_mad={k}, t={t}, fg_px={fg}, n_cc={ncc})"


def _post_snippet(dm: Dict[str, Any]) -> str:
    rc = dm.get("reject_counts") or {}
    raw = dm.get("num_contours_total", "na")
    geom_ok = dm.get("num_contours_after_geom_filters", "na")
    scored_n = dm.get("num_contours_scored", "na")
    after_score = dm.get("num_contours_after_score_threshold", "na")
    after_topk = dm.get("num_contours_after_topk", "na")
    border_n = int(rc.get("border_touch", 0) or 0)
    area_n = int(rc.get("min_area", 0) or 0) + int(rc.get("max_area", 0) or 0)
    score_drop = "na"
    try:
        if isinstance(scored_n, int) and isinstance(after_score, int):
            score_drop = int(scored_n) - int(after_score)
    except Exception:
        pass
    return (
        f"post(raw_cc={raw}, geom_ok={geom_ok}, scored={scored_n}, "
        f"area_filtered≈{area_n}, border_filtered={border_n}, score_filtered={score_drop}, "
        f"final_ranked={after_score}, kept={after_topk})"
    )


def _mask_from_top_score_candidates(
    shape: Tuple[int, int],
    candidate_rows: List[Dict[str, Any]],
    specs: List[Dict[str, Any]],
    top_k: int,
    min_score: float,
) -> np.ndarray:
    h, w = shape
    spec_by_id = {int(s["candidate_id"]): s["cnt"] for s in specs if s.get("candidate_id") is not None}
    eligible: List[Tuple[int, float, int]] = []
    for row in candidate_rows:
        cid = int(row.get("candidate_id", 0))
        sc = row.get("score")
        if sc is None:
            continue
        try:
            sf = float(sc)
        except (TypeError, ValueError):
            continue
        if sf < float(min_score):
            continue
        if cid not in spec_by_id:
            continue
        eligible.append((cid, sf, 0))
    eligible.sort(key=lambda t: -t[1])
    out = np.zeros((h, w), dtype=np.uint8)
    for cid, _s, _ in eligible[: max(0, int(top_k))]:
        cnt = spec_by_id.get(cid)
        if cnt is None:
            continue
        cv2.drawContours(out, [np.asarray(cnt)], -1, 255, thickness=cv2.FILLED)
    return out > 0


def _candidate_rows_for_csv(
    *,
    case_id: str,
    run_id: str,
    records: List[Dict[str, Any]],
    specs: List[Dict[str, Any]],
    gt_points: List[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    spec_by_id = {int(s["candidate_id"]): s["cnt"] for s in specs if s.get("candidate_id") is not None}
    rows: List[Dict[str, Any]] = []
    for r in records:
        cid = int(r.get("candidate_id", 0))
        gt_match = False
        gt_index: Optional[int] = None
        cnt = spec_by_id.get(cid)
        if cnt is not None and gt_points:
            for gi, (gx, gy) in enumerate(gt_points, start=1):
                try:
                    inside = cv2.pointPolygonTest(np.asarray(cnt), (float(gx), float(gy)), False)
                except Exception:
                    inside = -1.0
                if float(inside) >= 0.0:
                    gt_match = True
                    gt_index = gi
                    break
        rows.append(
            {
                "run_id": run_id,
                "case_id": case_id,
                "candidate_id": cid,
                "area": r.get("area"),
                "bbox_w": r.get("bbox_w"),
                "bbox_h": r.get("bbox_h"),
                "aspect_ratio": r.get("aspect_ratio"),
                "fill_ratio": r.get("fill_ratio"),
                "mean_anomaly": r.get("mean_anomaly"),
                "p95_anomaly": r.get("p95_anomaly"),
                "ring_mean": r.get("ring_mean"),
                "local_contrast": r.get("local_contrast"),
                "sign_consistency": r.get("sign_consistency"),
                "touches_border": r.get("border_touching"),
                "rank_score": r.get("score"),
                "rank_position": r.get("rank"),
                "kept_final": r.get("kept_final"),
                "gt_match": gt_match,
                "gt_index": gt_index,
            }
        )
    return rows


def _apply_params(cfg: PipelineConfig, row: Dict[str, Any]) -> None:
    cfg.thresholding.params["k_mad"] = float(row["k_mad"])
    p = cfg.contour_filter_postprocess.params
    p["top_k_keep"] = int(row["top_k_keep"])
    p["min_area"] = float(row["min_area"])
    p["ranking_mode"] = str(row["ranking_mode"])
    p["min_contour_score"] = float(row["min_contour_score"])
    p["contour_score_threshold_mode"] = "absolute"


def _build_pass1_grid() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tk, ma, rm, ms in product(
        PASS1_TOP_K,
        PASS1_MIN_AREA,
        PASS1_RANKING_MODES,
        PASS1_MIN_CONTOUR_SCORE,
    ):
        rows.append(
            {
                "k_mad": float(PASS1_K_MAD),
                "top_k_keep": int(tk),
                "min_area": float(ma),
                "ranking_mode": rm,
                "min_contour_score": float(ms),
            }
        )
    assert len(rows) == PASS1_GRID_N == 108
    return rows


def _rank_key_for_screening(row: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """
    Lower tuple is better (for sorting ascending).
    1) Maximize GT hits on case1+case2 → negate for sort
    2) Minimize case3 final detections (false positives)
    3) Minimize total final detections across all cases
    4) Prefer lower top_k_keep when tied
    """
    g1 = int(row.get("case1_gt_hit") or 0)
    g2 = int(row.get("case2_gt_hit") or 0)
    gt12 = g1 + g2
    c3 = int(row.get("case3_final") or 0)
    t1 = int(row.get("case1_final") or 0)
    t2 = int(row.get("case2_final") or 0)
    total = t1 + t2 + c3
    tk = int(row.get("top_k_keep") or 99)
    return (-gt12, c3, total, tk)


def _run_one_configuration(
    *,
    params: Dict[str, Any],
    run_label: str,
    run_dir: Path,
    base_cfg: PipelineConfig,
    cached: Dict[str, Any],
    samples: List[Any],
    gt_radius: float,
    pass_name: str,
) -> Dict[str, Any]:
    """Execute one full downstream run for all samples; write summary + CSVs. Returns summary row dict."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(base_cfg)
    _apply_params(cfg, params)
    pl = DefectDetectionPipeline(cfg)

    case_metrics: Dict[str, Dict[str, Any]] = {}
    lines_out: List[str] = []

    for sp in samples:
        ck = _case_key(sp.pair_id)
        up = cached[sp.pair_id]
        res = pl.run_from_normalized(sp, up, silent=True)
        art = res.artifacts
        dm = art.decision_metadata or {}
        am = art.alignment_metadata or {}
        cm = art.comparison_metadata or {}
        tm = art.thresholding_metadata or {}
        thr_st = art.thresholding_metadata or {}
        nm = art.normalization_metadata or {}
        amap_stats = {k: cm[k] for k in ("p50", "p95", "p99") if k in cm}
        if not amap_stats and art.anomaly_map is not None:
            a = np.asarray(art.anomaly_map, dtype=np.float32)
            amap_stats = {
                "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99)),
            }

        gt = get_ground_truth_points_for_pair(sp.pair_id)
        mask_final = res.defect_mask.astype(bool)
        m_final = compute_gt_point_coverage_metrics(mask_final, gt, radius_px=float(gt_radius))

        cand_records = list(dm.get("candidate_audit_records") or [])
        specs = list(dm.get("contour_audit_specs") or [])
        min_sc = float(params["min_contour_score"])
        mask_top3 = _mask_from_top_score_candidates(
            mask_final.shape,
            cand_records,
            specs,
            top_k=3,
            min_score=min_sc,
        )
        m_top3 = compute_gt_point_coverage_metrics(mask_top3, gt, radius_px=float(gt_radius))

        n_final = int(dm.get("final_num_contours", dm.get("num_contours_after_topk", 0)))
        if not gt:
            false_pos_v = int(n_final)
        else:
            false_pos_v = int(max(0, n_final - int(m_final.gt_covered_within_radius)))

        line = (
            f"CASE {ck}: "
            f"{_align_snippet(am)} "
            f"{_compare_snippet(cm, amap_stats, nm)} "
            f"{_threshold_snippet(tm, thr_st)} "
            f"{_post_snippet(dm)} "
            f"gt_hits(top3)={int(m_top3.gt_covered_within_radius)}/{len(gt)} "
            f"gt_hits(final)={int(m_final.gt_covered_within_radius)}/{len(gt)} "
            f"false_pos_final={false_pos_v}"
        )
        lines_out.append(line)

        kept = [r for r in cand_records if r.get("kept_final")]
        kept.sort(key=lambda x: -float(x.get("score") or 0.0))
        top_show = kept[:8]
        spec_by_id = {int(s["candidate_id"]): s["cnt"] for s in specs}
        parts = []
        for r in top_show:
            cid = int(r.get("candidate_id", 0))
            hit = False
            c0 = spec_by_id.get(cid)
            if c0 is not None and gt:
                for gx, gy in gt:
                    try:
                        if cv2.pointPolygonTest(np.asarray(c0), (float(gx), float(gy)), False) >= 0:
                            hit = True
                            break
                    except Exception:
                        pass
            parts.append(
                f"(id={cid}, area={r.get('area')}, score={r.get('score')}, "
                f"ring={r.get('ring_mean')}, sign={r.get('sign_consistency')}, hit_gt={hit})"
            )
        lines_out.append(f"  top_final=[{', '.join(parts)}]")

        rows_csv = _candidate_rows_for_csv(
            case_id=ck,
            run_id=run_label,
            records=cand_records,
            specs=specs,
            gt_points=list(gt),
        )
        csv_path = run_dir / f"{ck}_candidates.csv"
        if rows_csv:
            _ensure_parent(csv_path)
            fieldnames = list(rows_csv[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in rows_csv:
                    w.writerow(row)

        case_metrics[ck] = {
            "n_final": n_final,
            "gt_hit": int(m_final.gt_covered_within_radius),
            "gt_total": len(gt),
        }

    c1 = case_metrics.get("case1", {})
    c2 = case_metrics.get("case2", {})
    c3 = case_metrics.get("case3", {})
    have3 = {"case1", "case2", "case3"} <= set(case_metrics.keys())
    ok = bool(
        have3
        and c1.get("gt_hit") == 3
        and c1.get("gt_total") == 3
        and c2.get("gt_hit") == 3
        and c2.get("gt_total") == 3
        and c3.get("n_final", -1) == 0
    )

    gt12 = int(c1.get("gt_hit", 0)) + int(c2.get("gt_hit", 0))
    c3fp = int(c3.get("n_final", 0))
    total_d = sum(int(case_metrics[k].get("n_final", 0)) for k in case_metrics)

    txt: List[str] = []
    if not have3:
        txt.append(
            "WARN: need case1, case2, case3 in loaded samples for SUCCESS; got: "
            + ",".join(sorted(case_metrics.keys()))
        )
        txt.append("")
    txt.extend(
        [
            f"{pass_name} RUN SUMMARY",
            f"params: top_k={params['top_k_keep']}, min_area={params['min_area']}, "
            f"k_mad={params['k_mad']}, ranking={params['ranking_mode']}, min_score={params['min_contour_score']}",
            f"case1: final={c1.get('n_final', 'na')}, gt_hit={c1.get('gt_hit', 'na')}/{c1.get('gt_total', 'na')}",
            f"case2: final={c2.get('n_final', 'na')}, gt_hit={c2.get('gt_hit', 'na')}/{c2.get('gt_total', 'na')}",
            f"case3: final={c3.get('n_final', 'na')}, gt_hit=0/0, false_pos={c3.get('n_final', 'na')}",
            f"SUCCESS={'YES' if ok else 'NO'}",
            "",
            *lines_out,
        ]
    )
    summary_path = run_dir / "summary.txt"
    _ensure_parent(summary_path)
    summary_path.write_text("\n".join(txt), encoding="utf-8")

    summary_row = {
        "pass": pass_name,
        "run_label": run_label,
        "success": ok,
        "top_k_keep": params["top_k_keep"],
        "min_area": params["min_area"],
        "k_mad": params["k_mad"],
        "ranking_mode": params["ranking_mode"],
        "min_contour_score": params["min_contour_score"],
        "case1_final": c1.get("n_final"),
        "case1_gt_hit": c1.get("gt_hit"),
        "case2_final": c2.get("n_final"),
        "case2_gt_hit": c2.get("gt_hit"),
        "case3_final": c3.get("n_final"),
        "case3_false_pos": c3.get("n_final"),
        "gt_hits_case1_case2": gt12,
        "total_final_detections": total_d,
        "gt_recall_12": (gt12 / 6.0) if (c1.get("gt_total") == 3 and c2.get("gt_total") == 3) else None,
    }
    return summary_row


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-pass downstream screening sweep (artifact_residual path).")
    ap.add_argument("--root-pattern", type=str, default=DEFAULT_ROOT_PATTERN)
    ap.add_argument("--inspected-pattern", type=str, default=INSPECTED_PATTERN)
    ap.add_argument("--reference-pattern", type=str, default=REFERENCE_PATTERN)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--gt-radius", type=float, default=5.0)
    ap.add_argument(
        "--pass2-top-n",
        type=int,
        default=12,
        help="Number of best Pass 1 configs to carry into Pass 2 (default 12, use 10–15).",
    )
    ap.add_argument("--out", type=str, default=str(OUT_ROOT_SCREENING))
    args = ap.parse_args()

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    pass1_root = out_root / "pass1"
    pass2_root = out_root / "pass2"
    pass1_root.mkdir(parents=True, exist_ok=True)
    pass2_root.mkdir(parents=True, exist_ok=True)

    samples = load_sample_pairs(
        root_pattern=args.root_pattern,
        inspected_pattern=args.inspected_pattern,
        reference_pattern=args.reference_pattern,
        recursive=args.recursive,
        sort_results=True,
    )
    samples = [s for s in samples if re.search(r"case\s*[123]", s.pair_id, re.I)] or samples
    if not samples:
        print("No samples loaded; check --root-pattern.", file=sys.stderr)
        sys.exit(1)

    base_cfg = build_search_euclidean_artifact_residual_mad_config()
    pipeline = DefectDetectionPipeline(base_cfg)
    cached: Dict[str, Any] = {}
    for sp in samples:
        up, _ = pipeline.run_through_normalization(sp, silent=True)
        cached[sp.pair_id] = up

    # ----- Pass 1 -----
    pass1_grid = _build_pass1_grid()
    pass1_rows: List[Dict[str, Any]] = []
    pass1_success: List[str] = []

    pass1_enumerated = list(enumerate(pass1_grid))
    for idx, p in tqdm(
        pass1_enumerated,
        desc="Pass 1 (postprocess)",
        unit="config",
        total=len(pass1_enumerated),
    ):
        run_label = f"p1_{idx:03d}"
        run_dir = pass1_root / f"run_{idx:03d}"
        row = _run_one_configuration(
            params=p,
            run_label=run_label,
            run_dir=run_dir,
            base_cfg=base_cfg,
            cached=cached,
            samples=samples,
            gt_radius=args.gt_radius,
            pass_name="PASS1",
        )
        row["pass1_index"] = idx
        pass1_rows.append(row)
        if row.get("success"):
            pass1_success.append(run_label)

    # Rank Pass 1
    ranked_p1 = sorted(pass1_rows, key=_rank_key_for_screening)
    for rank, r in enumerate(ranked_p1, start=1):
        r["screening_rank"] = rank

    top_n = max(1, min(int(args.pass2_top_n), len(ranked_p1)))
    top_configs = ranked_p1[:top_n]

    # Write pass1 outputs
    if pass1_rows:
        p1_csv = out_root / "pass1_summary.csv"
        _ensure_parent(p1_csv)
        with p1_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pass1_rows[0].keys()))
            w.writeheader()
            for row in pass1_rows:
                w.writerow(row)
    top_p1 = out_root / "top_configs_pass1.csv"
    _ensure_parent(top_p1)
    with top_p1.open("w", newline="", encoding="utf-8") as f:
        if top_configs:
            w = csv.DictWriter(f, fieldnames=list(top_configs[0].keys()))
            w.writeheader()
            for row in top_configs:
                w.writerow(row)

    tqdm.write("")
    tqdm.write(f"Pass 1: total combinations tested: {len(pass1_grid)} (expected {PASS1_GRID_N}).")
    tqdm.write("Pass 1: top 10 configs by screening rank (best first):")
    for r in ranked_p1[:10]:
        tqdm.write(
            f"  rank={r.get('screening_rank')}  label={r.get('run_label')}  "
            f"gt12={r.get('gt_hits_case1_case2')}/6  case3_final={r.get('case3_final')}  "
            f"total_dets={r.get('total_final_detections')}  "
            f"top_k={r.get('top_k_keep')}  min_area={r.get('min_area')}  "
            f"ranking={r.get('ranking_mode')}  min_score={r.get('min_contour_score')}"
        )

    # ----- Pass 2 -----
    pass2_rows: List[Dict[str, Any]] = []
    pass2_list = [(parent, km) for parent in top_configs for km in PASS2_K_MAD]
    pass2_enumerated = list(enumerate(pass2_list))
    pass2_success: List[str] = []

    for j, (parent, k_mad) in tqdm(
        pass2_enumerated,
        desc="Pass 2 (k_mad)",
        unit="config",
        total=len(pass2_enumerated),
    ):
        p2_params = {
            "k_mad": float(k_mad),
            "top_k_keep": int(parent["top_k_keep"]),
            "min_area": float(parent["min_area"]),
            "ranking_mode": str(parent["ranking_mode"]),
            "min_contour_score": float(parent["min_contour_score"]),
        }
        run_label = f"p2_{j:03d}"
        run_dir = pass2_root / f"run_{j:03d}"
        row = _run_one_configuration(
            params=p2_params,
            run_label=run_label,
            run_dir=run_dir,
            base_cfg=base_cfg,
            cached=cached,
            samples=samples,
            gt_radius=args.gt_radius,
            pass_name="PASS2",
        )
        row["pass2_index"] = j
        row["pass1_parent_rank"] = parent.get("screening_rank")
        row["pass1_parent_label"] = parent.get("run_label")
        row["pass1_index"] = parent.get("pass1_index")
        pass2_rows.append(row)
        if row.get("success"):
            pass2_success.append(run_label)

    combined = pass1_rows + pass2_rows
    if combined:
        field_order: List[str] = []
        seen_k = set()
        for row in combined:
            for k in row:
                if k not in seen_k:
                    seen_k.add(k)
                    field_order.append(k)
        comb_csv = out_root / "combined_summary.csv"
        _ensure_parent(comb_csv)
        with comb_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=field_order, extrasaction="ignore")
            w.writeheader()
            for row in combined:
                w.writerow(row)
    if pass2_rows:
        p2_csv = out_root / "pass2_summary.csv"
        _ensure_parent(p2_csv)
        with p2_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pass2_rows[0].keys()))
            w.writeheader()
            for row in pass2_rows:
                w.writerow(row)

    all_success = pass1_success + pass2_success
    ranked_all = sorted(combined, key=_rank_key_for_screening) if combined else []

    tqdm.write("")
    tqdm.write(f"Pass 2: additional combinations tested: {len(pass2_list)}")
    tqdm.write(f"Pass 2: shortlisted from Pass 1 (top {top_n} by screening rank).")
    tqdm.write("Best overall (Pass 1 + Pass 2 combined ranking, best first):")
    for r in ranked_all[:15]:
        tqdm.write(
            f"  {r.get('pass')}  {r.get('run_label')}  "
            f"gt12={r.get('gt_hits_case1_case2')}/6  case3_final={r.get('case3_final')}  "
            f"k_mad={r.get('k_mad')}  SUCCESS={r.get('success')}"
        )
    tqdm.write(
        f"Any SUCCESS=YES: {'YES' if any(r.get('success') for r in combined) else 'NO'} "
        f"(count={sum(1 for r in combined if r.get('success'))})"
    )


if __name__ == "__main__":
    main()
