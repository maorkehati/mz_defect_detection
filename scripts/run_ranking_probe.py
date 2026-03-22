"""
Compact diagnostic runs for contour ranking modes (local contrast emphasis).

Runs 4–5 hand-picked downstream configurations on cached upstream (alignment +
normalization), prints strong per-case logs, and writes CSV + summary under
``outs/diagnostics/ranking_probe/``.

Example:
  python scripts/run_ranking_probe.py
  python scripts/run_ranking_probe.py --root-pattern \"<path-to-parent>/*\"
"""

from __future__ import annotations

import argparse
import copy
import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import PipelineConfig, build_search_euclidean_artifact_residual_mad_config
from data import load_sample_pairs
from pipeline import DefectDetectionPipeline
from utils.ground_truth_defects import pair_id_to_case_key

DEFAULT_ROOT_PATTERN = str(REPO_ROOT.parent / "*")
INSPECTED_PATTERN = "case*_inspected_image.tif"
REFERENCE_PATTERN = "case*_reference_image.tif"

OUT_ROOT = REPO_ROOT / "outs" / "diagnostics" / "ranking_probe"

# Configs A–D (required) + E baseline (optional sweep winner style).
PROBE_RUNS: List[Dict[str, Any]] = [
    {
        "run_name": "A_contrast_area_log_k3",
        "ranking_mode": "contrast_area_log",
        "top_k_keep": 3,
        "min_area": 0,
        "min_contour_score": 0.0,
        "k_mad": 4.0,
    },
    {
        "run_name": "B_contrast_area_log_k5",
        "ranking_mode": "contrast_area_log",
        "top_k_keep": 5,
        "min_area": 0,
        "min_contour_score": 0.0,
        "k_mad": 4.0,
    },
    {
        "run_name": "C_contrast_ratio_area_sqrt_k3",
        "ranking_mode": "contrast_ratio_area_sqrt",
        "top_k_keep": 3,
        "min_area": 0,
        "min_contour_score": 0.0,
        "k_mad": 4.0,
    },
    {
        "run_name": "D_contrast_ratio_area_sqrt_k5",
        "ranking_mode": "contrast_ratio_area_sqrt",
        "top_k_keep": 5,
        "min_area": 0,
        "min_contour_score": 0.0,
        "k_mad": 4.0,
    },
    {
        "run_name": "E_baseline_artifact_consistent_local_contrast_k3",
        "ranking_mode": "artifact_consistent_local_contrast",
        "top_k_keep": 3,
        "min_area": 0,
        "min_contour_score": 0.0,
        "k_mad": 4.0,
    },
]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _case_key(pair_id: str) -> str:
    return pair_id_to_case_key(pair_id) or pair_id


def _threshold_snippet(tm: Dict[str, Any], thr_stats: Dict[str, Any]) -> str:
    k = tm.get("k_mad", "na")
    t = tm.get("threshold_value", thr_stats.get("threshold", "na"))
    fg = thr_stats.get("positive_count", "na")
    ncc = tm.get("threshold_component_count", "na")
    return f"threshold(n_cc={ncc}, k_mad={k}, t={t}, fg_px={fg})"


def _apply_params(cfg: PipelineConfig, row: Dict[str, Any]) -> None:
    cfg.thresholding.params["k_mad"] = float(row["k_mad"])
    p = cfg.contour_filter_postprocess.params
    p["top_k_keep"] = int(row["top_k_keep"])
    p["min_area"] = float(row["min_area"])
    p["ranking_mode"] = str(row["ranking_mode"])
    p["min_contour_score"] = float(row["min_contour_score"])
    p["contour_score_threshold_mode"] = "absolute"
    if row.get("min_sign_dominance") is not None:
        p["min_sign_dominance"] = float(row["min_sign_dominance"])
    if row.get("min_asymmetry") is not None:
        p["min_asymmetry"] = float(row["min_asymmetry"])


def _format_top_candidate_line(rc: Dict[str, Any]) -> str:
    base = (
        f"(rank={rc['rank_position']}, id={rc['candidate_id']}, area={rc['area']:.2f}, "
        f"mean={rc['mean_anomaly']:.4f}, p95={rc['p95_anomaly']:.4f}, ring={rc['ring_mean']:.4f}, "
        f"local_contrast={rc['local_contrast']:.4f}, contrast_ratio={rc['contrast_ratio']:.4f}, "
        f"score={rc['rank_score']:.6f}, gt_match={rc['gt_match']}"
    )
    if "mean_z_pos" in rc and rc.get("mean_z_pos") is not None:
        try:
            mzp = float(rc["mean_z_pos"])
            mzn = float(rc["mean_z_neg"])
            sd = float(rc.get("sign_dominance", float("nan")))
            zds = str(rc.get("z_dominant_sign", "?"))
            base += f", mean_z_pos={mzp:.4f}, mean_z_neg={mzn:.4f}, sign_dominance={sd:.4f}, z_dom={zds}"
        except (TypeError, ValueError):
            pass
    return base + ")"


def _run_probe(
    *,
    params: Dict[str, Any],
    run_dir: Path,
    base_cfg: PipelineConfig,
    cached: Dict[str, Any],
    samples: List[Any],
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(base_cfg)
    _apply_params(cfg, params)
    pl = DefectDetectionPipeline(cfg)

    summary_lines: List[str] = []
    per_case: Dict[str, Dict[str, Any]] = {}

    for sp in samples:
        ck = _case_key(sp.pair_id)
        up = cached[sp.pair_id]
        res = pl.run_from_normalized(sp, up, silent=True)
        art = res.artifacts
        dm = art.decision_metadata or {}
        cm = art.comparison_metadata or {}
        tm = art.thresholding_metadata or {}
        thr_st = art.thresholding_metadata or {}

        raw_n = int(dm.get("num_contours_total", 0))
        ranked_n = int(dm.get("num_contours_scored", 0))
        kept = int(dm.get("num_contours_after_topk", dm.get("final_num_contours", 0)))
        rm = str(dm.get("ranking_mode", params["ranking_mode"]))

        diag = dm.get("gt_ranking_diagnostics") or {}
        gt_pos = list(diag.get("gt_rank_positions") or [])
        h3 = int(diag.get("gt_hits_in_top3", 0))
        h5 = int(diag.get("gt_hits_in_top5", 0))
        h10 = int(diag.get("gt_hits_in_top10", 0))
        gt_tot = int(diag.get("gt_total", 0))

        ranked_full = list(dm.get("ranked_candidates_full") or [])
        top10 = ranked_full[:10]

        print(f"\nCASE {ck}:")
        print(f"  {_threshold_snippet(tm, thr_st)}")
        sig = cm.get("sigma_noise", "na")
        fn = cm.get("final_normalization", "")
        if sig != "na" or fn:
            print(f"  compare(sigma_noise={sig}, final_norm={fn})")
        if cm.get("comparator_mode") == "local_patch_ncc":
            print(
                "  ncc("
                f"patch={cm.get('patch_size')}, sigma_mad={cm.get('sigma_mad')}, "
                f"sim_mean={cm.get('similarity_mean')}, sim_p95={cm.get('similarity_p95')}, "
                f"anomaly_mean={cm.get('anomaly_mean_before_scale')}, "
                f"anomaly_p95={cm.get('anomaly_p95_before_scale')}, "
                f"z_mean={cm.get('anomaly_mean_after_z')}, z_p95={cm.get('anomaly_p95_after_z')})"
            )
        else:
            wsig = cm.get("whitening_sigma", None)
            if wsig is not None:
                rsb = cm.get("residual_std_before", "na")
                rsa = cm.get("residual_std_after", "na")
                print(f"  whitening(sigma={wsig}, residual_std_before={rsb}, residual_std_after={rsa})")
        print(
            f"  post(raw_candidates={raw_n}, ranked_candidates={ranked_n}, kept={kept}, ranking_mode={rm})"
        )
        if gt_tot > 0:
            print(
                f"  gt_hits_top3={h3}/{gt_tot}, gt_hits_top5={h5}/{gt_tot}, gt_hits_top10={h10}/{gt_tot}"
            )
            print(f"  gt_rank_positions={gt_pos}")
        else:
            print("  gt_rank_positions=[]  (no GT for this case)")
        print("  top_candidates=[")
        for rc in top10:
            print(f"    {_format_top_candidate_line(rc)},")
        print("  ]")

        # CSV: one row per ranked candidate
        csv_path = run_dir / f"{ck}_ranked_candidates.csv"
        if ranked_full:
            _ensure_parent(csv_path)
            fieldnames = [
                "case_id",
                "candidate_id",
                "rank_position",
                "area",
                "bbox_w",
                "bbox_h",
                "aspect_ratio",
                "fill_ratio",
                "mean_anomaly",
                "p95_anomaly",
                "ring_mean",
                "local_contrast",
                "contrast_ratio",
                "sign_consistency",
                "mean_z_pos",
                "mean_z_neg",
                "sign_dominance",
                "z_dominant_sign",
                "mean_z_reference",
                "mean_z_inspected",
                "asymmetry",
                "touches_border",
                "rank_score",
                "kept_final",
                "gt_match",
                "gt_index",
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for rc in ranked_full:
                    row = {k: rc.get(k) for k in fieldnames if k != "case_id"}
                    row["case_id"] = ck
                    w.writerow(row)

        top_scores = [float(x.get("rank_score", 0.0)) for x in ranked_full[:3]]
        per_case[ck] = {
            "final": kept,
            "gt_hits_top3": h3,
            "gt_hits_top5": h5,
            "gt_hits_top10": h10,
            "gt_rank_positions": gt_pos,
            "gt_total": gt_tot,
            "top3_scores": top_scores,
            "ranked_full": ranked_full,
        }

        summary_lines.append("")
        summary_lines.append(f"CASE {ck}:")
        summary_lines.append(f"  {_threshold_snippet(tm, thr_st)}")
        sig = cm.get("sigma_noise", "na")
        fn = cm.get("final_normalization", "")
        if sig != "na" or fn:
            summary_lines.append(f"  compare(sigma_noise={sig}, final_norm={fn})")
        if cm.get("comparator_mode") == "local_patch_ncc":
            summary_lines.append(
                "  ncc("
                f"patch={cm.get('patch_size')}, sigma_mad={cm.get('sigma_mad')}, "
                f"sim_mean={cm.get('similarity_mean')}, sim_p95={cm.get('similarity_p95')}, "
                f"anomaly_mean={cm.get('anomaly_mean_before_scale')}, "
                f"anomaly_p95={cm.get('anomaly_p95_before_scale')}, "
                f"z_mean={cm.get('anomaly_mean_after_z')}, z_p95={cm.get('anomaly_p95_after_z')})"
            )
        else:
            wsig = cm.get("whitening_sigma", None)
            if wsig is not None:
                rsb = cm.get("residual_std_before", "na")
                rsa = cm.get("residual_std_after", "na")
                summary_lines.append(
                    f"  whitening(sigma={wsig}, residual_std_before={rsb}, residual_std_after={rsa})"
                )
        summary_lines.append(
            f"  post(raw_candidates={raw_n}, ranked_candidates={ranked_n}, kept={kept}, ranking_mode={rm})"
        )
        if gt_tot > 0:
            summary_lines.append(
                f"  gt_hits_top3={h3}/{gt_tot}, gt_hits_top5={h5}/{gt_tot}, gt_hits_top10={h10}/{gt_tot}"
            )
            summary_lines.append(f"  gt_rank_positions={gt_pos}")
        else:
            summary_lines.append("  gt_rank_positions=[]  (no GT for this case)")
        summary_lines.append("  top_candidates=[")
        for rc in top10:
            summary_lines.append(f"    {_format_top_candidate_line(rc)},")
        summary_lines.append("  ]")

    c1 = per_case.get("case1", {})
    c2 = per_case.get("case2", {})
    c3 = per_case.get("case3", {})

    print("\nRUN SUMMARY")
    msd = params.get("min_sign_dominance", None)
    msd_s = f", min_sign_dominance={msd}" if msd is not None else ""
    print(
        f"params: ranking_mode={params['ranking_mode']}, top_k_keep={params['top_k_keep']}, "
        f"min_area={params['min_area']}, min_contour_score={params['min_contour_score']}, "
        f"k_mad={params['k_mad']}{msd_s}"
    )
    print(
        f"case1: final={c1.get('final', 'na')}, "
        f"gt_hit_top3={c1.get('gt_hits_top3', 0)}/3, gt_hit_top5={c1.get('gt_hits_top5', 0)}/3, "
        f"gt_rank_positions={c1.get('gt_rank_positions', [])}"
    )
    print(
        f"case2: final={c2.get('final', 'na')}, "
        f"gt_hit_top3={c2.get('gt_hits_top3', 0)}/3, gt_hit_top5={c2.get('gt_hits_top5', 0)}/3, "
        f"gt_rank_positions={c2.get('gt_rank_positions', [])}"
    )
    print(
        f"case3: final={c3.get('final', 'na')}, false_pos_final={c3.get('final', 'na')}, "
        f"top_scores={c3.get('top3_scores', [])}"
    )

    msd2 = params.get("min_sign_dominance", None)
    msd_line = f", min_sign_dominance={msd2}" if msd2 is not None else ""
    mas2 = params.get("min_asymmetry", None)
    mas_line = f", min_asymmetry={mas2}" if mas2 is not None else ""
    summary_lines = [
        "RUN SUMMARY",
        f"params: ranking_mode={params['ranking_mode']}, top_k_keep={params['top_k_keep']}, "
        f"min_area={params['min_area']}, min_contour_score={params['min_contour_score']}, "
        f"k_mad={params['k_mad']}{msd_line}{mas_line}",
        f"case1: final={c1.get('final', 'na')}, "
        f"gt_hit_top3={c1.get('gt_hits_top3', 0)}/3, gt_hit_top5={c1.get('gt_hits_top5', 0)}/3, "
        f"gt_rank_positions={c1.get('gt_rank_positions', [])}",
        f"case2: final={c2.get('final', 'na')}, "
        f"gt_hit_top3={c2.get('gt_hits_top3', 0)}/3, gt_hit_top5={c2.get('gt_hits_top5', 0)}/3, "
        f"gt_rank_positions={c2.get('gt_rank_positions', [])}",
        f"case3: final={c3.get('final', 'na')}, false_pos_final={c3.get('final', 'na')}, "
        f"top_scores={c3.get('top3_scores', [])}",
        "",
        *summary_lines,
    ]
    summary_path = run_dir / "summary.txt"
    _ensure_parent(summary_path)
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "run_name": params["run_name"],
        "ranking_mode": params["ranking_mode"],
        "params": dict(params),
        "per_case": per_case,
    }


def _mean_rank(pos: List[int]) -> float:
    if not pos:
        return float("inf")
    return float(sum(pos)) / float(len(pos))


def _sum_top3_scores(scores: List[float]) -> float:
    if not scores:
        return float("inf")
    return float(sum(scores))


def main() -> None:
    ap = argparse.ArgumentParser(description="Ranking probe (contrast-focused modes + diagnostics).")
    ap.add_argument("--root-pattern", type=str, default=DEFAULT_ROOT_PATTERN)
    ap.add_argument("--inspected-pattern", type=str, default=INSPECTED_PATTERN)
    ap.add_argument("--reference-pattern", type=str, default=REFERENCE_PATTERN)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip config E (artifact_consistent_local_contrast) to keep 4 runs.",
    )
    ap.add_argument("--out", type=str, default=str(OUT_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

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

    results: List[Dict[str, Any]] = []
    runs = PROBE_RUNS if not args.skip_baseline else [r for r in PROBE_RUNS if not r["run_name"].startswith("E_")]

    print(f"Loaded {len(samples)} samples. Output root: {out_root}")
    for params in runs:
        run_name = str(params["run_name"])
        run_dir = out_root / run_name
        print("\n" + "=" * 72)
        print(f"RUN {run_name}")
        print("=" * 72)
        row = _run_probe(
            params=params,
            run_dir=run_dir,
            base_cfg=base_cfg,
            cached=cached,
            samples=samples,
        )
        if row:
            results.append(row)

    # ----- Cross-run comparison -----
    print("\n" + "=" * 72)
    print("COMPARISON ACROSS RUNS")
    print("=" * 72)

    def _pick_best_gt_case(case: str) -> Optional[str]:
        best: Optional[Tuple[float, str]] = None
        for r in results:
            pc = r["per_case"].get(case, {})
            pos = list(pc.get("gt_rank_positions") or [])
            m = _mean_rank(pos)
            name = str(r["run_name"])
            if best is None or m < best[0]:
                best = (m, name)
        return best[1] if best else None

    def _pick_lowest_case3_top3() -> Optional[str]:
        best: Optional[Tuple[float, str]] = None
        for r in results:
            pc = r["per_case"].get("case3", {})
            s = _sum_top3_scores(list(pc.get("top3_scores") or []))
            name = str(r["run_name"])
            if best is None or s < best[0]:
                best = (s, name)
        return best[1] if best else None

    b1 = _pick_best_gt_case("case1")
    b2 = _pick_best_gt_case("case2")
    b3 = _pick_lowest_case3_top3()
    baseline = next((r for r in results if str(r["run_name"]).startswith("E_")), None)

    print(f"Best mean GT rank position (case1): {b1}")
    print(f"Best mean GT rank position (case2): {b2}")
    print(f"Lowest sum of top-3 rank_scores (case3 clutter strength): {b3}")
    if baseline:
        print(
            "Baseline E (artifact_consistent_local_contrast k=3) present — compare A–D vs E "
            "using printed RUN SUMMARY and CSVs."
        )
    else:
        print("No baseline E in this session (--skip-baseline or omitted).")

    for r in results:
        print(
            f"  {r['run_name']}: "
            f"case1_pos={r['per_case'].get('case1', {}).get('gt_rank_positions', [])} "
            f"case2_pos={r['per_case'].get('case2', {}).get('gt_rank_positions', [])} "
            f"case3_top3_scores={r['per_case'].get('case3', {}).get('top3_scores', [])}"
        )


if __name__ == "__main__":
    main()
