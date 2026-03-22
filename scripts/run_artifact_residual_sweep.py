"""
Localized sweep for the **artifact_residual** path only.

Caches preprocessing → search_euclidean alignment → linear_gain_offset normalization once per sample,
then sweeps comparator (top-hat + edge), MAD k, and minimal contour postprocess.

Does not mix in gradient_difference or other legacy grids.

Outputs (under ``outs/sweeps/<sweep_name>/``):
  - summary.csv, summary.txt
  - figures/ optional PNGs for selected configs (compact + artifact_residual debug)

Default: subsample the full Cartesian grid to ~100 configs (seeded) so runtime stays modest.
Use ``--max-configs -1`` for the full grid (~1500+ combos; can be slow).

Usage:
  python scripts/run_artifact_residual_sweep.py --sweep-name art_res_local
  python scripts/run_artifact_residual_sweep.py --max-configs -1 --figures all   # full grid + all figures
"""

from __future__ import annotations

import argparse
import copy
import csv
import random
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

try:
    from tqdm import tqdm

    _TQDM_AVAILABLE = True
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]
    _TQDM_AVAILABLE = False


def _tqdm(iterable, **kwargs):
    if _TQDM_AVAILABLE and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def _tqdm_write(msg: str) -> None:
    if _TQDM_AVAILABLE and tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg, flush=True)


from config import PipelineConfig, build_search_euclidean_artifact_residual_mad_config
from data import load_sample_pairs
from dd_types import PipelineArtifacts, SamplePair
from pipeline import DefectDetectionPipeline
from utils.ground_truth_defects import get_ground_truth_points_for_pair, load_defect_locations
from utils.gt_coverage import compute_gt_point_coverage_metrics
from visualization.debug import save_compact_pipeline_figure

# --- Sweep dimensions (localized; full product is large — use --max-configs by default) ---
TOP_HAT_KERNEL_SIZES = [5, 7, 9, 11]
EDGE_MODES = ["off", "hard"]
EDGE_PERCENTILES = [85.0, 90.0, 93.0]
EDGE_DILATE_KERNELS = [3, 5]
K_MAD_VALUES = [3.0, 3.5, 4.0, 4.5]
MIN_AREAS = [4.0, 6.0, 8.0, 10.0]
TOP_K_KEEP = [4, 6]

# Fixed downstream (not swept); matches focused artifact_residual defaults.
FIXED_RANKING_MODE = "intensity_size_balanced"
FIXED_REJECT_SIGN = False


def _pair_id_to_case_key(pair_id: str) -> str:
    m = re.search(r"case\s*(\d+)", pair_id, re.I)
    return f"case{int(m.group(1))}" if m else pair_id


def _full_grid_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    idx = 0
    for tup in product(
        TOP_HAT_KERNEL_SIZES,
        EDGE_MODES,
        EDGE_PERCENTILES,
        EDGE_DILATE_KERNELS,
        K_MAD_VALUES,
        MIN_AREAS,
        TOP_K_KEEP,
    ):
        th, em, ep, edk, km, ma, tk = tup
        rows.append(
            {
                "config_index": idx,
                "top_hat_kernel_size": int(th),
                "edge_mode": str(em),
                "edge_percentile": float(ep),
                "edge_dilate_kernel": int(edk),
                "k_mad": float(km),
                "min_area": float(ma),
                "top_k_keep": int(tk),
            }
        )
        idx += 1
    return rows


def _subsample_rows(rows: List[Dict[str, Any]], max_configs: int, seed: int) -> List[Dict[str, Any]]:
    if max_configs < 0 or len(rows) <= max_configs:
        return [{**dict(r), "config_index": i} for i, r in enumerate(rows)]
    rng = random.Random(seed)
    picked = rng.sample(list(rows), max_configs)
    picked.sort(key=lambda x: x["config_index"])
    return [{**dict(r), "config_index": i} for i, r in enumerate(picked)]


def _apply_row(base: PipelineConfig, row: Dict[str, Any]) -> PipelineConfig:
    cfg = copy.deepcopy(base)
    cfg.choices.comparison = "artifact_residual"
    cfg.artifact_residual.params.update(
        {
            "top_hat_kernel_size": int(row["top_hat_kernel_size"]),
            "top_hat_iterations": 1,
            "pre_blur_sigma": 1.0,
            "combine_mode": "max",
            "norm_percentile_low": 1.0,
            "norm_percentile_high": 99.0,
            "use_valid_mask": True,
            "edge_mode": str(row["edge_mode"]),
            "edge_percentile": float(row["edge_percentile"]),
            "edge_dilate_kernel": int(row["edge_dilate_kernel"]),
            "edge_dilate_iterations": 1,
            "edge_weight_on_edges": 0.25,
            "edge_gradient_ksize": 3,
            "edge_source": "inspected",
            "min_valid_fraction": 0.0,
            # Omit intermediates during sweep to limit memory; figures re-run comparator with debug on-demand.
            "debug_save_intermediates": False,
        }
    )
    cfg.thresholding.params["k_mad"] = float(row["k_mad"])

    p = cfg.contour_filter_postprocess.params
    p["min_area"] = float(row["min_area"])
    p["top_k_keep"] = int(row["top_k_keep"])
    p["ranking_mode"] = FIXED_RANKING_MODE
    p["reject_on_low_sign_consistency"] = FIXED_REJECT_SIGN
    p["min_sign_consistency"] = None
    return cfg


def _config_id(i: int) -> str:
    return f"c{i:05d}"


def _summary_to_param_row(sr: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild sweep row for _apply_row from a summary CSV row."""
    return {
        "config_index": int(sr["config_index"]),
        "top_hat_kernel_size": int(sr["top_hat_kernel_size"]),
        "edge_mode": str(sr["edge_mode"]),
        "edge_percentile": float(sr["edge_percentile"]),
        "edge_dilate_kernel": int(sr["edge_dilate_kernel"]),
        "k_mad": float(sr["k_mad"]),
        "min_area": float(sr["min_area"]),
        "top_k_keep": int(sr["top_k_keep"]),
    }


def _composite_score(
    gt_hits_within_r_c1: int,
    gt_hits_within_r_c2: int,
    n_gt_c1: int,
    n_gt_c2: int,
    n_c1: int,
    n_c2: int,
    n_c3: int,
    *,
    weight_clean_case3: float = 5.0,
    weight_count_mismatch: float = 0.2,
) -> float:
    """
    Higher is better: maximize **within-radius** GT point hits vs listed GT counts on case1+case2,
    penalize detections on case3, lightly penalize deviation from 3 detections on case1/2.
    """
    r1 = gt_hits_within_r_c1 / max(1, n_gt_c1)
    r2 = gt_hits_within_r_c2 / max(1, n_gt_c2)
    cov_term = 100.0 * (r1 + r2)
    pen_c3 = weight_clean_case3 * float(n_c3) + (2.0 if n_c3 > 0 else 0.0)
    mismatch = weight_count_mismatch * (abs(n_c1 - 3) + abs(n_c2 - 3))
    return float(cov_term - pen_c3 - mismatch)


def run_artifact_residual_sweep(
    *,
    sweep_name: str,
    samples: List[SamplePair],
    max_configs: int,
    subsample_seed: int,
    figures_mode: str,
    top_figures: int,
    gt_radius: float,
) -> None:
    out_root = REPO_ROOT / "outs" / "sweeps" / sweep_name
    out_root.mkdir(parents=True, exist_ok=True)
    fig_root = out_root / "figures"
    fig_root.mkdir(parents=True, exist_ok=True)

    # Preload GT counts per case (for coverage denominator)
    _ = load_defect_locations()
    gt_by_case: Dict[str, List[Tuple[int, int]]] = {}
    for s in samples:
        ck = _pair_id_to_case_key(s.pair_id)
        gt_by_case[ck] = get_ground_truth_points_for_pair(s.pair_id)

    base_cfg = build_search_euclidean_artifact_residual_mad_config()
    base_cfg.output.return_artifacts = True

    cache_probe = DefectDetectionPipeline(base_cfg)
    cached_by_pair: Dict[str, PipelineArtifacts] = {}
    for sample in _tqdm(samples, desc="Cache upstream", unit="sample"):
        art, _ = cache_probe.run_through_normalization(sample, silent=True)
        cached_by_pair[sample.pair_id] = art

    all_rows = _full_grid_rows()
    total_grid = len(all_rows)
    rows = _subsample_rows(all_rows, max_configs, subsample_seed)
    n_conf = len(rows)

    _tqdm_write(
        f"[artifact_residual_sweep] sweep_name={sweep_name} grid_size={total_grid} "
        f"configs_run={n_conf} (max_configs={max_configs}) samples={len(samples)} tqdm={'on' if _TQDM_AVAILABLE else 'off'}"
    )

    results_unsorted: List[Dict[str, Any]] = []

    for row in _tqdm(rows, desc="Sweep configs", unit="cfg"):
        cfg = _apply_row(base_cfg, row)
        cfg.output.return_artifacts = True
        pipe = DefectDetectionPipeline(cfg)
        cid = _config_id(row["config_index"])

        counts: Dict[str, int] = {}
        gt_metrics: Dict[str, Any] = {}

        for sample in samples:
            ck = _pair_id_to_case_key(sample.pair_id)
            cached = cached_by_pair[sample.pair_id]
            result = pipe.run_from_normalized(sample, cached, silent=True)
            dm = result.artifacts.decision_metadata or {}
            n_det = int(
                dm.get(
                    "final_num_contours",
                    dm.get("num_contours_after_topk", dm.get("num_kept_contours", 0)),
                )
            )
            counts[ck] = n_det
            pts = gt_by_case.get(ck, [])
            gt_metrics[ck] = compute_gt_point_coverage_metrics(
                result.defect_mask,
                pts,
                radius_px=float(gt_radius),
            )

        c1 = int(counts.get("case1", 0))
        c2 = int(counts.get("case2", 0))
        c3 = int(counts.get("case3", 0))
        m1 = gt_metrics.get("case1")
        m2 = gt_metrics.get("case2")
        ng1 = int(m1.gt_total) if m1 else 0
        ng2 = int(m2.gt_total) if m2 else 0
        g1e = int(m1.gt_covered_exact) if m1 else 0
        g2e = int(m2.gt_covered_exact) if m2 else 0
        g1r = int(m1.gt_covered_within_radius) if m1 else 0
        g2r = int(m2.gt_covered_within_radius) if m2 else 0
        f1e = float(m1.coverage_fraction_exact) if m1 else 0.0
        f2e = float(m2.coverage_fraction_exact) if m2 else 0.0
        f1r = float(m1.coverage_fraction_within_radius) if m1 else 0.0
        f2r = float(m2.coverage_fraction_within_radius) if m2 else 0.0
        gt12_tot = ng1 + ng2
        gt12_rad = g1r + g2r
        gt_cov_frac_case12 = float(gt12_rad) / float(max(1, gt12_tot))

        score = _composite_score(g1r, g2r, ng1, ng2, c1, c2, c3)

        results_unsorted.append(
            {
                "config_id": cid,
                "config_index": row["config_index"],
                "top_hat_kernel_size": row["top_hat_kernel_size"],
                "edge_mode": row["edge_mode"],
                "edge_percentile": row["edge_percentile"],
                "edge_dilate_kernel": row["edge_dilate_kernel"],
                "k_mad": row["k_mad"],
                "min_area": row["min_area"],
                "top_k_keep": row["top_k_keep"],
                "case1_detections": c1,
                "case2_detections": c2,
                "case3_detections": c3,
                "case1_gt_total": ng1,
                "case2_gt_total": ng2,
                "case1_gt_covered_exact": g1e,
                "case2_gt_covered_exact": g2e,
                "case1_gt_covered_within_radius": g1r,
                "case2_gt_covered_within_radius": g2r,
                "case1_coverage_fraction_exact": round(f1e, 6),
                "case2_coverage_fraction_exact": round(f2e, 6),
                "case1_coverage_fraction_within_radius": round(f1r, 6),
                "case2_coverage_fraction_within_radius": round(f2r, 6),
                "gt_total_case12": gt12_tot,
                "gt_covered_within_radius_case12": gt12_rad,
                "gt_coverage_fraction_within_radius_case12": round(gt_cov_frac_case12, 6),
                "gt_radius": int(gt_radius),
                "composite_score": round(score, 4),
            }
        )

    summary_rows = sorted(results_unsorted, key=lambda r: (-float(r["composite_score"]), r["config_id"]))

    fieldnames = [
        "config_id",
        "config_index",
        "top_hat_kernel_size",
        "edge_mode",
        "edge_percentile",
        "edge_dilate_kernel",
        "k_mad",
        "min_area",
        "top_k_keep",
        "case1_detections",
        "case2_detections",
        "case3_detections",
        "case1_gt_total",
        "case2_gt_total",
        "case1_gt_covered_exact",
        "case2_gt_covered_exact",
        "case1_gt_covered_within_radius",
        "case2_gt_covered_within_radius",
        "case1_coverage_fraction_exact",
        "case2_coverage_fraction_exact",
        "case1_coverage_fraction_within_radius",
        "case2_coverage_fraction_within_radius",
        "gt_total_case12",
        "gt_covered_within_radius_case12",
        "gt_coverage_fraction_within_radius_case12",
        "gt_radius",
        "composite_score",
    ]
    csv_path = out_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r[k] for k in fieldnames})

    tk = max(1, min(top_figures, len(summary_rows)))
    top_block = summary_rows[:tk]
    worst_block = list(reversed(summary_rows[-tk:])) if len(summary_rows) > 1 else []

    lines = [
        f"sweep_name: {sweep_name}",
        f"artifact_residual_localized: true",
        f"full_grid_size: {total_grid}",
        f"configs_evaluated: {n_conf}",
        f"subsample_seed: {subsample_seed}",
        f"max_configs: {max_configs}",
        f"gt_proximity_radius_px: {gt_radius}",
        "",
        "Sweep dimensions:",
        f"  top_hat_kernel_size: {TOP_HAT_KERNEL_SIZES}",
        f"  edge_mode: {EDGE_MODES}",
        f"  edge_percentile: {EDGE_PERCENTILES}",
        f"  edge_dilate_kernel: {EDGE_DILATE_KERNELS}",
        f"  k_mad: {K_MAD_VALUES}",
        f"  min_area: {MIN_AREAS}",
        f"  top_k_keep: {TOP_K_KEEP}",
        "",
        "Fixed: preprocessing=gaussian_preprocess, alignment=search_euclidean, normalization=linear_gain_offset",
        f"Fixed post: ranking_mode={FIXED_RANKING_MODE}, reject_on_low_sign_consistency={FIXED_REJECT_SIGN}",
        "",
        "Composite score (higher better): 100*(hits_r_case1/n_gt1 + hits_r_case2/n_gt2) "
        "- penalize case3 dets - 0.2*|n-3| on case1/2  (hits_r = GT covered within gt_radius)",
        "",
        "Top configurations by composite_score:",
    ]
    for r in summary_rows[:15]:
        lines.append(
            f"  {r['config_id']} score={r['composite_score']} "
            f"dets=({r['case1_detections']},{r['case2_detections']},{r['case3_detections']}) "
            f"gt_within_r=({r['case1_gt_covered_within_radius']}/{r['case1_gt_total']},"
            f"{r['case2_gt_covered_within_radius']}/{r['case2_gt_total']}) "
            f"frac12={r['gt_coverage_fraction_within_radius_case12']} "
            f"k={r['top_hat_kernel_size']} edge={r['edge_mode']}/p{r['edge_percentile']}/dil{r['edge_dilate_kernel']} "
            f"k_mad={r['k_mad']} min_a={r['min_area']} topk={r['top_k_keep']}"
        )

    txt_path = out_root / "summary.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Optional figures (compact + artifact_residual_debug from save_compact_pipeline_figure)
    if figures_mode == "none":
        _tqdm_write("[artifact_residual_sweep] figures_mode=none — skipping PNGs")
    else:
        draw_list: List[Dict[str, Any]] = []
        if figures_mode == "all":
            draw_list = list(results_unsorted)
        else:
            seen: set[str] = set()
            for r in top_block + worst_block:
                cid = str(r["config_id"])
                if cid not in seen:
                    seen.add(cid)
                    draw_list.append(r)

        for sr in _tqdm(draw_list, desc="Figures", unit="cfg"):
            param = _summary_to_param_row(sr)
            cfg = _apply_row(base_cfg, param)
            cfg.output.return_artifacts = True
            pipe = DefectDetectionPipeline(cfg)
            cid = str(sr["config_id"])
            sub = fig_root / cid
            sub.mkdir(parents=True, exist_ok=True)
            for sample in samples:
                cached = cached_by_pair[sample.pair_id]
                result = pipe.run_from_normalized(sample, cached, silent=True)
                art = result.artifacts
                out_png = sub / f"{sample.pair_id}_pipeline.png"
                try:
                    save_compact_pipeline_figure(
                        pair_id=sample.pair_id,
                        artifacts=art,
                        comparator=pipe.comparator,
                        comparator_cfg=pipe._resolve_comparison_config(),
                        output_path=out_png,
                    )
                except Exception as exc:
                    _tqdm_write(f"[WARN] figure failed config={cid} pair={sample.pair_id} reason={exc}")

    print(f"[artifact_residual_sweep] wrote {csv_path}")
    print(f"[artifact_residual_sweep] wrote {txt_path}")
    if figures_mode != "none":
        print(f"[artifact_residual_sweep] figures under {fig_root}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Localized artifact_residual sweep (cached upstream).",
    )
    ap.add_argument("--sweep-name", type=str, default="artifact_residual_local", help="Subfolder under outs/sweeps/")
    ap.add_argument(
        "--max-configs",
        type=int,
        default=100,
        help="Max configs to evaluate (subsampled from full grid with --seed). Use -1 for full grid.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Subsample RNG seed when truncating grid.")
    ap.add_argument(
        "--figures",
        choices=("all", "top", "none"),
        default="top",
        help="all: every evaluated config; top: best+worst composite configs; none: skip PNGs",
    )
    ap.add_argument("--top-figures", type=int, default=12, help="How many best (and worst) configs for figures when --figures top")
    ap.add_argument(
        "--gt-radius",
        type=float,
        default=5.0,
        help="Euclidean distance (px): GT counts covered if nearest defect pixel is within this radius",
    )
    ap.add_argument(
        "--root-pattern",
        type=str,
        default=str(REPO_ROOT.parent / "*"),
        help="Glob for dataset root",
    )
    args = ap.parse_args()
    max_c = args.max_configs

    samples = load_sample_pairs(
        root_pattern=args.root_pattern,
        inspected_pattern="case*_inspected_image.tif",
        reference_pattern="case*_reference_image.tif",
        recursive=False,
        sort_results=True,
    )
    if not samples:
        print("[artifact_residual_sweep] No samples found; check --root-pattern and filenames.")
        sys.exit(1)

    run_artifact_residual_sweep(
        sweep_name=args.sweep_name,
        samples=samples,
        max_configs=max_c,
        subsample_seed=args.seed,
        figures_mode=args.figures,
        top_figures=args.top_figures,
        gt_radius=args.gt_radius,
    )


if __name__ == "__main__":
    main()
