"""
Focused parameter sweep: cache preprocessing/alignment/normalization once per sample,
then sweep comparator → thresholding → postprocessing only.

**Single full run (artifact_residual, all samples):** use ``scripts/run_pipeline.py`` (default config).

For a **localized artifact_residual-only** sweep (recommended for tuning), use
``scripts/run_artifact_residual_sweep.py`` instead — this file mixes gradient + artifact grids.

Outputs under outs/sweeps/<sweep_name>/:
  - summary.csv, summary.txt (includes GT point coverage vs final mask; see --gt-radius)
  - <config_id>/<pair_id>_pipeline.png for each (config, sample)

Does not modify the normal single-config pipeline entry points.

Default --max-configs caps runtime (full Cartesian product is large); uses seeded shuffle
to sample a diverse subset when capping.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


class _TqdmFallbackBar:
    """Iterable wrapper when tqdm is not installed (no progress display)."""

    def __init__(self, iterable):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def set_postfix(self, ordered_dict=None, **kwargs) -> None:
        pass

    def set_postfix_str(self, s: str) -> None:
        pass


def _tqdm(iterable, **kwargs):
    if _TQDM_AVAILABLE and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return _TqdmFallbackBar(iterable)


def _tqdm_write(msg: str) -> None:
    if _TQDM_AVAILABLE and tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg, flush=True)


def _rank_short(name: str, max_len: int = 22) -> str:
    s = str(name)
    return s if len(s) <= max_len else s[: max_len - 2] + ".."


from config import PipelineConfig, build_search_euclidean_gradient_difference_mad_config
from data import load_sample_pairs
from dd_types import PipelineArtifacts, SamplePair
from pipeline import DefectDetectionPipeline
from utils.ground_truth_defects import get_ground_truth_points_for_pair, load_defect_locations
from utils.gt_coverage import compute_gt_point_coverage_metrics
from visualization.debug import save_compact_pipeline_figure

ROOT_PATTERN = str(REPO_ROOT.parent / "*")
INSPECTED_PATTERN = "case*_inspected_image.tif"
REFERENCE_PATTERN = "case*_reference_image.tif"

TOPHAT_KERNEL_SIZES = [7, 9, 11, 13]
PRE_BLUR_SIGMAS = [0.5, 1.0]
EDGE_WEIGHTS = [0.20, 0.35]
EDGE_PERCENTILES = [80.0, 85.0]
K_MAD_VALUES = [3.5, 4.0, 4.5, 5.0, 5.5]
TOP_K_KEEP = [3, 4, 5]
MIN_CONTOUR_SCORE = [3.0, 4.0, 4.5, 5.0]
MIN_AREA = [15.0, 30.0]
RING_RADIUS = [5, 7, 9]
MIN_SIGN_CONSISTENCY: List[Optional[float]] = [None, 0.6, 0.7]
RANKING_MODES = [
    "artifact_consistent_local_contrast",
    "local_contrast_balanced",
    "intensity_size_balanced",
]


def _pair_id_to_case_key(pair_id: str) -> str:
    m = re.search(r"case\s*(\d+)", pair_id, re.I)
    return f"case{int(m.group(1))}" if m else pair_id


def _build_sweep_rows(max_configs: Optional[int]) -> List[Dict[str, Any]]:
    art_combos = list(product(TOPHAT_KERNEL_SIZES, PRE_BLUR_SIGMAS))
    grad_combos = list(product(EDGE_WEIGHTS, EDGE_PERCENTILES))

    comp_variants: List[Tuple[str, Dict[str, Any]]] = []
    for th, pb in art_combos:
        comp_variants.append(
            (
                "artifact_residual",
                {"tophat_kernel_size": int(th), "pre_blur_sigma": float(pb)},
            )
        )
    for ew, ep in grad_combos:
        comp_variants.append(
            (
                "gradient_difference",
                {"edge_weight_on_edges": float(ew), "edge_percentile": float(ep)},
            )
        )

    post_combos = list(
        product(
            K_MAD_VALUES,
            TOP_K_KEEP,
            MIN_CONTOUR_SCORE,
            MIN_AREA,
            RING_RADIUS,
            MIN_SIGN_CONSISTENCY,
            RANKING_MODES,
        )
    )

    full: List[Dict[str, Any]] = []
    for comp_choice, comp_extra in comp_variants:
        for (
            k_mad,
            top_k,
            min_score,
            min_area,
            ring_r,
            min_sign,
            ranking_mode,
        ) in post_combos:
            full.append(
                {
                    "comparator_choice": comp_choice,
                    "comparator_params": dict(comp_extra),
                    "k_mad": float(k_mad),
                    "top_k_keep": int(top_k),
                    "min_contour_score": float(min_score),
                    "min_area": float(min_area),
                    "ring_radius_px": int(ring_r),
                    "min_sign_consistency": min_sign,
                    "ranking_mode": ranking_mode,
                }
            )

    if max_configs is not None and len(full) > max_configs:
        rng = random.Random(42)
        rng.shuffle(full)
        full = full[:max_configs]

    for i, row in enumerate(full):
        row["config_index"] = i

    return full


def _apply_sweep_to_config(base: PipelineConfig, row: Dict[str, Any]) -> PipelineConfig:
    cfg = copy.deepcopy(base)
    cfg.choices.comparison = str(row["comparator_choice"])
    cfg.thresholding.params["k_mad"] = float(row["k_mad"])
    p = cfg.contour_filter_postprocess.params
    p["top_k_keep"] = int(row["top_k_keep"])
    p["min_contour_score"] = float(row["min_contour_score"])
    p["min_area"] = float(row["min_area"])
    p["ring_radius_px"] = int(row["ring_radius_px"])
    p["min_sign_consistency"] = row["min_sign_consistency"]
    p["ranking_mode"] = str(row["ranking_mode"])

    cp = row["comparator_params"]
    if row["comparator_choice"] == "artifact_residual":
        cfg.artifact_residual.params.update(
            {
                "pre_blur_sigma": float(cp["pre_blur_sigma"]),
                "tophat_kernel_size": int(cp["tophat_kernel_size"]),
                "combine_mode": "max",
                "norm_percentile_low": 1.0,
                "norm_percentile_high": 99.0,
                "use_valid_mask": True,
            }
        )
    else:
        cfg.gradient_difference.params["edge_suppression_enabled"] = True
        cfg.gradient_difference.params["edge_weight_on_edges"] = float(cp["edge_weight_on_edges"])
        cfg.gradient_difference.params["edge_percentile"] = float(cp["edge_percentile"])
    return cfg


def _config_id(row: Dict[str, Any]) -> str:
    return f"c{int(row['config_index']):05d}"


def _thresh_positive_fraction(binary_mask_raw: np.ndarray | None) -> float:
    if binary_mask_raw is None:
        return 0.0
    m = np.asarray(binary_mask_raw).astype(bool)
    return float(np.count_nonzero(m) / max(1, m.size))


def run_sweep(
    *,
    sweep_name: str,
    max_configs: Optional[int],
    samples: List[SamplePair],
    gt_radius: float = 5.0,
) -> None:
    out_root = REPO_ROOT / "outs" / "sweeps" / sweep_name
    out_root.mkdir(parents=True, exist_ok=True)

    _ = load_defect_locations()
    gt_by_case: Dict[str, List[Tuple[int, int]]] = {}
    for s in samples:
        ck = _pair_id_to_case_key(s.pair_id)
        gt_by_case[ck] = get_ground_truth_points_for_pair(s.pair_id)

    base_cfg = build_search_euclidean_gradient_difference_mad_config()
    base_cfg.output.return_artifacts = True

    cache_probe = DefectDetectionPipeline(base_cfg)
    cached_by_pair: Dict[str, PipelineArtifacts] = {}
    for sample in samples:
        art, _ = cache_probe.run_through_normalization(sample, silent=True)
        cached_by_pair[sample.pair_id] = art

    rows_spec = _build_sweep_rows(max_configs)
    total = len(rows_spec)
    _tqdm_write(
        f"[sweep] sweep_name={sweep_name} total_configurations={total} samples={len(samples)} "
        f"(tqdm={'on' if _TQDM_AVAILABLE else 'off'})"
    )

    summary_rows: List[Dict[str, Any]] = []

    pbar_outer = _tqdm(
        rows_spec,
        desc="Sweep configs",
        unit="cfg",
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for row in pbar_outer:
        cfg = _apply_sweep_to_config(base_cfg, row)
        cfg.output.return_artifacts = True
        pipe = DefectDetectionPipeline(cfg)
        cid = _config_id(row)

        pbar_outer.set_postfix(
            k=float(row["k_mad"]),
            comp=str(row["comparator_choice"])[:14],
            rank=_rank_short(str(row["ranking_mode"])),
            refresh=False,
        )

        counts: Dict[str, int] = {"case1": 0, "case2": 0, "case3": 0}
        areas: Dict[str, float] = {"case1": 0.0, "case2": 0.0, "case3": 0.0}
        pos_frac: Dict[str, float] = {"case1": 0.0, "case2": 0.0, "case3": 0.0}
        gt_metrics: Dict[str, Any] = {}

        inner_iter = (
            _tqdm(
                samples,
                desc=cid,
                leave=False,
                unit="case",
                dynamic_ncols=True,
                mininterval=0.2,
            )
            if _TQDM_AVAILABLE
            else samples
        )
        for sample in inner_iter:
            ck = _pair_id_to_case_key(sample.pair_id)
            cached = cached_by_pair[sample.pair_id]
            result = pipe.run_from_normalized(sample, cached, silent=True)
            art = result.artifacts
            dm = art.decision_metadata or {}
            n_det = int(
                dm.get(
                    "final_num_contours",
                    dm.get("num_contours_after_topk", dm.get("num_kept_contours", 0)),
                )
            )
            t_area = float(dm.get("total_kept_area", 0.0))
            pf = _thresh_positive_fraction(art.binary_mask_raw)

            counts[ck] = n_det
            areas[ck] = t_area
            pos_frac[ck] = pf
            pts = gt_by_case.get(ck, [])
            gt_metrics[ck] = compute_gt_point_coverage_metrics(
                result.defect_mask,
                pts,
                radius_px=float(gt_radius),
            )

            fig_dir = out_root / cid
            fig_dir.mkdir(parents=True, exist_ok=True)
            out_png = fig_dir / f"{sample.pair_id}_pipeline.png"
            try:
                save_compact_pipeline_figure(
                    pair_id=sample.pair_id,
                    artifacts=art,
                    comparator=pipe.comparator,
                    comparator_cfg=pipe._resolve_comparison_config(),
                    output_path=out_png,
                )
            except Exception as exc:
                _tqdm_write(f"[sweep] WARN figure failed config={cid} pair={sample.pair_id} reason={exc}")

        c1 = int(counts.get("case1", 0))
        c2 = int(counts.get("case2", 0))
        c3 = int(counts.get("case3", 0))
        objective = abs(c1 - 3) + abs(c2 - 3) + abs(c3 - 0)
        exact = c1 == 3 and c2 == 3 and c3 == 0

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

        summary_rows.append(
            {
                "config_id": cid,
                "config_index": row["config_index"],
                "comparator_choice": row["comparator_choice"],
                "comparator_params_json": json.dumps(row["comparator_params"], sort_keys=True),
                "threshold_k": row["k_mad"],
                "ranking_mode": row["ranking_mode"],
                "top_k_keep": row["top_k_keep"],
                "min_contour_score": row["min_contour_score"],
                "min_area": row["min_area"],
                "ring_radius_px": row["ring_radius_px"],
                "min_sign_consistency": row["min_sign_consistency"],
                "case1_count": c1,
                "case2_count": c2,
                "case3_count": c3,
                "case1_total_area": float(areas.get("case1", 0.0)),
                "case2_total_area": float(areas.get("case2", 0.0)),
                "case3_total_area": float(areas.get("case3", 0.0)),
                "case1_thresh_pos_frac": float(pos_frac.get("case1", 0.0)),
                "case2_thresh_pos_frac": float(pos_frac.get("case2", 0.0)),
                "case3_thresh_pos_frac": float(pos_frac.get("case3", 0.0)),
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
                "gt_radius": float(gt_radius),
                "objective": objective,
                "exact_match": exact,
            }
        )

    summary_rows.sort(key=lambda r: (-bool(r["exact_match"]), r["objective"], r["config_id"]))

    csv_path = out_root / "summary.csv"
    fieldnames = [
        "config_id",
        "comparator_choice",
        "comparator_params_json",
        "threshold_k",
        "ranking_mode",
        "top_k_keep",
        "min_contour_score",
        "min_area",
        "ring_radius_px",
        "min_sign_consistency",
        "case1_count",
        "case2_count",
        "case3_count",
        "case1_total_area",
        "case2_total_area",
        "case3_total_area",
        "case1_thresh_pos_frac",
        "case2_thresh_pos_frac",
        "case3_thresh_pos_frac",
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
        "objective",
        "exact_match",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            row_out = {k: r.get(k) for k in fieldnames}
            if row_out.get("min_sign_consistency") is None:
                row_out["min_sign_consistency"] = ""
            w.writerow(row_out)

    best_obj = min(summary_rows, key=lambda r: r["objective"])["objective"] if summary_rows else None
    any_exact = any(r["exact_match"] for r in summary_rows)
    top10 = summary_rows[:10]

    txt_path = out_root / "summary.txt"
    lines = [
        f"sweep_name: {sweep_name}",
        f"configurations_tested: {total}",
        f"gt_radius_px (Euclidean, vs final defect_mask): {gt_radius}",
        f"best_objective: {best_obj}",
        f"any_exact_match (case1=3, case2=3, case3=0): {any_exact}",
        "",
        "Top 10 rows by (exact_match desc, objective asc):",
    ]
    for r in top10:
        lines.append(
            f"  {r['config_id']} obj={r['objective']} exact={r['exact_match']} "
            f"counts=({r['case1_count']},{r['case2_count']},{r['case3_count']}) "
            f"gt_cov_r=({r['case1_gt_covered_within_radius']}/{r['case1_gt_total']},"
            f"{r['case2_gt_covered_within_radius']}/{r['case2_gt_total']}) "
            f"frac12={r['gt_coverage_fraction_within_radius_case12']} "
            f"comp={r['comparator_choice']} k={r['threshold_k']} rank={r['ranking_mode']}"
        )
    lines.append("")
    if any_exact:
        lines.append("Exact-match configurations:")
        for r in summary_rows:
            if r["exact_match"]:
                lines.append(f"  {r['config_id']}  {r['comparator_params_json']}")
    else:
        lines.append("No exact match found. Closest objectives are listed above.")

    lines.append("")
    lines.append("Best configuration (lowest objective):")
    if summary_rows:
        b = min(summary_rows, key=lambda x: (x["objective"], x["config_id"]))
        lines.append(f"  config_id={b['config_id']} objective={b['objective']}")
        lines.append(
            f"  counts case1={b['case1_count']} case2={b['case2_count']} case3={b['case3_count']}"
        )

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[sweep] wrote {csv_path}")
    print(f"[sweep] wrote {txt_path}")
    print(f"[sweep] configurations_tested={total} best_objective={best_obj} any_exact_match={any_exact}")
    print("[sweep] top config_ids:", ", ".join(r["config_id"] for r in top10[:5]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Focused comparator/threshold/postprocess sweep with cached normalization.")
    ap.add_argument("--sweep-name", type=str, default="focused_v1", help="Subfolder under outs/sweeps/")
    ap.add_argument(
        "--max-configs",
        type=int,
        default=400,
        help="Max sweep configurations (default 400). Full grid is huge; uses seeded shuffle when truncating. Use -1 for no cap.",
    )
    ap.add_argument(
        "--gt-radius",
        type=float,
        default=5.0,
        help="Euclidean distance (px): GT point covered if nearest True defect pixel is within this radius (see utils.gt_coverage)",
    )
    args = ap.parse_args()
    max_c = None if args.max_configs < 0 else args.max_configs

    samples = load_sample_pairs(
        root_pattern=ROOT_PATTERN,
        inspected_pattern=INSPECTED_PATTERN,
        reference_pattern=REFERENCE_PATTERN,
        recursive=False,
        sort_results=True,
    )
    if not samples:
        print("[sweep] No samples found; check ROOT_PATTERN and filenames.")
        sys.exit(1)

    run_sweep(sweep_name=args.sweep_name, max_configs=max_c, samples=samples, gt_radius=float(args.gt_radius))


if __name__ == "__main__":
    main()
