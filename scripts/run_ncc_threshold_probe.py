"""
NCC comparator threshold probe: sweep ``k_mad`` (and optionally ``patch_size``) with fixed
postprocess/ranking. Does not modify comparator logic.

Default grid: patch_size in {7, 9} × k_mad in {1.5, 2.0, 2.5, 3.0, 3.5} → 10 runs.

Outputs under ``outs/diagnostics/ncc_threshold_probe/<run_id>/`` (per-run logs + CSVs) and
``comparison_summary.txt`` at the output root.

Example:
  python scripts/run_ncc_threshold_probe.py
  python scripts/run_ncc_threshold_probe.py --patches 9 --no-patch-sweep   # k_mad only (5 runs)
"""

from __future__ import annotations

import argparse
import copy
import csv
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import PipelineConfig, build_search_euclidean_artifact_residual_mad_config
from data import load_sample_pairs
from pipeline import DefectDetectionPipeline
from run_ranking_probe import (
    _apply_params,
    _format_top_candidate_line,
    _threshold_snippet,
)
from utils.ground_truth_defects import pair_id_to_case_key

DEFAULT_ROOT_PATTERN = str(REPO_ROOT.parent / "*")
INSPECTED_PATTERN = "case*_inspected_image.tif"
REFERENCE_PATTERN = "case*_reference_image.tif"

OUT_ROOT = REPO_ROOT / "outs" / "diagnostics" / "ncc_threshold_probe"

K_MAD_DEFAULT = [1.5, 2.0, 2.5, 3.0, 3.5]
PATCH_DEFAULT = [7, 9]

FIXED_POST: Dict[str, Any] = {
    "ranking_mode": "contrast_area_log",
    "top_k_keep": 3,
    "min_area": 0.0,
    "min_contour_score": 0.0,
}


def _case_key(pair_id: str) -> str:
    return pair_id_to_case_key(pair_id) or pair_id


def _run_id(patch_size: int, k_mad: float) -> str:
    ks = str(k_mad).replace(".", "p")
    return f"patch{patch_size}_kmad{ks}"


def _build_base_config() -> PipelineConfig:
    base_cfg = build_search_euclidean_artifact_residual_mad_config()
    base_cfg.artifact_residual.params["ncc_eps"] = 1e-6
    base_cfg.artifact_residual.params["sigma_mad_floor"] = 1e-6
    p = base_cfg.contour_filter_postprocess.params
    p["min_area"] = 0.0
    p["min_contour_score"] = 0.0
    p["max_aspect_ratio"] = None
    p["min_fill_ratio"] = None
    p["exclude_border_touching"] = False
    p["min_sign_dominance"] = None
    p["min_asymmetry"] = None
    return base_cfg


def _apply_patch_and_params(cfg: PipelineConfig, row: Dict[str, Any]) -> None:
    cfg.artifact_residual.params["patch_size"] = int(row["patch_size"])
    _apply_params(cfg, row)


def _denom(gt_total: int, fallback: int = 3) -> int:
    return int(gt_total) if int(gt_total) > 0 else fallback


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _run_one_configuration(
    *,
    base_cfg: PipelineConfig,
    row: Dict[str, Any],
    run_dir: Path,
    samples: List[Any],
    cached: Dict[str, Any],
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(base_cfg)
    _apply_patch_and_params(cfg, row)
    pl = DefectDetectionPipeline(cfg)

    detail_lines: List[str] = []
    per_case: Dict[str, Dict[str, Any]] = {}

    ps = int(row["patch_size"])
    km = float(row["k_mad"])
    header = f"run_id={_run_id(ps, km)} patch_size={ps} k_mad={km}"

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
        rm = str(dm.get("ranking_mode", row["ranking_mode"]))

        diag = dm.get("gt_ranking_diagnostics") or {}
        gt_pos = list(diag.get("gt_rank_positions") or [])
        h3 = int(diag.get("gt_hits_in_top3", 0))
        h5 = int(diag.get("gt_hits_in_top5", 0))
        gt_tot = int(diag.get("gt_total", 0))

        ranked_full = list(dm.get("ranked_candidates_full") or [])
        top10 = ranked_full[:10]

        block: List[str] = [
            f"CASE {ck}:",
            f"  {_threshold_snippet(tm, thr_st)}",
        ]
        if cm.get("comparator_mode") == "local_patch_ncc":
            block.append(
                "  ncc("
                f"patch={cm.get('patch_size')}, sigma_mad={cm.get('sigma_mad')}, "
                f"sim_mean={cm.get('similarity_mean')}, sim_p95={cm.get('similarity_p95')}, "
                f"anomaly_mean={cm.get('anomaly_mean_before_scale')}, "
                f"anomaly_p95={cm.get('anomaly_p95_before_scale')}, "
                f"z_mean={cm.get('anomaly_mean_after_z')}, z_p95={cm.get('anomaly_p95_after_z')})"
            )
        block.append(
            f"  post(raw_candidates={raw_n}, ranked_candidates={ranked_n}, kept={kept}, ranking_mode={rm})"
        )
        if gt_tot > 0:
            block.append(f"  gt_hits_top3={h3}/{gt_tot}, gt_hits_top5={h5}/{gt_tot}")
            block.append(f"  gt_rank_positions={gt_pos}")
        else:
            block.append("  gt_rank_positions=[]  (no GT for this case)")
        block.append("  top_candidates=[")
        for rc in top10:
            block.append(f"    {_format_top_candidate_line(rc)},")
        block.append("  ]")
        detail_lines.extend(block)
        detail_lines.append("")

        per_case[ck] = {
            "final": kept,
            "gt_hits_top3": h3,
            "gt_hits_top5": h5,
            "gt_rank_positions": gt_pos,
            "gt_total": gt_tot,
            "ranked_n": ranked_n,
            "raw_n": raw_n,
            "ranked_full": ranked_full,
        }

        if ranked_full:
            csv_path = run_dir / f"{ck}_ranked_candidates.csv"
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
                    crow = {k: rc.get(k) for k in fieldnames if k != "case_id"}
                    crow["case_id"] = ck
                    w.writerow(crow)

    c1 = per_case.get("case1", {})
    c2 = per_case.get("case2", {})
    c3 = per_case.get("case3", {})

    d1 = _denom(int(c1.get("gt_total", 0)))
    d2 = _denom(int(c2.get("gt_total", 0)))

    summary_lines = [
        "RUN SUMMARY",
        (
            f"params: patch_size={ps}, k_mad={km}, ranking_mode={row['ranking_mode']}, "
            f"top_k_keep={row['top_k_keep']}, min_area={row['min_area']}, "
            f"min_contour_score={row['min_contour_score']}"
        ),
        (
            f"case1: final={c1.get('final', 'na')}, "
            f"gt_hit_top3={c1.get('gt_hits_top3', 0)}/{d1}, gt_hit_top5={c1.get('gt_hits_top5', 0)}/{d1}, "
            f"gt_rank_positions={c1.get('gt_rank_positions', [])}"
        ),
        (
            f"case2: final={c2.get('final', 'na')}, "
            f"gt_hit_top3={c2.get('gt_hits_top3', 0)}/{d2}, gt_hit_top5={c2.get('gt_hits_top5', 0)}/{d2}, "
            f"gt_rank_positions={c2.get('gt_rank_positions', [])}"
        ),
        (
            f"case3: final={c3.get('final', 'na')}, false_pos_final={c3.get('final', 'na')}"
        ),
        "",
    ]

    detail_text = "\n".join(detail_lines).rstrip()
    text_lines = [header, ""] + summary_lines + detail_lines

    (run_dir / "log.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    hits12_top3 = int(c1.get("gt_hits_top3", 0)) + int(c2.get("gt_hits_top3", 0))
    f1 = int(c1.get("final", 0))
    f2 = int(c2.get("final", 0))
    f3 = int(c3.get("final", 0))

    compact = "\n".join([header, ""] + summary_lines)

    return {
        "run_id": _run_id(ps, km),
        "patch_size": ps,
        "k_mad": km,
        "row": dict(row),
        "gt_hits_case12_top3": hits12_top3,
        "case3_final": f3,
        "total_detections": f1 + f2 + f3,
        "per_case": per_case,
        "text": "\n".join(text_lines),
        "compact": compact,
        "detail": detail_text,
    }


def _sort_key(rec: Dict[str, Any]) -> Tuple[int, int, int, int, float]:
    """Sort: highest case1+2 top3 hits, then lowest case3 dets, then lowest total dets."""
    return (
        -int(rec["gt_hits_case12_top3"]),
        int(rec["case3_final"]),
        int(rec["total_detections"]),
        int(rec["patch_size"]),
        float(rec["k_mad"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="NCC threshold probe (k_mad ± patch_size).")
    ap.add_argument("--root-pattern", type=str, default=DEFAULT_ROOT_PATTERN)
    ap.add_argument("--inspected-pattern", type=str, default=INSPECTED_PATTERN)
    ap.add_argument("--reference-pattern", type=str, default=REFERENCE_PATTERN)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--out", type=str, default=str(OUT_ROOT))
    ap.add_argument(
        "--k-mad",
        type=str,
        default=",".join(str(x) for x in K_MAD_DEFAULT),
        help="Comma-separated k_mad values (default: 1.5,...,3.5).",
    )
    ap.add_argument(
        "--patches",
        type=str,
        default="7,9",
        help="Comma-separated patch sizes (default: 7,9). Use single value to limit grid.",
    )
    ap.add_argument(
        "--no-patch-sweep",
        action="store_true",
        help="Only sweep k_mad; use first value of --patches as fixed patch_size.",
    )
    args = ap.parse_args()

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    k_list = [float(x.strip()) for x in args.k_mad.split(",") if x.strip()]
    patch_tokens = [x.strip() for x in args.patches.split(",") if x.strip()]
    patch_list = [int(x) for x in patch_tokens]

    if args.no_patch_sweep:
        patch_list = [patch_list[0]]

    combos = list(product(patch_list, k_list))
    if len(combos) > 10:
        print(
            f"Error: {len(combos)} configurations > 10. Reduce --k-mad or --patches.",
            file=sys.stderr,
        )
        sys.exit(1)

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

    base_cfg = _build_base_config()
    pipeline = DefectDetectionPipeline(base_cfg)
    cached: Dict[str, Any] = {}
    for sp in samples:
        up, _ = pipeline.run_through_normalization(sp, silent=True)
        cached[sp.pair_id] = up

    print(f"Loaded {len(samples)} samples. Output root: {out_root}")
    print(f"Configurations ({len(combos)}): patches={patch_list}, k_mad={k_list}")

    results: List[Dict[str, Any]] = []
    for patch_size, k_mad in combos:
        row = {
            **FIXED_POST,
            "patch_size": patch_size,
            "k_mad": k_mad,
        }
        rid = _run_id(patch_size, k_mad)
        run_dir = out_root / rid
        print("\n" + "=" * 72)
        print(f"CONFIG {rid}")
        print("=" * 72)
        rec = _run_one_configuration(
            base_cfg=base_cfg,
            row=row,
            run_dir=run_dir,
            samples=samples,
            cached=cached,
        )
        print(rec["compact"])
        if rec["detail"]:
            print(rec["detail"])
        results.append(rec)

    ordered = sorted(results, key=_sort_key)
    comp_lines = [
        "NCC threshold probe — comparison summary",
        "Sort: (1) max GT hits top3 on case1+case2, (2) min case3 final detections, (3) min total detections",
        "",
    ]
    for i, rec in enumerate(ordered, start=1):
        comp_lines.append(
            f"{i}. {rec['run_id']}  patch={rec['patch_size']}  k_mad={rec['k_mad']}  "
            f"gt12_top3_hits={rec['gt_hits_case12_top3']}  case3_final={rec['case3_final']}  "
            f"total_det={rec['total_detections']}"
        )
    comp_text = "\n".join(comp_lines) + "\n"
    (out_root / "comparison_summary.txt").write_text(comp_text, encoding="utf-8")

    print("\n" + "=" * 72)
    print("ORDERED COMPARISON (best first)")
    print("=" * 72)
    print(comp_text)


if __name__ == "__main__":
    main()
