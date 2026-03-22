"""Main repository entry point for running the full defect-detection pipeline.

**Primary path (recommended):** artifact_residual comparator + search Euclidean + MAD + peak NMS postprocess::

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config search_euclidean_artifact_residual_mad

This reads sample pairs from a dataset glob (see ``--root-pattern``) and writes outputs under
``<repo>/outs/``. The pipeline also saves compact progression figures and, for artifact_residual,
a separate diagnostic PNG.

Other scripts:
  - ``scripts/run_artifact_residual_sweep.py`` — hyperparameter sweep for artifact_residual only.
  - ``scripts/run_experiments.py`` — batch multiple named experiments from ``configs/experiment_matrix.py``.
  - ``scripts/run_focused_sweep.py`` — mixed comparator sweep (legacy); prefer artifact_residual sweep above.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import (
    build_search_euclidean_artifact_residual_mad_config,
    build_search_euclidean_artifact_residual_peak_nms_config,
    build_search_euclidean_gradient_difference_edge_suppressed_mad_config,
    build_search_euclidean_gradient_difference_mad_config,
    build_search_euclidean_gradient_difference_otsu_config,
    build_ecc_affine_ssim_otsu_config,
    build_ecc_affine_projected_euclidean_ssim_otsu_config,
    build_search_euclidean_edge_distance_ssim_otsu_config,
    build_search_euclidean_ssim_otsu_config,
    build_ecc_euclidean_ssim_config,
    build_ecc_euclidean_ssim_fixed_threshold_config,
    build_ecc_euclidean_ssim_otsu_config,
    build_ecc_translation_ssim_config,
    build_default_config,
    build_orb_ssim_config,
    build_orb_ssim_fixed_threshold_config,
    build_orb_ssim_otsu_config,
    build_ssim_config,
)
from data import load_sample_pairs
from pipeline import DefectDetectionPipeline
from utils.ground_truth_defects import get_ground_truth_points_for_pair
from utils.gt_coverage import compute_gt_point_coverage_metrics
from visualization import (
    save_anomaly_map,
    save_binary_mask,
    save_detection_figure,
)

# Default dataset: parent of the repo (folder containing ``defect_detection`` and dataset cases).
DEFAULT_ROOT_PATTERN = str(REPO_ROOT.parent / "*")
INSPECTED_PATTERN = "case*_inspected_image.tif"
REFERENCE_PATTERN = "case*_reference_image.tif"

# Default "active path" for defect detection (artifact_residual). Override with:
#   python scripts/run_pipeline.py --config search_euclidean_gradient_difference_edge_suppressed_mad
# or env: PIPELINE_VARIANT=...
DEFAULT_PIPELINE_VARIANT = "search_euclidean_artifact_residual_mad"

# All outputs under the repository (flat PNGs + per-case subfolders for legacy panels/masks).
OUTPUT_DIR = REPO_ROOT / "outs" / "detection_results"


def _build_selected_config(variant: str) -> tuple[str, object]:
    variant = variant.strip().lower()
    if variant == "default":
        cfg = build_default_config()
    elif variant == "ssim":
        cfg = build_ssim_config()
    elif variant == "orb_ssim":
        cfg = build_orb_ssim_config()
    elif variant == "orb_ssim_otsu":
        cfg = build_orb_ssim_otsu_config()
    elif variant == "orb_ssim_fixed":
        cfg = build_orb_ssim_fixed_threshold_config()
    elif variant == "ecc_translation_ssim":
        cfg = build_ecc_translation_ssim_config()
    elif variant == "ecc_euclidean_ssim":
        cfg = build_ecc_euclidean_ssim_config()
    elif variant == "ecc_euclidean_ssim_otsu":
        cfg = build_ecc_euclidean_ssim_otsu_config()
    elif variant == "ecc_euclidean_ssim_fixed":
        cfg = build_ecc_euclidean_ssim_fixed_threshold_config()
    elif variant == "ecc_affine_ssim_otsu":
        cfg = build_ecc_affine_ssim_otsu_config()
    elif variant == "ecc_affine_projected_euclidean_ssim_otsu":
        cfg = build_ecc_affine_projected_euclidean_ssim_otsu_config()
    elif variant == "search_euclidean_edge_distance_ssim_otsu":
        cfg = build_search_euclidean_edge_distance_ssim_otsu_config()
    elif variant == "search_euclidean_ssim_otsu":
        cfg = build_search_euclidean_ssim_otsu_config()
    elif variant == "search_euclidean_gradient_difference_otsu":
        cfg = build_search_euclidean_gradient_difference_otsu_config()
    elif variant == "search_euclidean_gradient_difference_mad":
        cfg = build_search_euclidean_gradient_difference_mad_config()
    elif variant == "search_euclidean_gradient_difference_edge_suppressed_mad":
        cfg = build_search_euclidean_gradient_difference_edge_suppressed_mad_config()
    elif variant == "search_euclidean_artifact_residual_mad":
        cfg = build_search_euclidean_artifact_residual_mad_config()
    elif variant == "search_euclidean_artifact_residual_peak_nms":
        cfg = build_search_euclidean_artifact_residual_peak_nms_config()
    else:
        raise ValueError(f"Unknown pipeline variant={variant!r}.")

    print(f"Active pipeline variant: {variant}")
    print(f"preprocessing: {cfg.choices.preprocessing}")
    print(f"alignment: {cfg.choices.alignment}")
    print(f"normalization: {cfg.choices.normalization}")
    print(f"comparison: {cfg.choices.comparison}")
    print(f"thresholding: {cfg.choices.thresholding}")
    print(f"postprocessing: {cfg.choices.postprocessing}")

    return variant, cfg


def _final_detection_count(decision_metadata: dict) -> int:
    dm = decision_metadata or {}
    v = dm.get(
        "final_num_contours",
        dm.get("num_contours_after_topk", dm.get("num_kept_contours", 0)),
    )
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _print_run_summary(
    pair_id: str,
    *,
    decision_metadata: dict,
    defect_mask,
    gt_radius_px: float,
) -> None:
    """One-line summary after pipeline diagnostics (detections + optional GT coverage)."""
    n_det = _final_detection_count(decision_metadata)
    pts = get_ground_truth_points_for_pair(pair_id)
    if not pts:
        print(
            f"[run_pipeline] {pair_id}  final_detections={n_det}  gt_points=0 (no GT file or empty)"
        )
        return
    m = compute_gt_point_coverage_metrics(defect_mask, pts, radius_px=float(gt_radius_px))
    print(
        f"[run_pipeline] {pair_id}  final_detections={n_det}  "
        f"gt_coverage={m.gt_covered_within_radius}/{m.gt_total} within {gt_radius_px:g}px "
        f"(exact {m.gt_covered_exact}/{m.gt_total}, frac_r={m.coverage_fraction_within_radius:.3f})"
    )


def _parse_args() -> argparse.Namespace:
    out = OUTPUT_DIR
    epilog = f"""Examples:
  # Default: artifact_residual path, dataset = parent of repo (see --root-pattern)
  python scripts/run_pipeline.py

  # Explicit config name (same as --variant)
  python scripts/run_pipeline.py --config search_euclidean_artifact_residual_mad

  # Custom dataset location
  python scripts/run_pipeline.py --root-pattern "D:/datasets/my_cases/*"

Primary outputs (under {out}):
  <pair_id>_pipeline.png              — compact 7-panel progression (+ artifact_residual_debug.png)
  <pair_id>/detection_panels.png     — full detection figure (if --save-panels)
  <pair_id>/defect_mask.png          — binary mask (if --save-masks)
Environment:
  PIPELINE_VARIANT   — default config name if --config/--variant omitted
"""
    p = argparse.ArgumentParser(
        description="Run the defect-detection pipeline on all sample pairs from a dataset glob.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    p.add_argument(
        "--variant",
        "--config",
        dest="variant",
        type=str,
        default=os.environ.get("PIPELINE_VARIANT", DEFAULT_PIPELINE_VARIANT),
        metavar="NAME",
        help=(
            "Config builder name (same as --variant). "
            f"Default {DEFAULT_PIPELINE_VARIANT!r} or PIPELINE_VARIANT env."
        ),
    )
    p.add_argument(
        "--root-pattern",
        type=str,
        default=DEFAULT_ROOT_PATTERN,
        help=f"Glob for dataset root(s). Default: parent-of-repo ({DEFAULT_ROOT_PATTERN!r}).",
    )
    p.add_argument(
        "--inspected-pattern",
        type=str,
        default=INSPECTED_PATTERN,
        help="Glob for inspected images.",
    )
    p.add_argument(
        "--reference-pattern",
        type=str,
        default=REFERENCE_PATTERN,
        help="Glob for reference images.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recursive glob for sample pairs.",
    )
    p.add_argument(
        "--gt-radius",
        type=float,
        default=5.0,
        help="Pixels (Euclidean): GT coverage summary uses nearest defect pixel within this radius.",
    )
    p.add_argument(
        "--save-panels",
        dest="save_panels",
        action="store_true",
        default=True,
        help="Save detection_panels.png per case (default: on).",
    )
    p.add_argument(
        "--no-save-panels",
        dest="save_panels",
        action="store_false",
        help="Skip detection_panels.png.",
    )
    p.add_argument(
        "--save-masks",
        dest="save_masks",
        action="store_true",
        default=True,
        help="Save defect_mask.png/.tiff per case (default: on).",
    )
    p.add_argument(
        "--no-save-masks",
        dest="save_masks",
        action="store_false",
        help="Skip binary mask files.",
    )
    p.add_argument(
        "--save-score-maps",
        dest="save_score_maps",
        action="store_true",
        default=True,
        help="Save anomaly score maps (default: on).",
    )
    p.add_argument(
        "--no-save-score-maps",
        dest="save_score_maps",
        action="store_false",
        help="Skip anomaly map exports.",
    )
    return p.parse_args()


def run_pipeline() -> None:
    args = _parse_args()
    samples = load_sample_pairs(
        root_pattern=args.root_pattern,
        inspected_pattern=args.inspected_pattern,
        reference_pattern=args.reference_pattern,
        recursive=args.recursive,
        sort_results=True,
    )

    _, cfg = _build_selected_config(args.variant)
    pipeline = DefectDetectionPipeline(cfg)

    save_panels = args.save_panels
    save_masks = args.save_masks
    save_score_maps = args.save_score_maps

    if save_panels or save_masks or save_score_maps:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(samples)} sample pairs from root_pattern={args.root_pattern!r}")
    print(f"Outputs: {OUTPUT_DIR} (compact + artifact_residual figures written by the pipeline)")

    for idx, sample in enumerate(samples, start=1):
        result = pipeline.run(sample)
        defect_pixels = int(result.defect_mask.astype(bool).sum())
        total_pixels = int(result.defect_mask.size)
        ratio = (defect_pixels / total_pixels) if total_pixels else 0.0

        print(
            f"[{idx:03d}/{len(samples):03d}] {result.pair_id} | "
            f"defect_pixels={defect_pixels} ({ratio:.4%})"
        )
        _print_run_summary(
            result.pair_id,
            decision_metadata=result.artifacts.decision_metadata or {},
            defect_mask=result.defect_mask,
            gt_radius_px=float(args.gt_radius),
        )

        case_dir = OUTPUT_DIR / result.pair_id
        if save_panels or save_masks or save_score_maps:
            case_dir.mkdir(parents=True, exist_ok=True)

        if save_panels:
            save_detection_figure(result, case_dir / "detection_panels.png")

        if save_masks:
            save_binary_mask(result.defect_mask, case_dir / "defect_mask.png")
            save_binary_mask(result.defect_mask, case_dir / "defect_mask.tiff")

        if save_score_maps and result.artifacts.anomaly_map is not None:
            save_anomaly_map(
                result.artifacts.anomaly_map,
                case_dir / "anomaly_score.tiff",
                normalize_for_view=False,
            )
            save_anomaly_map(
                result.artifacts.anomaly_map,
                case_dir / "anomaly_score_display.png",
                normalize_for_view=True,
            )


if __name__ == "__main__":
    run_pipeline()

# Back-compat for notebooks / imports that expect a string constant (matches CLI default):
PIPELINE_VARIANT = DEFAULT_PIPELINE_VARIANT
