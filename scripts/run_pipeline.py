"""Main repository entry point for running the full defect-detection pipeline.

This script reads input images from the dataset glob path (input-only) and writes
all outputs under the repository root.
"""

from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import (
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
from visualization import (
    save_anomaly_map,
    save_binary_mask,
    save_detection_figure,
)

# Input dataset location (read-only source path).
ROOT_PATTERN = r"C:\Users\mayoa\Desktop\home exercise\*"
INSPECTED_PATTERN = "case*_inspected_image.tif"
REFERENCE_PATTERN = "case*_reference_image.tif"

SAVE_PLOTS = True
SAVE_MASKS = True
SAVE_SCORE_MAPS = True
RECURSIVE = False
PIPELINE_VARIANT = "search_euclidean_gradient_difference_edge_suppressed_mad"  # one of: search_euclidean_gradient_difference_mad, search_euclidean_gradient_difference_edge_suppressed_mad, search_euclidean_gradient_difference_otsu, search_euclidean_ssim_otsu, default, ssim, orb_ssim, orb_ssim_otsu, orb_ssim_fixed, ecc_translation_ssim, ecc_euclidean_ssim, ecc_euclidean_ssim_otsu, ecc_euclidean_ssim_fixed, ecc_affine_ssim_otsu, ecc_affine_projected_euclidean_ssim_otsu, search_euclidean_edge_distance_ssim_otsu

# Outputs are intentionally rooted inside this repository.
OUTPUT_DIR = REPO_ROOT / "outs" / "detection_results"


def _build_selected_config() -> tuple[str, object]:
    variant = PIPELINE_VARIANT.strip().lower()
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
    else:
        raise ValueError(f"Unknown PIPELINE_VARIANT='{PIPELINE_VARIANT}'.")

    print(f"Active pipeline variant: {variant}")
    print(f"preprocessing: {cfg.choices.preprocessing}")
    print(f"alignment: {cfg.choices.alignment}")
    print(f"normalization: {cfg.choices.normalization}")
    print(f"comparison: {cfg.choices.comparison}")
    print(f"thresholding: {cfg.choices.thresholding}")
    print(f"postprocessing: {cfg.choices.postprocessing}")

    # Example manual overrides:
    # cfg.orb_affine_alignment.params["top_matches"] = 200
    # cfg.ssim_comparator.params["win_size"] = 7
    # cfg.fixed_threshold.params["threshold_value"] = 0.12
    # cfg.contour_filter_postprocess.params["min_area"] = 5.0
    # cfg.ecc_translation_alignment.params["number_of_iterations"] = 300
    # cfg.ecc_euclidean_alignment.params["number_of_iterations"] = 300
    # cfg.ecc_euclidean_alignment.params["termination_eps"] = 1e-7

    return variant, cfg


def run_pipeline() -> None:
    samples = load_sample_pairs(
        root_pattern=ROOT_PATTERN,
        inspected_pattern=INSPECTED_PATTERN,
        reference_pattern=REFERENCE_PATTERN,
        recursive=RECURSIVE,
        sort_results=True,
    )

    _, cfg = _build_selected_config()
    pipeline = DefectDetectionPipeline(cfg)

    if SAVE_PLOTS or SAVE_MASKS or SAVE_SCORE_MAPS:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(samples)} sample pairs from pattern: {ROOT_PATTERN}")

    for idx, sample in enumerate(samples, start=1):
        result = pipeline.run(sample)
        defect_pixels = int(result.defect_mask.astype(bool).sum())
        total_pixels = int(result.defect_mask.size)
        ratio = (defect_pixels / total_pixels) if total_pixels else 0.0

        print(
            f"[{idx:03d}/{len(samples):03d}] {result.pair_id} | "
            f"defect_pixels={defect_pixels} ({ratio:.4%})"
        )

        case_dir = OUTPUT_DIR / result.pair_id
        if SAVE_PLOTS or SAVE_MASKS or SAVE_SCORE_MAPS:
            case_dir.mkdir(parents=True, exist_ok=True)

        if SAVE_PLOTS:
            save_detection_figure(result, case_dir / "detection_panels.png")

        if SAVE_MASKS:
            save_binary_mask(result.defect_mask, case_dir / "defect_mask.png")
            save_binary_mask(result.defect_mask, case_dir / "defect_mask.tiff")

        if SAVE_SCORE_MAPS and result.artifacts.anomaly_map is not None:
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

