from __future__ import annotations

from dataclasses import dataclass

ROOT_PATTERN = r"C:\Users\mayoa\Desktop\home exercise\*"
INSPECTED_PATTERN = "case*_inspected_image.tif"
REFERENCE_PATTERN = "case*_reference_image.tif"

SHOW_PLOTS = False
SAVE_PLOTS = True
SAVE_MASKS = True
SAVE_SCORE_MAPS = True

EXPERIMENT_OUTPUT_ROOT_NAME = "experiment_runs"


@dataclass
class ExperimentSpec:
    name: str
    variant: str
    description: str
    overrides: dict[str, object]


"""
Archived experiments are temporarily disabled while we stabilize one strong default configuration.
We are focusing on coarse-to-fine Euclidean search alignment + gradient-difference + MAD threshold + contour filtering first.
"""

EXPERIMENTS: list[ExperimentSpec] = [
    # Chosen as the main stabilization path because:
    # - global alignment is needed
    # - small rotation may be present
    # - ORB was unstable on repetitive shapes
    # - SSIM is more interpretable than raw abs diff here
    # - Otsu is a reasonable non-hand-tuned threshold for first-pass debugging
    ExperimentSpec(
        name="focus_search_euclidean_gradient_difference_edge_suppressed_mad",
        variant="search_euclidean_gradient_difference_edge_suppressed_mad",
        description="Focused main configuration: search Euclidean + gradient-difference with edge suppression + MAD + contour filtering",
        overrides={},
    ),
    # -----------------------------
    # Archived experiments below.
    # -----------------------------
    # ExperimentSpec(
    #     name="focus_ecc_affine_ssim_otsu",
    #     variant="ecc_affine_ssim_otsu",
    #     description="Focused affine ECC configuration: ECC affine alignment + SSIM comparator + Otsu threshold + contour filtering",
    #     overrides={"contour_filter_postprocess.min_area": 5.0},
    # ),
    # ExperimentSpec(
    #     name="focus_ecc_affine_projected_euclidean_ssim_otsu",
    #     variant="ecc_affine_projected_euclidean_ssim_otsu",
    #     description="Focused projected-ECC configuration: affine ECC initialization projected to Euclidean + SSIM + Otsu + contour filtering",
    #     overrides={"contour_filter_postprocess.min_area": 5.0},
    # ),

    # ExperimentSpec(
    #     name="baseline_default",
    #     variant="default",
    #     description="Current baseline: phase-correlation + linear normalization + absolute difference + MAD + morphology",
    #     overrides={},
    # ),
    # ExperimentSpec(
    #     name="ssim_only",
    #     variant="ssim",
    #     description="Swap only the comparator to SSIM while keeping current alignment/normalization",
    #     overrides={},
    # ),
    # """
    # ORB-based experiments are disabled.
    #
    # Reason:
    # - ORB matching is unstable on repetitive geometric structures
    # - Produces inconsistent correspondences
    # - Leads to poor affine estimation and degraded alignment
    #
    # Kept for reference only.
    # """
    # ExperimentSpec(
    #     name="orb_ssim",
    #     variant="orb_ssim",
    #     description="ORB affine alignment + SSIM comparator + MAD threshold",
    #     overrides={},
    # ),
    # ExperimentSpec(
    #     name="orb_ssim_otsu",
    #     variant="orb_ssim_otsu",
    #     description="ORB affine + SSIM + Otsu threshold + contour filtering",
    #     overrides={},
    # ),
    # ExperimentSpec(
    #     name="orb_ssim_fixed_015",
    #     variant="orb_ssim_fixed",
    #     description="ORB affine + SSIM + fixed threshold 0.15 + contour filtering",
    #     overrides={"fixed_threshold.threshold_value": 0.15},
    # ),
    # ExperimentSpec(
    #     name="orb_ssim_fixed_010",
    #     variant="orb_ssim_fixed",
    #     description="ORB affine + SSIM + fixed threshold 0.10 + contour filtering",
    #     overrides={"fixed_threshold.threshold_value": 0.10},
    # ),
    # ExperimentSpec(
    #     name="orb_ssim_fixed_020",
    #     variant="orb_ssim_fixed",
    #     description="ORB affine + SSIM + fixed threshold 0.20 + contour filtering",
    #     overrides={"fixed_threshold.threshold_value": 0.20},
    # ),
    # ExperimentSpec(
    #     name="orb_ssim_otsu_contour_min5",
    #     variant="orb_ssim_otsu",
    #     description="ORB affine + SSIM + Otsu + contour filtering with min_area=5",
    #     overrides={"contour_filter_postprocess.min_area": 5.0},
    # ),
    # ExperimentSpec(
    #     name="orb_ssim_otsu_contour_min20",
    #     variant="orb_ssim_otsu",
    #     description="ORB affine + SSIM + Otsu + contour filtering with min_area=20",
    #     overrides={"contour_filter_postprocess.min_area": 20.0},
    # ),
    # ExperimentSpec(
    #     name="focus_ecc_euclidean_ssim_otsu",
    #     variant="ecc_euclidean_ssim_otsu",
    #     description="Archived focused ECC baseline: ECC Euclidean alignment + SSIM comparator + Otsu threshold + contour filtering",
    #     overrides={"contour_filter_postprocess.min_area": 5.0},
    # ),
    # ExperimentSpec(
    #     name="ecc_euclidean_ssim_fixed_015",
    #     variant="ecc_euclidean_ssim_fixed",
    #     description="ECC Euclidean alignment + SSIM + fixed threshold 0.15 + contour filtering",
    #     overrides={"fixed_threshold.threshold_value": 0.15},
    # ),
    # ExperimentSpec(
    #     name="ecc_euclidean_ssim_fixed_010",
    #     variant="ecc_euclidean_ssim_fixed",
    #     description="ECC Euclidean alignment + SSIM + fixed threshold 0.10 + contour filtering",
    #     overrides={"fixed_threshold.threshold_value": 0.10},
    # ),
    # ExperimentSpec(
    #     name="ecc_euclidean_ssim_fixed_020",
    #     variant="ecc_euclidean_ssim_fixed",
    #     description="ECC Euclidean alignment + SSIM + fixed threshold 0.20 + contour filtering",
    #     overrides={"fixed_threshold.threshold_value": 0.20},
    # ),
    # ExperimentSpec(
    #     name="ecc_euclidean_ssim_otsu_contour_min20",
    #     variant="ecc_euclidean_ssim_otsu",
    #     description="ECC Euclidean alignment + SSIM + Otsu + contour filtering with min_area=20",
    #     overrides={"contour_filter_postprocess.min_area": 20.0},
    # ),
    # ExperimentSpec(
    #     name="ecc_euclidean_ssim_otsu_contour_min5",
    #     variant="ecc_euclidean_ssim_otsu",
    #     description="ECC Euclidean alignment + SSIM + Otsu + contour filtering with min_area=5",
    #     overrides={"contour_filter_postprocess.min_area": 5.0},
    # ),
]

