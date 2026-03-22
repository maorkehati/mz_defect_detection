from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModuleChoiceConfig:
    preprocessing: str = "gaussian_preprocess"
    alignment: str = "translation_phase_correlation"
    normalization: str = "linear_gain_offset"
    comparison: str = "absolute_difference"
    thresholding: str = "mad_threshold"
    postprocessing: str = "basic_morphology"


@dataclass
class PreprocessingConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "convert_to_grayscale": True,
        "gaussian_sigma": 0.8,
    })


@dataclass
class AlignmentConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "subpixel_refinement": True,
        "max_shift": None,
        "interpolation_order": 1,
    })


@dataclass
class OrbAffineAlignmentConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "nfeatures": 2000,
        "top_matches": 300,
        "min_matches_for_estimation": 12,
        "ransac_reproj_threshold": 3.0,
        "allow_fallback_to_identity": True,
    })


@dataclass
class EccTranslationAlignmentConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "number_of_iterations": 200,
        "termination_eps": 1e-6,
        "gaussian_filter_size": 5,
        "allow_fallback_to_identity": True,
        "use_gradient_images": False,
    })


@dataclass
class EccEuclideanAlignmentConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "number_of_iterations": 200,
        "termination_eps": 1e-6,
        "gaussian_filter_size": 5,
        "allow_fallback_to_identity": True,
        "use_gradient_images": False,
    })


@dataclass
class EccAffineAlignmentConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "number_of_iterations": 200,
        "termination_eps": 1e-6,
        "gaussian_filter_size": 5,
        "allow_fallback_to_identity": True,
        "use_gradient_images": False,
    })


@dataclass
class EccAffineProjectedEuclideanAlignmentConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "number_of_iterations": 500,
        "termination_eps": 1e-6,
        "gaussian_filter_size": 5,
        "allow_fallback_to_identity": True,
        "use_gradient_images": False,
        "translation_refine_radius_x_px_stage1": 2.0,
        "translation_refine_radius_y_px_stage1": 2.0,
        "translation_refine_step_x_px_stage1": 0.5,
        "translation_refine_step_y_px_stage1": 0.5,
        "translation_refine_radius_x_px_stage2": 0.5,
        "translation_refine_radius_y_px_stage2": 0.5,
        "translation_refine_step_x_px_stage2": 0.1,
        "translation_refine_step_y_px_stage2": 0.1,
    })


@dataclass
class SearchEuclideanAlignmentConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "coarse_angle_min": -4.0,
        "coarse_angle_max": 4.0,
        "coarse_steps": 17,
        "refine_half_width": 0.75,
        "refine_steps": 15,
        "overlap_threshold": 0.92,
        "upsample_factor": 20,
    })


@dataclass
class NormalizationConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "fit_gain": True,
        "fit_offset": True,
        "robust": True,
        "clip_output": False,
        "min_fit_pixels": 1000,
    })


@dataclass
class ComparisonConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "gradient_weight": 0.25,
        "coarse_sigma": 2.0,
        "coarse_weight": 0.25,
    })


@dataclass
class SsimComparatorConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "use_valid_mask": True,
        "win_size": 7,
    })


@dataclass
class GradientDifferenceComparatorConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "pre_blur_sigma": 1.0,
        "post_blur_sigma": 1.0,
        "gradient_ksize": 3,
        "norm_percentile_low": 1.0,
        "norm_percentile_high": 99.0,
        "use_valid_mask": True,
        # Optional; disabled by default to preserve baseline behavior.
        "edge_suppression_enabled": False,
        "edge_percentile": 85.0,
        "edge_weight_on_edges": 0.35,
    })


@dataclass
class ArtifactResidualComparatorConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "pre_blur_sigma": 1.0,
        # Prefer top_hat_kernel_size; tophat_kernel_size kept as alias for older configs.
        "top_hat_kernel_size": 9,
        "top_hat_iterations": 1,
        "combine_mode": "max",
        "norm_percentile_low": 1.0,
        "norm_percentile_high": 99.0,
        "use_valid_mask": True,
        "edge_mode": "hard",
        "edge_percentile": 90.0,
        "edge_dilate_kernel": 5,
        "edge_dilate_iterations": 1,
        "edge_weight_on_edges": 0.25,
        "edge_gradient_ksize": 3,
        "edge_source": "inspected",
        "min_valid_fraction": 0.0,
        "debug_save_intermediates": True,
    })


@dataclass
class ThresholdingConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "k_mad": 4.0,
        "min_threshold": 0.0,
        "use_valid_mask": True,
        "use_core_mask": True,
        "core_erode_iterations": 1,
    })


@dataclass
class OtsuThresholdConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "use_valid_mask": True,
    })


@dataclass
class FixedThresholdConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "threshold_value": 0.15,
        "use_valid_mask": True,
    })


@dataclass
class PostprocessingConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "remove_small_objects": False,
        "min_component_area": 1,
        "morph_open_iterations": 0,
        "morph_close_iterations": 1,
    })


@dataclass
class ContourFilterPostprocessConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "min_area": 1.0,
        "max_area": None,
        # None = no limit for aspect / fill (backward compatible).
        "max_aspect_ratio": None,
        "min_fill_ratio": None,
        "exclude_border_touching": False,
        "border_margin_px": 2,
        # Max detections after ranking + score filter (cap only, not a quota).
        "top_k_keep": None,
        # integrated_anomaly | intensity_size_balanced | intensity_peak_balanced | local_contrast_balanced | artifact_consistent_local_contrast
        "ranking_mode": "integrated_anomaly",
        # Signed-residual consistency: computed for audit/ranking. Hard rejection requires both
        # ``reject_on_low_sign_consistency`` True and ``min_sign_consistency`` set (see contour_filter_postprocess).
        "min_sign_consistency": None,
        # If True and ``min_sign_consistency`` is set, drop candidates below the threshold (hard gate).
        # Default False: never hard-reject on sign (ranking modes may still use sign softly).
        "reject_on_low_sign_consistency": False,
        # Background ring half-width for local_contrast_balanced (dilate radius in px; 0 = skip ring).
        "ring_radius_px": 0,
        # Contour-level gate on ranking_score (None = disabled). Applied before top_k_keep cap.
        "min_contour_score": None,
        "contour_score_threshold_mode": "absolute",
        # Pre-contour morphology on the threshold mask (0 = disabled for each).
        "morph_open_kernel": 0,
        "morph_open_iterations": 0,
        "morph_close_kernel": 0,
        "morph_close_iterations": 0,
    })


@dataclass
class OutputConfig:
    save_intermediate: bool = False
    save_dir: Optional[str] = None
    return_artifacts: bool = True


@dataclass
class DebugConfig:
    enable_debug_visualization: bool = True
    save_debug_images: bool = True
    show_debug_images: bool = False
    debug_dir: Optional[str] = None


@dataclass
class PipelineConfig:
    choices: ModuleChoiceConfig = field(default_factory=ModuleChoiceConfig)

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    orb_affine_alignment: OrbAffineAlignmentConfig = field(default_factory=OrbAffineAlignmentConfig)
    ecc_translation_alignment: EccTranslationAlignmentConfig = field(default_factory=EccTranslationAlignmentConfig)
    ecc_euclidean_alignment: EccEuclideanAlignmentConfig = field(default_factory=EccEuclideanAlignmentConfig)
    ecc_affine_alignment: EccAffineAlignmentConfig = field(default_factory=EccAffineAlignmentConfig)
    ecc_affine_projected_euclidean_alignment: EccAffineProjectedEuclideanAlignmentConfig = field(
        default_factory=EccAffineProjectedEuclideanAlignmentConfig
    )
    search_euclidean_alignment: SearchEuclideanAlignmentConfig = field(
        default_factory=SearchEuclideanAlignmentConfig
    )
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    ssim_comparator: SsimComparatorConfig = field(default_factory=SsimComparatorConfig)
    gradient_difference: GradientDifferenceComparatorConfig = field(default_factory=GradientDifferenceComparatorConfig)
    artifact_residual: ArtifactResidualComparatorConfig = field(default_factory=ArtifactResidualComparatorConfig)
    thresholding: ThresholdingConfig = field(default_factory=ThresholdingConfig)
    otsu_threshold: OtsuThresholdConfig = field(default_factory=OtsuThresholdConfig)
    fixed_threshold: FixedThresholdConfig = field(default_factory=FixedThresholdConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    contour_filter_postprocess: ContourFilterPostprocessConfig = field(
        default_factory=ContourFilterPostprocessConfig
    )

    output: OutputConfig = field(default_factory=OutputConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    fail_on_shape_mismatch: bool = True
    verbose: bool = True


def build_default_config() -> PipelineConfig:
    return PipelineConfig(
        choices=ModuleChoiceConfig(
            preprocessing="gaussian_preprocess",
            alignment="translation_phase_correlation",
            normalization="linear_gain_offset",
            comparison="absolute_difference",
            thresholding="mad_threshold",
            postprocessing="basic_morphology",
        ),
        debug=DebugConfig(
            enable_debug_visualization=True,
            save_debug_images=True,
            show_debug_images=False,
            debug_dir=None,
        ),
    )


def build_ssim_config() -> PipelineConfig:
    cfg = build_default_config()
    cfg.choices.comparison = "ssim_comparator"
    cfg.choices.thresholding = "mad_threshold"
    cfg.ssim_comparator.params["use_valid_mask"] = True
    cfg.ssim_comparator.params["win_size"] = 7
    return cfg


def build_orb_ssim_config() -> PipelineConfig:
    cfg = build_default_config()
    cfg.choices.alignment = "orb_affine"
    cfg.choices.comparison = "ssim_comparator"
    cfg.choices.thresholding = "mad_threshold"
    cfg.orb_affine_alignment.params["nfeatures"] = 2000
    cfg.orb_affine_alignment.params["top_matches"] = 300
    cfg.ssim_comparator.params["use_valid_mask"] = True
    cfg.ssim_comparator.params["win_size"] = 7
    return cfg


def build_orb_ssim_otsu_config() -> PipelineConfig:
    cfg = build_default_config()
    cfg.choices.alignment = "orb_affine"
    cfg.choices.comparison = "ssim_comparator"
    cfg.choices.thresholding = "otsu_threshold"
    cfg.choices.postprocessing = "contour_filter_postprocess"
    cfg.orb_affine_alignment.params["nfeatures"] = 2000
    cfg.orb_affine_alignment.params["top_matches"] = 300
    cfg.ssim_comparator.params["use_valid_mask"] = True
    cfg.ssim_comparator.params["win_size"] = 7
    cfg.otsu_threshold.params["use_valid_mask"] = True
    cfg.contour_filter_postprocess.params["min_area"] = 1.0
    cfg.contour_filter_postprocess.params["max_area"] = None
    return cfg


def build_orb_ssim_fixed_threshold_config() -> PipelineConfig:
    cfg = build_default_config()
    cfg.choices.alignment = "orb_affine"
    cfg.choices.comparison = "ssim_comparator"
    cfg.choices.thresholding = "fixed_threshold"
    cfg.choices.postprocessing = "contour_filter_postprocess"
    cfg.orb_affine_alignment.params["nfeatures"] = 2000
    cfg.orb_affine_alignment.params["top_matches"] = 300
    cfg.ssim_comparator.params["use_valid_mask"] = True
    cfg.ssim_comparator.params["win_size"] = 7
    cfg.fixed_threshold.params["threshold_value"] = 0.15
    cfg.fixed_threshold.params["use_valid_mask"] = True
    cfg.contour_filter_postprocess.params["min_area"] = 1.0
    cfg.contour_filter_postprocess.params["max_area"] = None
    return cfg


def build_ecc_translation_ssim_config() -> PipelineConfig:
    cfg = build_default_config()
    cfg.choices.alignment = "ecc_translation"
    cfg.choices.comparison = "ssim_comparator"
    cfg.choices.thresholding = "mad_threshold"
    cfg.ssim_comparator.params["use_valid_mask"] = True
    cfg.ssim_comparator.params["win_size"] = 7
    return cfg


def build_ecc_euclidean_ssim_config() -> PipelineConfig:
    cfg = build_default_config()
    cfg.choices.alignment = "ecc_euclidean"
    cfg.choices.comparison = "ssim_comparator"
    cfg.choices.thresholding = "mad_threshold"
    cfg.ssim_comparator.params["use_valid_mask"] = True
    cfg.ssim_comparator.params["win_size"] = 7
    return cfg


def build_ecc_euclidean_ssim_otsu_config() -> PipelineConfig:
    cfg = build_default_config()
    cfg.choices.alignment = "ecc_euclidean"
    cfg.choices.comparison = "ssim_comparator"
    cfg.choices.thresholding = "otsu_threshold"
    cfg.choices.postprocessing = "contour_filter_postprocess"
    cfg.ssim_comparator.params["use_valid_mask"] = True
    cfg.ssim_comparator.params["win_size"] = 7
    cfg.otsu_threshold.params["use_valid_mask"] = True
    cfg.contour_filter_postprocess.params["min_area"] = 1.0
    cfg.contour_filter_postprocess.params["max_area"] = None
    return cfg


def build_ecc_euclidean_ssim_fixed_threshold_config() -> PipelineConfig:
    cfg = build_default_config()
    cfg.choices.alignment = "ecc_euclidean"
    cfg.choices.comparison = "ssim_comparator"
    cfg.choices.thresholding = "fixed_threshold"
    cfg.choices.postprocessing = "contour_filter_postprocess"
    cfg.ssim_comparator.params["use_valid_mask"] = True
    cfg.ssim_comparator.params["win_size"] = 7
    cfg.fixed_threshold.params["threshold_value"] = 0.15
    cfg.fixed_threshold.params["use_valid_mask"] = True
    cfg.contour_filter_postprocess.params["min_area"] = 1.0
    cfg.contour_filter_postprocess.params["max_area"] = None
    return cfg


def build_ecc_affine_config() -> PipelineConfig:
    cfg = build_default_config()
    cfg.choices.preprocessing = "gaussian_preprocess"
    cfg.choices.alignment = "ecc_affine"
    cfg.choices.normalization = "linear_gain_offset"
    cfg.choices.comparison = "ssim_comparator"
    cfg.choices.thresholding = "otsu_threshold"
    cfg.choices.postprocessing = "contour_filter_postprocess"

    # Explicit affine-ECC constants (robust but not over-tuned).
    cfg.ecc_affine_alignment.params["number_of_iterations"] = 500
    cfg.ecc_affine_alignment.params["termination_eps"] = 1e-6
    cfg.ecc_affine_alignment.params["gaussian_filter_size"] = 5
    cfg.ecc_affine_alignment.params["allow_fallback_to_identity"] = True
    cfg.ecc_affine_alignment.params["use_gradient_images"] = False

    cfg.ssim_comparator.params["use_valid_mask"] = True
    cfg.ssim_comparator.params["win_size"] = 7
    cfg.otsu_threshold.params["use_valid_mask"] = True
    cfg.contour_filter_postprocess.params["min_area"] = 1.0
    cfg.contour_filter_postprocess.params["max_area"] = None
    return cfg


def build_ecc_affine_ssim_otsu_config() -> PipelineConfig:
    cfg = build_ecc_affine_config()
    cfg.contour_filter_postprocess.params["min_area"] = 5.0
    return cfg


def build_ecc_affine_projected_euclidean_config() -> PipelineConfig:
    cfg = build_default_config()
    cfg.choices.preprocessing = "gaussian_preprocess"
    cfg.choices.alignment = "ecc_affine_projected_euclidean"
    cfg.choices.normalization = "linear_gain_offset"
    cfg.choices.comparison = "ssim_comparator"
    cfg.choices.thresholding = "otsu_threshold"
    cfg.choices.postprocessing = "contour_filter_postprocess"
    cfg.ssim_comparator.params["use_valid_mask"] = True
    cfg.ssim_comparator.params["win_size"] = 7
    cfg.otsu_threshold.params["use_valid_mask"] = True
    cfg.contour_filter_postprocess.params["min_area"] = 1.0
    cfg.contour_filter_postprocess.params["max_area"] = None
    return cfg


def build_ecc_affine_projected_euclidean_ssim_otsu_config() -> PipelineConfig:
    cfg = build_ecc_affine_projected_euclidean_config()
    cfg.contour_filter_postprocess.params["min_area"] = 5.0
    return cfg


def build_search_euclidean_ssim_otsu_config() -> PipelineConfig:
    cfg = build_default_config()
    cfg.choices.preprocessing = "gaussian_preprocess"
    cfg.choices.alignment = "search_euclidean"
    cfg.choices.normalization = "linear_gain_offset"
    cfg.choices.comparison = "ssim_comparator"
    cfg.choices.thresholding = "otsu_threshold"
    cfg.choices.postprocessing = "contour_filter_postprocess"

    # Explicit script-matching alignment constants.
    cfg.search_euclidean_alignment.params["coarse_angle_min"] = -4.0
    cfg.search_euclidean_alignment.params["coarse_angle_max"] = 4.0
    cfg.search_euclidean_alignment.params["coarse_steps"] = 17
    cfg.search_euclidean_alignment.params["refine_half_width"] = 0.75
    cfg.search_euclidean_alignment.params["refine_steps"] = 15
    cfg.search_euclidean_alignment.params["overlap_threshold"] = 0.92
    cfg.search_euclidean_alignment.params["upsample_factor"] = 20

    cfg.otsu_threshold.params["use_valid_mask"] = True
    cfg.ssim_comparator.params["use_valid_mask"] = True
    cfg.ssim_comparator.params["win_size"] = 7
    cfg.contour_filter_postprocess.params["min_area"] = 5.0
    cfg.contour_filter_postprocess.params["max_area"] = None
    return cfg


def build_search_euclidean_edge_distance_ssim_otsu_config() -> PipelineConfig:
    # Kept for backward variant compatibility; uses the same
    # structure-driven search implementation as build_search_euclidean_ssim_otsu_config.
    return build_search_euclidean_ssim_otsu_config()


def build_search_euclidean_gradient_difference_otsu_config() -> PipelineConfig:
    cfg = build_default_config()
    cfg.choices.preprocessing = "gaussian_preprocess"
    cfg.choices.alignment = "search_euclidean"
    cfg.choices.normalization = "linear_gain_offset"
    cfg.choices.comparison = "gradient_difference"
    cfg.choices.thresholding = "otsu_threshold"
    cfg.choices.postprocessing = "contour_filter_postprocess"

    cfg.search_euclidean_alignment.params["coarse_angle_min"] = -4.0
    cfg.search_euclidean_alignment.params["coarse_angle_max"] = 4.0
    cfg.search_euclidean_alignment.params["coarse_steps"] = 17
    cfg.search_euclidean_alignment.params["refine_half_width"] = 0.75
    cfg.search_euclidean_alignment.params["refine_steps"] = 15
    cfg.search_euclidean_alignment.params["overlap_threshold"] = 0.92
    cfg.search_euclidean_alignment.params["upsample_factor"] = 20

    cfg.gradient_difference.params["pre_blur_sigma"] = 1.0
    cfg.gradient_difference.params["post_blur_sigma"] = 1.0
    cfg.gradient_difference.params["gradient_ksize"] = 3
    cfg.gradient_difference.params["norm_percentile_low"] = 1.0
    cfg.gradient_difference.params["norm_percentile_high"] = 99.0
    cfg.gradient_difference.params["use_valid_mask"] = True

    cfg.otsu_threshold.params["use_valid_mask"] = True
    cfg.contour_filter_postprocess.params["min_area"] = 5.0
    cfg.contour_filter_postprocess.params["max_area"] = None
    return cfg


def build_search_euclidean_gradient_difference_mad_config() -> PipelineConfig:
    cfg = build_default_config()
    cfg.choices.preprocessing = "gaussian_preprocess"
    cfg.choices.alignment = "search_euclidean"
    cfg.choices.normalization = "linear_gain_offset"
    cfg.choices.comparison = "gradient_difference"
    cfg.choices.thresholding = "mad_threshold"
    cfg.choices.postprocessing = "contour_filter_postprocess"

    cfg.search_euclidean_alignment.params["coarse_angle_min"] = -4.0
    cfg.search_euclidean_alignment.params["coarse_angle_max"] = 4.0
    cfg.search_euclidean_alignment.params["coarse_steps"] = 17
    cfg.search_euclidean_alignment.params["refine_half_width"] = 0.75
    cfg.search_euclidean_alignment.params["refine_steps"] = 15
    cfg.search_euclidean_alignment.params["overlap_threshold"] = 0.92
    cfg.search_euclidean_alignment.params["upsample_factor"] = 20

    cfg.gradient_difference.params["pre_blur_sigma"] = 1.0
    cfg.gradient_difference.params["post_blur_sigma"] = 1.0
    cfg.gradient_difference.params["gradient_ksize"] = 3
    cfg.gradient_difference.params["norm_percentile_low"] = 1.0
    cfg.gradient_difference.params["norm_percentile_high"] = 99.0
    cfg.gradient_difference.params["use_valid_mask"] = True
    cfg.gradient_difference.params["edge_suppression_enabled"] = False
    cfg.gradient_difference.params["edge_percentile"] = 85.0
    cfg.gradient_difference.params["edge_weight_on_edges"] = 0.35

    cfg.thresholding.params["k_mad"] = 4.0
    cfg.thresholding.params["min_threshold"] = 0.0
    cfg.thresholding.params["use_valid_mask"] = True
    cfg.thresholding.params["use_core_mask"] = True
    cfg.thresholding.params["core_erode_iterations"] = 1
    # Stricter geometry for wafer/chip imagery: drop thin edge fragments and border clutter before ranking/top-K.
    cfg.contour_filter_postprocess.params["min_area"] = 30.0
    cfg.contour_filter_postprocess.params["max_area"] = None
    cfg.contour_filter_postprocess.params["max_aspect_ratio"] = 5.0
    cfg.contour_filter_postprocess.params["min_fill_ratio"] = 0.20
    cfg.contour_filter_postprocess.params["exclude_border_touching"] = True
    cfg.contour_filter_postprocess.params["border_margin_px"] = 3
    cfg.contour_filter_postprocess.params["top_k_keep"] = 5
    cfg.contour_filter_postprocess.params["ranking_mode"] = "artifact_consistent_local_contrast"
    # No hard sign-consistency threshold; sign is soft modifier in ranking only.
    cfg.contour_filter_postprocess.params["min_sign_consistency"] = None
    cfg.contour_filter_postprocess.params["ring_radius_px"] = 7
    cfg.contour_filter_postprocess.params["morph_open_kernel"] = 3
    cfg.contour_filter_postprocess.params["morph_open_iterations"] = 1
    cfg.contour_filter_postprocess.params["morph_close_kernel"] = 5
    cfg.contour_filter_postprocess.params["morph_close_iterations"] = 1
    cfg.contour_filter_postprocess.params["min_contour_score"] = 4.5
    cfg.contour_filter_postprocess.params["contour_score_threshold_mode"] = "absolute"
    return cfg


def build_search_euclidean_gradient_difference_edge_suppressed_mad_config() -> PipelineConfig:
    cfg = build_search_euclidean_gradient_difference_mad_config()
    cfg.gradient_difference.params["edge_suppression_enabled"] = True
    cfg.gradient_difference.params["edge_percentile"] = 85.0
    cfg.gradient_difference.params["edge_weight_on_edges"] = 0.35
    return cfg


def build_search_euclidean_artifact_residual_mad_config() -> PipelineConfig:
    """
    Primary focused path for defect detection using **artifact_residual** (signed residual + white top-hat).

    Pipeline: ``gaussian_preprocess`` → ``search_euclidean`` → ``linear_gain_offset``
    → ``artifact_residual`` → ``mad_threshold`` → ``contour_filter_postprocess``.

    This builder is **standalone** (not derived from the gradient-difference MAD path) so
    comparator-specific defaults stay clean. The gradient-difference MAD configuration remains
    :func:`build_search_euclidean_gradient_difference_mad_config`.

    Defaults favor a conservative first run (higher ``k_mad``, simpler contour ranking, **no**
    hard sign-consistency rejection: ``min_sign_consistency`` is ``None`` and
    ``reject_on_low_sign_consistency`` is ``False``).

    ``debug_save_intermediates`` is **True** so the first comparator pass populates
    ``PipelineArtifacts.artifact_residual_intermediates`` for diagnostics and avoids an extra
    comparator run when saving ``*_artifact_residual_debug.png`` (see ``visualization.debug``).
    """
    cfg = build_default_config()
    cfg.choices.preprocessing = "gaussian_preprocess"
    cfg.choices.alignment = "search_euclidean"
    cfg.choices.normalization = "linear_gain_offset"
    cfg.choices.comparison = "artifact_residual"
    cfg.choices.thresholding = "mad_threshold"
    cfg.choices.postprocessing = "contour_filter_postprocess"

    # Same search-Euclidean alignment grid as other focused ``search_euclidean_*`` builders.
    cfg.search_euclidean_alignment.params["coarse_angle_min"] = -4.0
    cfg.search_euclidean_alignment.params["coarse_angle_max"] = 4.0
    cfg.search_euclidean_alignment.params["coarse_steps"] = 17
    cfg.search_euclidean_alignment.params["refine_half_width"] = 0.75
    cfg.search_euclidean_alignment.params["refine_steps"] = 15
    cfg.search_euclidean_alignment.params["overlap_threshold"] = 0.92
    cfg.search_euclidean_alignment.params["upsample_factor"] = 20

    cfg.artifact_residual.params.update(
        {
            "pre_blur_sigma": 1.0,
            "top_hat_kernel_size": 9,
            "top_hat_iterations": 1,
            "combine_mode": "max",
            "norm_percentile_low": 1.0,
            "norm_percentile_high": 99.0,
            "use_valid_mask": True,
            "edge_mode": "hard",
            "edge_percentile": 90.0,
            "edge_dilate_kernel": 5,
            "edge_dilate_iterations": 1,
            "edge_weight_on_edges": 0.25,
            "edge_gradient_ksize": 3,
            "edge_source": "inspected",
            "min_valid_fraction": 0.0,
            # Populate intermediates on the main run so diagnostic PNGs need no extra comparator pass.
            "debug_save_intermediates": True,
        }
    )

    # Slightly conservative MAD vs. the gradient path (k=4): fewer false positives on first run.
    cfg.thresholding.params["k_mad"] = 5.0
    cfg.thresholding.params["min_threshold"] = 0.0
    cfg.thresholding.params["use_valid_mask"] = True
    cfg.thresholding.params["use_core_mask"] = True
    cfg.thresholding.params["core_erode_iterations"] = 1

    # Simpler postprocessing: speckle/border/geometry filters + anomaly-centric ranking; no sign hard gate.
    cfg.contour_filter_postprocess.params.update(
        {
            "min_area": 8.0,
            "max_area": 12000.0,
            "max_aspect_ratio": 8.0,
            "min_fill_ratio": 0.12,
            "exclude_border_touching": True,
            "border_margin_px": 3,
            "top_k_keep": 6,
            "ranking_mode": "intensity_size_balanced",
            "min_sign_consistency": None,
            "reject_on_low_sign_consistency": False,
            "ring_radius_px": 0,
            "min_contour_score": None,
            "contour_score_threshold_mode": "absolute",
            "morph_open_kernel": 3,
            "morph_open_iterations": 1,
            "morph_close_kernel": 3,
            "morph_close_iterations": 1,
        }
    )

    return cfg