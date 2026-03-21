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
class NormalizationConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "fit_gain": True,
        "fit_offset": True,
        "robust": True,
        "clip_output": False,
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
class ThresholdingConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=lambda: {
        "k_mad": 4.0,
        "min_threshold": 0.0,
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
class OutputConfig:
    save_intermediate: bool = False
    save_dir: Optional[str] = None
    return_artifacts: bool = True


@dataclass
class PipelineConfig:
    choices: ModuleChoiceConfig = field(default_factory=ModuleChoiceConfig)

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    thresholding: ThresholdingConfig = field(default_factory=ThresholdingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)

    output: OutputConfig = field(default_factory=OutputConfig)

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
        )
    )


# Registry dictionaries used by `factories.py`.
#
# Entries are lazy constructors so optional dependencies only load
# when a specific method is selected.
def _build_gaussian_preprocess():
    from modules.preprocessing.gaussian_preprocess import GaussianPreprocessor

    return GaussianPreprocessor()


def _build_translation_phase_correlation():
    from modules.alignment.translation_phase_correlation import TranslationPhaseCorrelationAligner

    return TranslationPhaseCorrelationAligner()


def _build_orb_affine():
    from modules.alignment.orb_affine import OrbAffineAligner

    return OrbAffineAligner()


def _build_ecc_translation():
    from modules.alignment.ecc_alignment import EccTranslationAligner

    return EccTranslationAligner()


def _build_ecc_euclidean():
    from modules.alignment.ecc_alignment import EccEuclideanAligner

    return EccEuclideanAligner()


def _build_ecc_affine():
    from modules.alignment.ecc_alignment import EccAffineAligner

    return EccAffineAligner()


def _build_ecc_affine_projected_euclidean():
    from modules.alignment.ecc_alignment import EccAffineProjectedEuclideanAligner

    return EccAffineProjectedEuclideanAligner()


def _build_search_euclidean():
    from modules.alignment.search_euclidean import SearchEuclideanAligner

    return SearchEuclideanAligner()


def _build_linear_gain_offset():
    from modules.normalization.linear_gain_offset import LinearGainOffsetNormalizer

    return LinearGainOffsetNormalizer()


def _build_absolute_difference():
    from modules.comparison.absolute_difference import AbsoluteDifferenceComparator

    return AbsoluteDifferenceComparator()


def _build_ssim_comparator():
    from modules.comparison.ssim_comparator import SsimComparator

    return SsimComparator()


def _build_gradient_difference():
    from modules.comparison.gradient_difference import GradientDifferenceComparator

    return GradientDifferenceComparator()


def _build_mad_threshold():
    from modules.thresholding.mad_threshold import MadThresholding

    return MadThresholding()


def _build_otsu_threshold():
    from modules.thresholding.otsu_threshold import OtsuThresholding

    return OtsuThresholding()


def _build_fixed_threshold():
    from modules.thresholding.fixed_threshold import FixedThresholding

    return FixedThresholding()


def _build_basic_morphology():
    from modules.postprocessing.basic_morphology import BasicMorphologyPostprocessor

    return BasicMorphologyPostprocessor()


def _build_contour_filter_postprocess():
    from modules.postprocessing.contour_filter_postprocess import ContourFilterPostprocessor

    return ContourFilterPostprocessor()


PREPROCESSOR_REGISTRY = {"gaussian_preprocess": _build_gaussian_preprocess}
ALIGNER_REGISTRY = {
    "translation_phase_correlation": _build_translation_phase_correlation,
    "orb_affine": _build_orb_affine,
    "ecc_translation": _build_ecc_translation,
    "ecc_euclidean": _build_ecc_euclidean,
    "ecc_affine": _build_ecc_affine,
    "ecc_affine_projected_euclidean": _build_ecc_affine_projected_euclidean,
    "search_euclidean": _build_search_euclidean,
}
NORMALIZER_REGISTRY = {"linear_gain_offset": _build_linear_gain_offset}
COMPARATOR_REGISTRY = {
    "absolute_difference": _build_absolute_difference,
    "ssim_comparator": _build_ssim_comparator,
    "gradient_difference": _build_gradient_difference,
}
THRESHOLDING_REGISTRY = {
    "mad_threshold": _build_mad_threshold,
    "otsu_threshold": _build_otsu_threshold,
    "fixed_threshold": _build_fixed_threshold,
}
POSTPROCESSOR_REGISTRY = {
    "basic_morphology": _build_basic_morphology,
    "contour_filter_postprocess": _build_contour_filter_postprocess,
}

__all__ = [
    "PREPROCESSOR_REGISTRY",
    "ALIGNER_REGISTRY",
    "NORMALIZER_REGISTRY",
    "COMPARATOR_REGISTRY",
    "THRESHOLDING_REGISTRY",
    "POSTPROCESSOR_REGISTRY",
]