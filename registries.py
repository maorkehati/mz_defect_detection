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
# These map config choice strings to concrete module classes.
from modules.alignment import TranslationPhaseCorrelationAligner
from modules.comparison import AbsoluteDifferenceComparator
from modules.normalization import LinearGainOffsetNormalizer
from modules.postprocessing import BasicMorphologyPostprocessor
from modules.preprocessing import GaussianPreprocessor
from modules.thresholding import MadThresholding

PREPROCESSOR_REGISTRY = {"gaussian_preprocess": GaussianPreprocessor}
ALIGNER_REGISTRY = {"translation_phase_correlation": TranslationPhaseCorrelationAligner}
NORMALIZER_REGISTRY = {"linear_gain_offset": LinearGainOffsetNormalizer}
COMPARATOR_REGISTRY = {"absolute_difference": AbsoluteDifferenceComparator}
THRESHOLDING_REGISTRY = {"mad_threshold": MadThresholding}
POSTPROCESSOR_REGISTRY = {"basic_morphology": BasicMorphologyPostprocessor}

__all__ = [
    "PREPROCESSOR_REGISTRY",
    "ALIGNER_REGISTRY",
    "NORMALIZER_REGISTRY",
    "COMPARATOR_REGISTRY",
    "THRESHOLDING_REGISTRY",
    "POSTPROCESSOR_REGISTRY",
]