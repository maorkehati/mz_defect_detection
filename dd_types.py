from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

Array = np.ndarray


@dataclass
class SamplePair:
    reference_image: Array
    inspected_image: Array
    pair_id: str = ""


@dataclass
class PipelineArtifacts:
    reference_raw: Optional[Array] = None
    inspected_raw: Optional[Array] = None

    # Backward-compatible aliases used by existing visualization code.
    reference_input: Optional[Array] = None
    inspected_input: Optional[Array] = None

    reference_preprocessed: Optional[Array] = None
    inspected_preprocessed: Optional[Array] = None

    reference_aligned: Optional[Array] = None
    inspected_aligned: Optional[Array] = None
    valid_mask: Optional[Array] = None
    alignment_metadata: Optional[Dict[str, Any]] = None

    reference_normalized: Optional[Array] = None
    inspected_normalized: Optional[Array] = None
    normalization_metadata: Optional[Dict[str, Any]] = None
    normalization_debug: Dict[str, Any] = field(default_factory=dict)

    anomaly_map: Optional[Array] = None
    ssim_map: Optional[Array] = None
    comparison_metadata: Dict[str, Any] = field(default_factory=dict)
    # Populated when comparison is artifact_residual and debug_save_intermediates is True:
    # dict of numpy arrays (residuals, enhanced maps, edge mask, etc.), not merged into comparison_metadata.
    artifact_residual_intermediates: Optional[Dict[str, Any]] = None
    threshold_map: Optional[Array] = None
    thresholding_metadata: Dict[str, Any] = field(default_factory=dict)
    binary_mask_raw: Optional[Array] = None
    binary_mask_final: Optional[Array] = None

    decision_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    pair_id: str
    defect_mask: Array
    artifacts: PipelineArtifacts

