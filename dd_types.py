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
    reference_input: Optional[Array] = None
    inspected_input: Optional[Array] = None

    reference_preprocessed: Optional[Array] = None
    inspected_preprocessed: Optional[Array] = None

    reference_aligned: Optional[Array] = None
    inspected_aligned: Optional[Array] = None
    alignment_metadata: Optional[Dict[str, Any]] = None

    reference_normalized: Optional[Array] = None
    inspected_normalized: Optional[Array] = None
    normalization_metadata: Optional[Dict[str, Any]] = None

    anomaly_map: Optional[Array] = None
    threshold_map: Optional[Array] = None
    binary_mask_raw: Optional[Array] = None
    binary_mask_final: Optional[Array] = None

    decision_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    pair_id: str
    defect_mask: Array
    artifacts: PipelineArtifacts

