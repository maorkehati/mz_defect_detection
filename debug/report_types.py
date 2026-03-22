from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


FAILURE_STAGE_LABELS = (
    "OUTSIDE_VALID_OVERLAP",
    "ALIGNMENT_WEAK_SIGNAL",
    "COMPARATOR_WEAK_SIGNAL",
    "THRESHOLD",
    "MORPHOLOGY",
    "NO_CONTOUR_FORMED",
    "FILTER_AREA",
    "FILTER_ASPECT_RATIO",
    "FILTER_FILL_RATIO",
    "FILTER_BORDER",
    "FILTER_SIGN_CONSISTENCY",
    "RANKED_OUT_TOPK",
    "FINAL_MASK_MISMATCH",
    "UNKNOWN",
)


@dataclass
class GTFateRecord:
    gt_id: int
    x: int
    y: int
    inside_valid_overlap: bool
    distance_to_invalid_border_px: Optional[float]
    anomaly_at_gt: Optional[float]
    anomaly_local_max_r5: Optional[float]
    anomaly_percentile: Optional[float]
    above_threshold: Optional[bool]
    threshold_support_r5: Optional[bool]
    threshold_component_id: Optional[int]
    threshold_component_area: Optional[int]
    survived_morph: Optional[bool]
    morph_component_id: Optional[int]
    candidate_id: Optional[int]
    contour_contains_gt: Optional[bool]
    contour_centroid_distance_to_gt_px: Optional[float]
    candidate_score: Optional[float]
    candidate_rank: Optional[int]
    kept_final: bool
    final_component_id: Optional[int]
    failure_stage: str
    rejection_reason: str
    short_details: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateAuditRecord:
    candidate_id: int
    centroid_x: float
    centroid_y: float
    area: float
    bbox_x: Optional[int]
    bbox_y: Optional[int]
    bbox_w: Optional[int]
    bbox_h: Optional[int]
    aspect_ratio: Optional[float]
    fill_ratio: Optional[float]
    border_touching: Optional[bool]
    border_distance: Optional[float]
    nearest_gt_id: Optional[int]
    nearest_gt_distance: Optional[float]
    gt_match: bool
    mean_anomaly: Optional[float]
    p95_anomaly: Optional[float]
    ring_mean: Optional[float]
    local_contrast: Optional[float]
    sign_consistency: Optional[float]
    dominant_sign: Optional[str]
    score: Optional[float]
    rank: Optional[int]
    pass_area: Optional[bool]
    pass_aspect_ratio: Optional[bool]
    pass_fill_ratio: Optional[bool]
    pass_border: Optional[bool]
    pass_sign_consistency: Optional[bool]
    kept_final: bool
    rejection_reason: str
    edge_overlap_fraction: Optional[float] = None
    border_overlap_fraction: Optional[float] = None
    likely_origin: Optional[str] = None

