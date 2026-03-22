from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


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
    mean_anomaly: Optional[float]
    p95_anomaly: Optional[float]
    ring_mean: Optional[float]
    local_contrast: Optional[float]
    sign_consistency: Optional[float]
    dominant_sign: Optional[str]
    score: Optional[float]
    rank: Optional[int]
    kept_final: bool
    rejection_reason: str
    pass_area: Optional[bool]
    pass_aspect_ratio: Optional[bool]
    pass_fill_ratio: Optional[bool]
    pass_border: Optional[bool]
    pass_sign_consistency: Optional[bool]
    mean_z_pos: Optional[float] = None
    mean_z_neg: Optional[float] = None
    sign_dominance: Optional[float] = None
    z_dominant_sign: Optional[str] = None
    mean_z_reference: Optional[float] = None
    asymmetry: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

