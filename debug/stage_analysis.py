from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def build_stage_summary_rows(artifacts: Any, decision_metadata: Dict[str, Any]) -> List[Tuple[str, str, float | int | str]]:
    rows: List[Tuple[str, str, float | int | str]] = []
    vm = np.asarray(artifacts.valid_mask).astype(bool) if artifacts.valid_mask is not None else None
    if vm is not None:
        rows.append(("ALIGNMENT", "valid_overlap_fraction", float(np.mean(vm.astype(np.float32)))))
        core = cv2.erode(vm.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
        rows.append(("ALIGNMENT", "eroded_core_fraction", float(np.mean(core.astype(np.float32)))))
    am = artifacts.alignment_metadata or {}
    for k in ("final_theta_deg", "final_tx", "final_ty", "best_score", "second_best_score", "score_margin"):
        if k in am and am.get(k) is not None:
            rows.append(("ALIGNMENT", k, am.get(k)))
    nm = artifacts.normalization_metadata or {}
    nd = artifacts.normalization_debug or {}
    for k in ("fit_pixel_count", "gain", "offset"):
        if k in nm and nm.get(k) is not None:
            rows.append(("NORMALIZATION", k, nm.get(k)))
    for k in ("before_mean", "after_mean", "before_median", "after_median", "status"):
        if k in nd and nd.get(k) is not None:
            rows.append(("NORMALIZATION", k, nd.get(k)))
    cm = artifacts.comparison_metadata or {}
    for k in ("mean", "std", "p50", "p90", "p95", "p99", "strong_edge_fraction"):
        if k in cm and cm.get(k) is not None:
            rows.append(("COMPARISON", k, cm.get(k)))
    tm = artifacts.thresholding_metadata or {}
    for k in ("threshold_value", "positive_fraction", "positive_pixel_count", "k_mad", "method"):
        if k in tm and tm.get(k) is not None:
            rows.append(("THRESHOLD", k, tm.get(k)))
    raw = np.asarray(artifacts.binary_mask_raw).astype(bool) if artifacts.binary_mask_raw is not None else None
    morph = decision_metadata.get("mask_after_morph")
    morph_mask = np.asarray(morph).astype(bool) if morph is not None else None
    if raw is not None:
        rows.append(("THRESHOLD", "component_count", int(cv2.connectedComponents(raw.astype(np.uint8))[0] - 1)))
    if morph_mask is not None:
        rows.append(("MORPHOLOGY", "post_morph_positive_fraction_valid", float(np.mean(morph_mask.astype(np.float32)))))
        rows.append(("MORPHOLOGY", "post_morph_component_count", int(cv2.connectedComponents(morph_mask.astype(np.uint8))[0] - 1)))
    for k in (
        "num_contours_total",
        "num_contours_after_geom_filters",
        "num_contours_after_score_threshold",
        "num_contours_after_topk",
        "final_num_contours",
    ):
        if k in decision_metadata and decision_metadata.get(k) is not None:
            rows.append(("POSTPROCESS", k, decision_metadata.get(k)))
    return rows

