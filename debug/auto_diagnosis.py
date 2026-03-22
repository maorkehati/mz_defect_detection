from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from debug.report_types import GTFateRecord


def build_auto_diagnosis(
    gt_fates: List[GTFateRecord],
    final_count: int,
    gt_expected: int,
    *,
    top_peaks: List[Dict[str, Any]] | None = None,
    candidates: List[Any] | None = None,
) -> str:
    """Rule-based, deterministic summary from computed diagnostics."""
    top_peaks = top_peaks or []
    candidates = candidates or []

    def _fmt(mode: str, facts: List[str], confidence: str) -> str:
        fact_txt = "; ".join(facts[:3]) if facts else "no supporting facts"
        return f"dominant_failure_mode={mode}; facts={fact_txt}; confidence={confidence}"

    if gt_expected == 0:
        if final_count == 0:
            return _fmt("CLEAN_NEGATIVE", ["no GT expected", "no final detections"], "high")
        border_fp = sum(1 for p in top_peaks if p.get("nearest_gt_id") is None and bool(p.get("near_border", False)))
        border_dom = bool(top_peaks) and (border_fp / max(1, len(top_peaks))) >= 0.5
        mode = "BORDER_ARTIFACT_FALSE_POSITIVES" if border_dom else "FALSE_POSITIVES"
        facts = [f"final_detections={final_count}", f"border_non_gt_top_peaks={border_fp}/{len(top_peaks)}"]
        return _fmt(mode, facts, "medium")

    misses = [g for g in gt_fates if not g.kept_final]
    if not misses:
        return _fmt("SUCCESS", [f"gt_hits={gt_expected}/{gt_expected}", f"final_detections={final_count}"], "high")

    # 1) Morphology too destructive
    if any(bool(g.threshold_support_r5) and not bool(g.survived_morph) for g in misses):
        facts = [
            f"misses_after_threshold={sum(1 for g in misses if g.threshold_support_r5)}",
            f"misses_after_morph={sum(1 for g in misses if not g.survived_morph)}",
        ]
        return _fmt("MORPHOLOGY_TOO_DESTRUCTIVE", facts, "high")

    # 2) Sign consistency hard filter
    if any(str(g.failure_stage) == "FILTER_SIGN_CONSISTENCY" for g in misses):
        n = sum(1 for g in misses if str(g.failure_stage) == "FILTER_SIGN_CONSISTENCY")
        facts = [f"gt_misses_sign_filter={n}/{len(misses)}", "threshold support exists before filtering"]
        return _fmt("SIGN_CONSISTENCY_FILTER_AGGRESSIVE", facts, "high" if n == len(misses) else "medium")

    # 3) Ranked out by top-k
    if any(str(g.failure_stage) == "RANKED_OUT_TOPK" for g in misses):
        n = sum(1 for g in misses if str(g.failure_stage) == "RANKED_OUT_TOPK")
        facts = [f"gt_ranked_out={n}/{len(misses)}", "candidate formed but dropped by ranking/top-k"]
        return _fmt("RANKING_TOPK_ISSUE", facts, "medium")

    # 4) Comparator weak
    weak = [g for g in misses if g.anomaly_percentile is not None and float(g.anomaly_percentile) < 70.0]
    if weak and len(weak) >= max(1, len(misses) // 2):
        facts = [f"weak_gt_percentiles={len(weak)}/{len(misses)}", "weak signal before threshold"]
        return _fmt("COMPARATOR_WEAK_SIGNAL", facts, "medium")

    # 5) Alignment weak near GT
    align_weak = [g for g in misses if str(g.failure_stage) == "ALIGNMENT_WEAK_SIGNAL"]
    if align_weak:
        facts = [f"alignment_weak_gt={len(align_weak)}/{len(misses)}", "local alignment residual elevated near missed GT"]
        return _fmt("ALIGNMENT_RESIDUAL_PROBLEM", facts, "medium")

    # 6) Border-dominated clutter among strong top peaks
    border_fp = sum(1 for p in top_peaks if p.get("nearest_gt_id") is None and bool(p.get("near_border", False)))
    if top_peaks and (border_fp / max(1, len(top_peaks))) >= 0.5:
        facts = [f"border_non_gt_top_peaks={border_fp}/{len(top_peaks)}", "top responses concentrated near border"]
        return _fmt("BORDER_DOMINATED_COMPARATOR_ARTIFACTS", facts, "medium")

    cnt = Counter(m.failure_stage for m in misses)
    stage, n = cnt.most_common(1)[0]
    conf = "high" if n >= max(1, len(misses) - 1) else "low"
    return _fmt(stage, [f"stage_count={n}/{len(misses)}"], conf)

