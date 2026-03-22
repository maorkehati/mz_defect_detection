"""
Single diagnostic run: reference–inspected z asymmetry gate + contrast_area_log + top_k=3.

Same CSV / summary format as ``run_sign_dominance_probe.py`` / ``run_ranking_probe._run_probe``.

Example:
  python scripts/run_asymmetry_probe.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import build_search_euclidean_artifact_residual_mad_config
from data import load_sample_pairs
from pipeline import DefectDetectionPipeline
from run_ranking_probe import _run_probe

DEFAULT_ROOT_PATTERN = str(REPO_ROOT.parent / "*")
INSPECTED_PATTERN = "case*_inspected_image.tif"
REFERENCE_PATTERN = "case*_reference_image.tif"

OUT_ROOT = REPO_ROOT / "outs" / "diagnostics" / "ranking_probe" / "asymmetry_ref_ins_k3"

SINGLE_RUN: Dict[str, Any] = {
    "run_name": "asymmetry_ref_ins_k3",
    "ranking_mode": "contrast_area_log",
    "top_k_keep": 3,
    "min_area": 0.0,
    "min_contour_score": 0.0,
    "k_mad": 4.0,
    "min_sign_dominance": None,
    "min_asymmetry": 2.0,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Asymmetry probe (z_ref vs max(z_pos,z_neg)).")
    ap.add_argument("--root-pattern", type=str, default=DEFAULT_ROOT_PATTERN)
    ap.add_argument("--inspected-pattern", type=str, default=INSPECTED_PATTERN)
    ap.add_argument("--reference-pattern", type=str, default=REFERENCE_PATTERN)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--out", type=str, default=str(OUT_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    samples = load_sample_pairs(
        root_pattern=args.root_pattern,
        inspected_pattern=args.inspected_pattern,
        reference_pattern=args.reference_pattern,
        recursive=args.recursive,
        sort_results=True,
    )
    samples = [s for s in samples if re.search(r"case\s*[123]", s.pair_id, re.I)] or samples
    if not samples:
        print("No samples loaded; check --root-pattern.", file=sys.stderr)
        sys.exit(1)

    base_cfg = build_search_euclidean_artifact_residual_mad_config()
    base_cfg.contour_filter_postprocess.params["min_area"] = 0.0
    base_cfg.contour_filter_postprocess.params["min_contour_score"] = 0.0
    base_cfg.contour_filter_postprocess.params["max_aspect_ratio"] = None
    base_cfg.contour_filter_postprocess.params["min_fill_ratio"] = None
    base_cfg.contour_filter_postprocess.params["exclude_border_touching"] = False
    base_cfg.contour_filter_postprocess.params["min_sign_dominance"] = None

    pipeline = DefectDetectionPipeline(base_cfg)
    cached: Dict[str, Any] = {}
    for sp in samples:
        up, _ = pipeline.run_through_normalization(sp, silent=True)
        cached[sp.pair_id] = up

    run_dir = out_root / str(SINGLE_RUN["run_name"])
    print(f"Loaded {len(samples)} samples. Output: {run_dir}")
    print("Comparator: artifact_residual (z_ins max vs z_ref); post: min_asymmetry + rank")
    _run_probe(
        params=SINGLE_RUN,
        run_dir=run_dir,
        base_cfg=base_cfg,
        cached=cached,
        samples=samples,
    )


if __name__ == "__main__":
    main()
