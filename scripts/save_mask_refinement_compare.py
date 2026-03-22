"""
Save final defect masks with refinement on vs off for the three exercise pairs.

Writes ``outs/mask_refinement_compare/<case_key>_{refined,unrefined}.npy`` for visual diff.

    python scripts/save_mask_refinement_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from config import build_search_euclidean_artifact_residual_mad_config
from data import load_sample_pairs
from pipeline import DefectDetectionPipeline
from utils.ground_truth_defects import pair_id_to_case_key


def main() -> None:
    out_dir = _REPO / "outs" / "mask_refinement_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        s
        for s in load_sample_pairs(root_pattern=str(_REPO.parent / "*"), sort_results=True)
        if pair_id_to_case_key(s.pair_id) in ("case1", "case2", "case3")
    ]

    for refined in (True, False):
        cfg = build_search_euclidean_artifact_residual_mad_config()
        cfg.peak_nms_postprocess.refine_component_support = refined
        cfg.peak_nms_postprocess.refine_mode = "hysteresis_local" if refined else "none"
        cfg.debug.enable_debug_visualization = False
        cfg.debug.save_debug_images = False
        p = DefectDetectionPipeline(cfg)
        for s in samples:
            ck = pair_id_to_case_key(s.pair_id)
            r = p.run(s)
            m = r.defect_mask.astype(np.uint8)
            tag = "refined" if refined else "unrefined"
            np.save(out_dir / f"{ck}_{tag}.npy", m)
            print(ck, tag, int(m.sum()), "px")


if __name__ == "__main__":
    main()
