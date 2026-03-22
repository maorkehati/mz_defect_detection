"""
Compare defect masks against golden ``outs/threshold_investigation/golden/*_defect_mask.npy``.

Run after changing peak accept thresholding::

    python scripts/verify_peak_threshold_equivalence.py

Golden files are produced once from the override-based baseline (see ``investigate_peak_accept_threshold``).
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


def main() -> int:
    repo = _REPO
    golden_dir = repo / "outs" / "threshold_investigation" / "golden"
    if not golden_dir.is_dir():
        print("Missing golden dir:", golden_dir, file=sys.stderr)
        return 2

    cfg = build_search_euclidean_artifact_residual_mad_config()
    cfg.debug.enable_debug_visualization = False
    cfg.debug.save_debug_images = False
    p = DefectDetectionPipeline(cfg)
    samples = [
        s
        for s in load_sample_pairs(root_pattern=str(repo.parent / "*"), sort_results=True)
        if pair_id_to_case_key(s.pair_id) in ("case1", "case2", "case3")
    ]
    cached = {}
    for s in samples:
        a, _ = p.run_through_normalization(s, silent=True)
        cached[s.pair_id] = a

    ok = True
    for s in samples:
        ck = pair_id_to_case_key(s.pair_id)
        gpath = golden_dir / f"{ck}_defect_mask.npy"
        if not gpath.is_file():
            print("Missing golden:", gpath, file=sys.stderr)
            ok = False
            continue
        gold = np.load(gpath)
        r = p.run_from_normalized(s, cached[s.pair_id], silent=True)
        cur = r.defect_mask.astype(np.uint8)
        if gold.shape != cur.shape:
            print(ck, "SHAPE mismatch", gold.shape, cur.shape)
            ok = False
            continue
        if not np.array_equal(gold.astype(np.uint8), cur):
            diff = int(np.count_nonzero(gold.astype(bool) ^ cur.astype(bool)))
            print(ck, "MASK mismatch differing pixels:", diff)
            ok = False
        else:
            print(ck, "OK bit-identical mask", int(cur.sum()), "px")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
