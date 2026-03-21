from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - environment-dependent
    cv2 = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import PipelineConfig
from modules.alignment.search_euclidean import SearchEuclideanAligner


@unittest.skipIf(cv2 is None, "OpenCV is required for ECC tests")
class SearchEuclideanAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.aligner = SearchEuclideanAligner()
        self.cfg = PipelineConfig().search_euclidean_alignment

    def _base_image(self, h: int = 128, w: int = 128) -> np.ndarray:
        img = np.zeros((h, w), dtype=np.float32)
        cv2.rectangle(img, (20, 25), (92, 70), 0.8, thickness=-1)
        cv2.circle(img, (40, 95), 14, 0.4, thickness=-1)  # off-center for asymmetry
        cv2.line(img, (70, 10), (110, 35), 0.6, thickness=3)
        return img.astype(np.float32)

    def _apply_warp(self, ref: np.ndarray, theta_deg: float, tx: float, ty: float) -> np.ndarray:
        h, w = ref.shape
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, float(theta_deg), 1.0).astype(np.float32)
        M[0, 2] += float(tx)
        M[1, 2] += float(ty)
        return cv2.warpAffine(ref, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)

    def test_search_euclidean_same_shape_and_valid_mask(self) -> None:
        ref = self._base_image()
        theta, tx, ty = 3.0, 7.0, -4.0
        ins = self._apply_warp(ref, theta_deg=theta, tx=tx, ty=ty)

        aligned_ref, ins_out, meta = self.aligner.run(ref, ins, self.cfg)
        self.assertEqual(aligned_ref.shape, ref.shape)
        self.assertEqual(ins_out.shape, ins.shape)

        valid_mask = meta.get("valid_mask")
        self.assertIsNotNone(valid_mask)
        self.assertEqual(np.asarray(valid_mask).shape, ref.shape)
        self.assertEqual(np.asarray(valid_mask).dtype, np.bool_)

    def test_search_euclidean_improves_residual(self) -> None:
        ref = self._base_image()
        theta, tx, ty = 2.5, 7.0, -4.0
        ins = self._apply_warp(ref, theta_deg=theta, tx=tx, ty=ty)

        before = float(np.mean(np.abs(ins - ref)))
        aligned_ref, _, meta = self.aligner.run(ref, ins, self.cfg)
        after = float(np.mean(np.abs(ins - aligned_ref)))

        # Should noticeably reduce mismatch; exact ratio depends on interpolation + valid overlap.
        self.assertLess(after, before * 0.6)
        self.assertEqual(meta.get("coarse_candidates_total"), 17)
        self.assertEqual(meta.get("refined_candidates_total"), 15)
        self.assertEqual(len(meta.get("coarse_candidate_records", [])), 17)
        self.assertEqual(len(meta.get("refined_candidate_records", [])), 15)

    def test_search_euclidean_identity_on_identical_images(self) -> None:
        ref = self._base_image()
        aligned_ref, _, meta = self.aligner.run(ref, ref.copy(), self.cfg)
        after = float(np.mean(np.abs(ref - aligned_ref)))
        self.assertLess(after, 0.05)
        self.assertTrue(np.isfinite(float(meta.get("best_theta_deg", 0.0))))


if __name__ == "__main__":
    unittest.main()

