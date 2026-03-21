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
from modules.alignment.ecc_alignment import EccEuclideanAligner, EccTranslationAligner


@unittest.skipIf(cv2 is None, "OpenCV is required for ECC tests")
class EccAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = PipelineConfig().ecc_translation_alignment

    def _base_image(self, h: int = 128, w: int = 128) -> np.ndarray:
        yy, xx = np.indices((h, w), dtype=np.float32)
        base = (
            0.3 * np.sin(xx / 6.0)
            + 0.2 * np.cos(yy / 9.0)
            + ((xx % 16) < 8).astype(np.float32) * 0.1
        )
        return base.astype(np.float32)

    def test_ecc_translation_identical_near_identity(self) -> None:
        aligner = EccTranslationAligner()
        image = self._base_image()
        aligned_ref, ins_out, meta = aligner.run(image, image.copy(), self.cfg)

        self.assertEqual(aligned_ref.shape, image.shape)
        self.assertEqual(ins_out.shape, image.shape)
        self.assertTrue(meta["warp_matrix"].shape == (2, 3))
        self.assertAlmostEqual(float(meta["translation_x"]), 0.0, places=2)
        self.assertAlmostEqual(float(meta["translation_y"]), 0.0, places=2)

    def test_ecc_translation_recovers_simple_shift(self) -> None:
        aligner = EccTranslationAligner()
        ref = np.zeros((128, 128), dtype=np.float32)
        cv2.rectangle(ref, (20, 20), (80, 60), 0.8, thickness=-1)
        cv2.circle(ref, (95, 95), 16, 0.5, thickness=-1)
        # Convention for this test:
        # - `ins` is the inspected/template target we want to align to.
        # - ECC receives (reference=ref, inspected=ins) and returns aligned_ref
        #   in inspected coordinates.
        tx, ty = 5.0, -3.0
        warp = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
        ins = cv2.warpAffine(ref, warp, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cfg = PipelineConfig().ecc_translation_alignment
        cfg.params["allow_fallback_to_identity"] = False
        cfg.params["number_of_iterations"] = 500
        before = float(np.mean(np.abs(ins - ref)))
        aligned_ref, _, meta = aligner.run(ref, ins, cfg)
        after = float(np.mean(np.abs(ins - aligned_ref)))

        self.assertEqual(aligned_ref.shape, ref.shape)
        self.assertLess(after, before * 0.2)  # should dramatically reduce mismatch
        self.assertAlmostEqual(float(meta["translation_x"]), tx, places=1)
        self.assertAlmostEqual(float(meta["translation_y"]), ty, places=1)
        mask = meta.get("valid_mask")
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, ref.shape)
        self.assertEqual(mask.dtype, np.bool_)

    def test_ecc_euclidean_runs_on_rotated_image(self) -> None:
        aligner = EccEuclideanAligner()
        cfg = PipelineConfig().ecc_euclidean_alignment
        ref = self._base_image()
        center = (ref.shape[1] / 2.0, ref.shape[0] / 2.0)
        m = cv2.getRotationMatrix2D(center, 2.5, 1.0).astype(np.float32)
        ins = cv2.warpAffine(ref, m, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        aligned_ref, ins_out, meta = aligner.run(ref, ins, cfg)
        self.assertEqual(aligned_ref.shape, ref.shape)
        self.assertEqual(ins_out.shape, ref.shape)
        self.assertIn("rotation_degrees_estimated", meta)

    def test_valid_mask_is_present_with_correct_shape(self) -> None:
        aligner = EccTranslationAligner()
        ref = self._base_image()
        ins = ref.copy()
        _, _, meta = aligner.run(ref, ins, self.cfg)
        mask = meta.get("valid_mask")
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, ref.shape)
        self.assertEqual(mask.dtype, np.bool_)


if __name__ == "__main__":
    unittest.main()
