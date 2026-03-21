from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from modules.base import AlignerBase


def _array_stats(x: np.ndarray) -> Dict[str, float | str]:
    arr = np.asarray(x)
    arr_f = np.asarray(arr, dtype=np.float32)
    return {
        "dtype": str(arr.dtype),
        "min": float(np.min(arr_f)),
        "max": float(np.max(arr_f)),
        "mean": float(np.mean(arr_f)),
        "std": float(np.std(arr_f)),
        "dynamic_range": float(np.max(arr_f) - np.min(arr_f)),
    }


def _minmax_to_unit(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _prepare_ecc_images(
    reference: np.ndarray,
    inspected: np.ndarray,
    use_gradient: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    ref = _minmax_to_unit(reference)
    ins = _minmax_to_unit(inspected)
    if not use_gradient:
        return ref, ins

    # Optional gradient-domain ECC can help under intensity drift.
    ref_gx = cv2.Sobel(ref, cv2.CV_32F, 1, 0, ksize=3)
    ref_gy = cv2.Sobel(ref, cv2.CV_32F, 0, 1, ksize=3)
    ins_gx = cv2.Sobel(ins, cv2.CV_32F, 1, 0, ksize=3)
    ins_gy = cv2.Sobel(ins, cv2.CV_32F, 0, 1, ksize=3)
    ref_mag = np.sqrt(ref_gx * ref_gx + ref_gy * ref_gy, dtype=np.float32)
    ins_mag = np.sqrt(ins_gx * ins_gx + ins_gy * ins_gy, dtype=np.float32)
    return _minmax_to_unit(ref_mag), _minmax_to_unit(ins_mag)


def _build_initial_warp(warp_mode: int) -> np.ndarray:
    if warp_mode in (cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE):
        return np.array(
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
    raise ValueError(f"Unsupported warp_mode: {warp_mode}")


def _run_ecc(
    template_image: np.ndarray,
    input_image: np.ndarray,
    warp_matrix: np.ndarray,
    warp_mode: int,
    number_of_iterations: int,
    termination_eps: float,
    gaussian_filter_size: int,
) -> Tuple[float, np.ndarray]:
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(number_of_iterations),
        float(termination_eps),
    )

    gfs = int(gaussian_filter_size)
    if gfs <= 0:
        gfs = 1
    if gfs % 2 == 0:
        gfs += 1

    # OpenCV ECC convention (explicit mapping):
    # - template_image is the target (destination) coordinate frame.
    # - input_image is the source (to-be-warped) coordinate frame.
    # - The returned warpMatrix maps *input_image coordinates* -> *template_image coordinates*
    #   (i.e., it describes how to align `input_image` onto `template_image`).
    # Some OpenCV builds require positional arguments here.
    try:
        cc, wm = cv2.findTransformECC(
            template_image,
            input_image,
            warp_matrix,
            warp_mode,
            criteria,
            None,
            gfs,
        )
        return float(cc), np.asarray(wm, dtype=np.float32)
    except Exception:
        cc, wm = cv2.findTransformECC(
            template_image,
            input_image,
            warp_matrix,
            warp_mode,
            criteria,
            None,
        )
        return float(cc), np.asarray(wm, dtype=np.float32)


def _warp_image(reference: np.ndarray, warp_matrix: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    h, w = output_shape
    # warp_matrix is estimated to align `reference` onto the inspected/template frame.
    # `warpAffine` by default treats the matrix as an inverse mapping (output->input).
    # We use `cv2.WARP_INVERSE_MAP` so warp_matrix is interpreted in the same
    # input->output (source->destination) sense as returned by findTransformECC.
    return cv2.warpAffine(
        np.asarray(reference, dtype=np.float32),
        np.asarray(warp_matrix, dtype=np.float32),
        (int(w), int(h)),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32, copy=False)


def _compute_valid_mask(reference_shape: tuple[int, int], warp_matrix: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    h_ref, w_ref = reference_shape
    ones = np.ones((h_ref, w_ref), dtype=np.float32)
    # Valid overlap mask is computed using the EXACT same warp convention
    # as the aligned-reference warp: warp the reference-domain "ones" into
    # the inspected/template coordinate frame using the same warpAffine flags.
    warped = cv2.warpAffine(
        ones,
        np.asarray(warp_matrix, dtype=np.float32),
        (int(output_shape[1]), int(output_shape[0])),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.asarray(warped > 0.5, dtype=bool)


def _rotation_degrees_from_euclidean_warp(warp_matrix: np.ndarray) -> float:
    a = float(warp_matrix[0, 0])
    b = float(warp_matrix[0, 1])
    return float(np.degrees(np.arctan2(b, a)))


def _extract_affine_params(warp_matrix: np.ndarray) -> Dict[str, float]:
    wm = np.asarray(warp_matrix, dtype=np.float32)
    a, b, txm = float(wm[0, 0]), float(wm[0, 1]), float(wm[0, 2])
    c, d, tym = float(wm[1, 0]), float(wm[1, 1]), float(wm[1, 2])
    scale_x = float(np.sqrt(max(1e-12, a * a + b * b)))
    scale_y = float(np.sqrt(max(1e-12, c * c + d * d)))
    rotation_deg = float(np.degrees(np.arctan2(b, a)))
    shear_proxy = float((a * c + b * d) / max(1e-12, scale_x * scale_y))
    return {
        "matrix_tx": txm,
        "matrix_ty": tym,
        "rotation_deg_from_matrix": rotation_deg,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "shear_proxy": shear_proxy,
    }


def _compute_ecc_similarity(template_image: np.ndarray, input_image: np.ndarray) -> float | None:
    try:
        cc = cv2.computeECC(
            np.asarray(template_image, dtype=np.float32),
            np.asarray(input_image, dtype=np.float32),
            None,
        )
        return float(cc)
    except Exception:
        return None


def _project_affine_linear_to_rotation(affine_warp: np.ndarray) -> np.ndarray:
    wm = np.asarray(affine_warp, dtype=np.float32)
    A = wm[:, :2].astype(np.float64)
    U, _s, Vt = np.linalg.svd(A, full_matrices=False)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return np.asarray(R, dtype=np.float32)


def _build_warp_from_rotation_translation(
    R: np.ndarray,
    tx: float,
    ty: float,
) -> np.ndarray:
    out = np.zeros((2, 3), dtype=np.float32)
    out[:, :2] = np.asarray(R, dtype=np.float32)
    out[0, 2] = float(tx)
    out[1, 2] = float(ty)
    return out


def _masked_abs_score(reference: np.ndarray, inspected: np.ndarray, valid_mask: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float32)
    ins = np.asarray(inspected, dtype=np.float32)
    vm = np.asarray(valid_mask).astype(bool)
    if vm.shape != ref.shape or not np.any(vm):
        return float("inf")
    return float(np.mean(np.abs(ref[vm] - ins[vm])))


class _BaseEccAligner(AlignerBase):
    def __init__(self, name: str, warp_mode: int) -> None:
        super().__init__(name=name)
        self._warp_mode = int(warp_mode)

    def run(self, reference_image, inspected_image, cfg):
        self.validate_config(cfg)
        self.validate_array(reference_image, "reference_image")
        self.validate_array(inspected_image, "inspected_image")
        self.validate_same_shape(
            reference_image, inspected_image, "reference_image", "inspected_image"
        )

        ref = np.asarray(reference_image, dtype=np.float32)
        ins = np.asarray(inspected_image, dtype=np.float32)
        if ref.ndim != 2 or ins.ndim != 2:
            raise ValueError(f"{self.name} requires single-channel 2D images.")

        n_iter = int(self.get_param(cfg, "number_of_iterations", 200))
        eps = float(self.get_param(cfg, "termination_eps", 1e-6))
        gfs = int(self.get_param(cfg, "gaussian_filter_size", 5))
        allow_fallback = bool(self.get_param(cfg, "allow_fallback_to_identity", True))
        use_gradient = bool(self.get_param(cfg, "use_gradient_images", False))

        ref_ecc, ins_ecc = _prepare_ecc_images(ref, ins, use_gradient=use_gradient)
        warp = _build_initial_warp(self._warp_mode)
        initial_warp = np.asarray(warp, dtype=np.float32).copy()
        initial_corr = _compute_ecc_similarity(ins_ecc, ref_ecc)
        input_stats = {
            "reference": _array_stats(ref),
            "inspected": _array_stats(ins),
            "reference_ecc": _array_stats(ref_ecc),
            "inspected_ecc": _array_stats(ins_ecc),
        }

        fallback_used = False
        ecc_converged = True
        corr = None
        try:
            corr, warp = _run_ecc(
                template_image=ins_ecc,
                input_image=ref_ecc,
                warp_matrix=warp,
                warp_mode=self._warp_mode,
                number_of_iterations=n_iter,
                termination_eps=eps,
                gaussian_filter_size=gfs,
            )
            aligned_ref = _warp_image(ref, warp, output_shape=ins.shape)
            valid_mask = _compute_valid_mask(ref.shape, warp, output_shape=ins.shape)
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(f"{self.name} failed to converge: {exc}") from exc
            fallback_used = True
            ecc_converged = False
            warp = _build_initial_warp(self._warp_mode)
            aligned_ref = ref.astype(np.float32, copy=False)
            valid_mask = np.ones_like(ins, dtype=bool)

        # Reported translation_x/translation_y are the effective shift that aligns
        # the *reference* onto the *inspected/template* frame.
        # Because warpAffine is called with WARP_INVERSE_MAP, the translation terms
        # in warpMatrix have the opposite sign relative to that "reference->inspected"
        # shift convention, so we negate them for reporting.
        tx = float(-warp[0, 2])
        ty = float(-warp[1, 2])
        metadata: Dict[str, Any] = {
            "method": self.name,
            "warp_mode": int(self._warp_mode),
            "motion_model_name": (
                "translation"
                if self._warp_mode == cv2.MOTION_TRANSLATION
                else "euclidean"
                if self._warp_mode == cv2.MOTION_EUCLIDEAN
                else "affine"
                if self._warp_mode == cv2.MOTION_AFFINE
                else "unknown"
            ),
            "requested_number_of_iterations": int(n_iter),
            "requested_termination_eps": float(eps),
            "gaussian_filter_size_used": int(gfs),
            "use_gradient_images": bool(use_gradient),
            "allow_fallback_to_identity": bool(allow_fallback),
            "initial_warp_matrix": initial_warp,
            "warp_matrix": np.asarray(warp, dtype=np.float32),
            "ecc_initial_correlation": None if initial_corr is None else float(initial_corr),
            "ecc_correlation": None if corr is None else float(corr),
            "ecc_converged": bool(ecc_converged),
            "ecc_iteration_count_exposed": None,
            "fallback_used": bool(fallback_used),
            "translation_x": tx,
            "translation_y": ty,
            "shift_x": tx,
            "shift_y": ty,
            "valid_mask": valid_mask.astype(bool),
            "valid_pixel_fraction": float(np.mean(valid_mask.astype(np.float32))),
            "ecc_input_stats": input_stats,
        }
        if self._warp_mode == cv2.MOTION_EUCLIDEAN:
            metadata["rotation_degrees_estimated"] = _rotation_degrees_from_euclidean_warp(warp)
        if self._warp_mode == cv2.MOTION_AFFINE:
            metadata["affine_params_initial"] = _extract_affine_params(initial_warp)
            metadata["affine_params_final"] = _extract_affine_params(warp)
        metadata["warp_convention_note"] = (
            "findTransformECC(template=inspected, input=reference) estimates warpMatrix "
            "to align reference->inspected. warpAffine uses WARP_INVERSE_MAP "
            "so the warpMatrix is treated as source->destination."
        )

        return aligned_ref, ins.astype(np.float32, copy=False), metadata


class EccTranslationAligner(_BaseEccAligner):
    def __init__(self) -> None:
        super().__init__(name="ecc_translation", warp_mode=cv2.MOTION_TRANSLATION)


class EccEuclideanAligner(_BaseEccAligner):
    def __init__(self) -> None:
        super().__init__(name="ecc_euclidean", warp_mode=cv2.MOTION_EUCLIDEAN)


class EccAffineAligner(_BaseEccAligner):
    def __init__(self) -> None:
        super().__init__(name="ecc_affine", warp_mode=cv2.MOTION_AFFINE)


class EccAffineProjectedEuclideanAligner(_BaseEccAligner):
    def __init__(self) -> None:
        super().__init__(name="ecc_affine_projected_euclidean", warp_mode=cv2.MOTION_AFFINE)

    def run(self, reference_image, inspected_image, cfg):
        # Step 1: run raw affine ECC exactly as implemented today.
        aligned_affine, inspected_out, affine_meta = super().run(reference_image, inspected_image, cfg)

        ref = np.asarray(reference_image, dtype=np.float32)
        ins = np.asarray(inspected_image, dtype=np.float32)
        affine_warp = np.asarray(affine_meta.get("warp_matrix"), dtype=np.float32)
        if affine_warp.shape != (2, 3):
            return aligned_affine, inspected_out, affine_meta

        # Step 2: project affine linear part to closest proper rotation.
        R = _project_affine_linear_to_rotation(affine_warp)
        tx0 = float(affine_warp[0, 2])
        ty0 = float(affine_warp[1, 2])

        # Stage-1 / Stage-2 explicit local translation refinement.
        r1x = float(self.get_param(cfg, "translation_refine_radius_x_px_stage1", 2.0))
        r1y = float(self.get_param(cfg, "translation_refine_radius_y_px_stage1", 2.0))
        s1x = float(self.get_param(cfg, "translation_refine_step_x_px_stage1", 0.5))
        s1y = float(self.get_param(cfg, "translation_refine_step_y_px_stage1", 0.5))
        r2x = float(self.get_param(cfg, "translation_refine_radius_x_px_stage2", 0.5))
        r2y = float(self.get_param(cfg, "translation_refine_radius_y_px_stage2", 0.5))
        s2x = float(self.get_param(cfg, "translation_refine_step_x_px_stage2", 0.1))
        s2y = float(self.get_param(cfg, "translation_refine_step_y_px_stage2", 0.1))

        def _offsets(radius: float, step: float) -> np.ndarray:
            rr = max(float(radius), 0.0)
            ss = float(step)
            if ss <= 0:
                return np.asarray([0.0], dtype=np.float32)
            return np.arange(-rr, rr + 0.5 * ss, ss, dtype=np.float32)

        def _eval_tx_ty(tx: float, ty: float) -> tuple[float, float, np.ndarray]:
            w = _build_warp_from_rotation_translation(R, tx, ty)
            ar = _warp_image(ref, w, output_shape=ins.shape)
            vm = _compute_valid_mask(ref.shape, w, output_shape=ins.shape)
            sc = _masked_abs_score(ar, ins, vm)
            return sc, float(np.mean(vm.astype(np.float32))), w

        seed_score, seed_overlap, _ = _eval_tx_ty(tx0, ty0)
        best1 = {"tx": tx0, "ty": ty0, "score": seed_score, "overlap_fraction": seed_overlap}
        stage1_records: list[dict] = []
        for dx in _offsets(r1x, s1x):
            for dy in _offsets(r1y, s1y):
                tx = tx0 + float(dx)
                ty = ty0 + float(dy)
                sc, ov, _ = _eval_tx_ty(tx, ty)
                rec = {"tx": float(tx), "ty": float(ty), "score": float(sc), "overlap_fraction": float(ov), "stage": "stage1"}
                stage1_records.append(rec)
                if sc < float(best1["score"]):
                    best1 = rec

        best2 = dict(best1)
        stage2_records: list[dict] = []
        for dx in _offsets(r2x, s2x):
            for dy in _offsets(r2y, s2y):
                tx = float(best1["tx"]) + float(dx)
                ty = float(best1["ty"]) + float(dy)
                sc, ov, _ = _eval_tx_ty(tx, ty)
                rec = {"tx": float(tx), "ty": float(ty), "score": float(sc), "overlap_fraction": float(ov), "stage": "stage2"}
                stage2_records.append(rec)
                if sc < float(best2["score"]):
                    best2 = rec

        final_warp = _build_warp_from_rotation_translation(R, float(best2["tx"]), float(best2["ty"]))
        aligned_ref = _warp_image(ref, final_warp, output_shape=ins.shape)
        valid_mask = _compute_valid_mask(ref.shape, final_warp, output_shape=ins.shape)

        aff_final = affine_meta.get("affine_params_final", {}) or {}
        suspicious = []
        if abs(float(aff_final.get("scale_x", 1.0)) - 1.0) > 0.1:
            suspicious.append("affine_scale_x_suspicious")
        if abs(float(aff_final.get("scale_y", 1.0)) - 1.0) > 0.1:
            suspicious.append("affine_scale_y_suspicious")
        if abs(float(aff_final.get("shear_proxy", 0.0))) > 0.15:
            suspicious.append("affine_shear_suspicious")
        if abs(float(aff_final.get("rotation_deg_from_matrix", 0.0))) > 8.0:
            suspicious.append("affine_rotation_suspicious")

        theta_deg = float(np.degrees(np.arctan2(float(R[0, 1]), float(R[0, 0]))))
        affine_meta.update(
            {
                "method": self.name,
                "motion_model_name": "affine_projected_euclidean",
                "affine_raw_warp_matrix": affine_warp.tolist(),
                "projected_rotation_matrix": np.asarray(R, dtype=np.float32).tolist(),
                "projected_theta_deg": float(theta_deg),
                "projected_tx_init": float(tx0),
                "projected_ty_init": float(ty0),
                "projected_refined_tx": float(best2["tx"]),
                "projected_refined_ty": float(best2["ty"]),
                "projected_stage1_best_tx": float(best1["tx"]),
                "projected_stage1_best_ty": float(best1["ty"]),
                "projected_stage1_best_score": float(best1["score"]),
                "projected_stage2_best_tx": float(best2["tx"]),
                "projected_stage2_best_ty": float(best2["ty"]),
                "projected_stage2_best_score": float(best2["score"]),
                "projected_translation_refinement_improvement": float(best2["score"] - seed_score),
                "projected_translation_candidates_total": int(len(stage1_records) + len(stage2_records)),
                "projected_stage1_records": stage1_records,
                "projected_stage2_records": stage2_records,
                "projected_score_seed": float(seed_score),
                "projected_score_final": float(best2["score"]),
                "projected_overlap_final": float(np.mean(valid_mask.astype(np.float32))),
                "affine_suspicious_flags": suspicious,
                "translation_x": float(-best2["tx"]),
                "translation_y": float(-best2["ty"]),
                "shift_x": float(-best2["tx"]),
                "shift_y": float(-best2["ty"]),
                "warp_matrix": np.asarray(final_warp, dtype=np.float32),
                "valid_mask": valid_mask.astype(bool),
                "valid_pixel_fraction": float(np.mean(valid_mask.astype(np.float32))),
                "rotation_degrees_estimated": float(theta_deg),
                "euclidean_projection_note": "Raw affine ECC projected to closest proper rotation via SVD, then tx/ty refined locally.",
            }
        )
        return aligned_ref, inspected_out, affine_meta
