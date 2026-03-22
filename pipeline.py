from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from config import PipelineConfig
from dd_types import DetectionResult, PipelineArtifacts, SamplePair
from utils.ground_truth_defects import get_ground_truth_points_for_pair
from utils.gt_coverage import compute_gt_point_coverage_metrics
from factories import (
    build_aligner,
    build_comparator,
    build_normalizer,
    build_postprocessor,
    build_preprocessor,
    build_thresholding,
)
from visualization.debug import save_compact_pipeline_figure, save_stage_visualizations

# Default outputs: ``<repo>/outs/...`` (this file lives at repo root).
_REPO_ROOT = Path(__file__).resolve().parent


class DefectDetectionPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

        self.preprocessor = build_preprocessor(cfg.choices.preprocessing)
        self.aligner = build_aligner(cfg.choices.alignment)
        self.normalizer = build_normalizer(cfg.choices.normalization)
        self.comparator = build_comparator(cfg.choices.comparison)
        self.thresholding = build_thresholding(cfg.choices.thresholding)
        self.postprocessor = build_postprocessor(cfg.choices.postprocessing)

    def run(self, sample: SamplePair) -> DetectionResult:
        artifacts, alignment_cfg = self._run_upstream(sample, silent=False)
        return self._run_downstream(sample, artifacts, alignment_cfg, silent=False, save_outputs=True)

    def run_through_normalization(self, sample: SamplePair, *, silent: bool = True) -> tuple[PipelineArtifacts, Any]:
        """Run preprocessing, alignment, normalization once; skip comparator and below."""
        return self._run_upstream(sample, silent=silent)

    def run_from_normalized(
        self,
        sample: SamplePair,
        cached_upstream: PipelineArtifacts,
        *,
        silent: bool = True,
    ) -> DetectionResult:
        """
        Run comparator → thresholding → postprocess only, reusing cached normalized images.
        Does not save figures or debug outputs (use sweep script to save compact figures).
        """
        snap = self._snapshot_upstream_artifacts(cached_upstream)
        return self._run_downstream(sample, snap, None, silent=silent, save_outputs=False)

    @staticmethod
    def _snapshot_upstream_artifacts(src: PipelineArtifacts) -> PipelineArtifacts:
        """Shallow copy of upstream fields only (same ndarray buffers)."""
        a = PipelineArtifacts()
        a.reference_raw = src.reference_raw
        a.inspected_raw = src.inspected_raw
        a.reference_input = src.reference_input
        a.inspected_input = src.inspected_input
        a.reference_preprocessed = src.reference_preprocessed
        a.inspected_preprocessed = src.inspected_preprocessed
        a.reference_aligned = src.reference_aligned
        a.inspected_aligned = src.inspected_aligned
        a.valid_mask = src.valid_mask
        a.alignment_metadata = src.alignment_metadata
        a.reference_normalized = src.reference_normalized
        a.inspected_normalized = src.inspected_normalized
        a.normalization_metadata = src.normalization_metadata
        a.normalization_debug = dict(src.normalization_debug) if src.normalization_debug else {}
        return a

    def _run_upstream(self, sample: SamplePair, *, silent: bool = False) -> tuple[PipelineArtifacts, Any]:
        artifacts = PipelineArtifacts(
            reference_raw=sample.reference_image,
            inspected_raw=sample.inspected_image,
            reference_input=sample.reference_image,
            inspected_input=sample.inspected_image,
        )

        ref = sample.reference_image
        ins = sample.inspected_image

        self._validate_inputs(ref, ins)
        if not silent:
            self._print_resolved_comparison_config(sample.pair_id)
            self._print_ref_ins_stats("raw", ref, ins)

        if self.cfg.preprocessing.enabled:
            ref, ins = self.preprocessor.run(ref, ins, self.cfg.preprocessing)
        artifacts.reference_preprocessed = ref
        artifacts.inspected_preprocessed = ins
        if not silent:
            self._print_ref_ins_stats("preprocessing", ref, ins)

        if self.cfg.alignment.enabled:
            alignment_cfg = self._resolve_alignment_config()
            ref, ins, alignment_metadata = self.aligner.run(ref, ins, alignment_cfg)
        else:
            alignment_cfg = None
            alignment_metadata = {}
        artifacts.reference_aligned = ref
        artifacts.inspected_aligned = ins
        artifacts.alignment_metadata = alignment_metadata
        artifacts.valid_mask = alignment_metadata.get("valid_mask", None)
        if not silent:
            self._print_ref_ins_stats("alignment", ref, ins)

        valid_mask = artifacts.valid_mask

        if self.cfg.normalization.enabled:
            self._set_runtime_param(self.cfg.normalization, "valid_mask", valid_mask)
            ref, ins, normalization_metadata = self.normalizer.run(
                ref, ins, self.cfg.normalization
            )
            self._clear_runtime_param(self.cfg.normalization, "valid_mask")
        else:
            normalization_metadata = {}
        artifacts.reference_normalized = ref
        artifacts.inspected_normalized = ins
        artifacts.normalization_metadata = normalization_metadata
        if not silent:
            self._print_ref_ins_stats("normalization", ref, ins)
        artifacts.normalization_debug = self._run_normalization_check(
            pair_id=sample.pair_id,
            reference_before=np.asarray(artifacts.reference_aligned, dtype=np.float32),
            inspected_before=np.asarray(artifacts.inspected_aligned, dtype=np.float32),
            reference_after=np.asarray(ref, dtype=np.float32),
            inspected_after=np.asarray(ins, dtype=np.float32),
            valid_mask=valid_mask,
            quiet=silent,
        )

        return artifacts, alignment_cfg

    def _run_downstream(
        self,
        sample: SamplePair,
        artifacts: PipelineArtifacts,
        alignment_cfg: Any,
        *,
        silent: bool,
        save_outputs: bool,
    ) -> DetectionResult:
        ref = artifacts.reference_normalized
        ins = artifacts.inspected_normalized
        if ref is None or ins is None:
            raise ValueError("run_from_normalized requires cached reference_normalized and inspected_normalized.")
        valid_mask = artifacts.valid_mask

        artifacts.comparison_metadata = {}
        artifacts.thresholding_metadata = {}
        artifacts.decision_metadata = {}
        artifacts.anomaly_map = None
        artifacts.ssim_map = None
        artifacts.artifact_residual_intermediates = None
        artifacts.threshold_map = None
        artifacts.binary_mask_raw = None
        artifacts.binary_mask_final = None

        comp_cfg = self._resolve_comparison_config()
        self._set_runtime_param(comp_cfg, "valid_mask", valid_mask)
        comparator_out = self.comparator.run(ref, ins, comp_cfg)
        self._clear_runtime_param(comp_cfg, "valid_mask")
        if isinstance(comparator_out, tuple) and len(comparator_out) == 2:
            anomaly_map, comparison_metadata = comparator_out
            ssim_map = comparison_metadata.pop("ssim_map", None)
            if ssim_map is not None:
                artifacts.ssim_map = ssim_map
            dbg_maps = comparison_metadata.pop("artifact_residual_debug_maps", None)
            if dbg_maps is not None:
                artifacts.artifact_residual_intermediates = dbg_maps
            artifacts.comparison_metadata.update(comparison_metadata)
        else:
            anomaly_map = comparator_out
        artifacts.anomaly_map = anomaly_map
        artifacts.comparison_metadata.update(self._compute_anomaly_stats(anomaly_map))
        if not silent:
            self._print_anomaly_stats(artifacts.comparison_metadata)

        self._set_runtime_param(self.cfg.thresholding, "valid_mask", valid_mask)
        threshold_out = self.thresholding.run(
            anomaly_map,
            self.cfg.thresholding,
        )
        self._clear_runtime_param(self.cfg.thresholding, "valid_mask")
        if isinstance(threshold_out, tuple) and len(threshold_out) == 3:
            binary_mask_raw, threshold_map, thresholding_metadata = threshold_out
            artifacts.thresholding_metadata.update(thresholding_metadata)
        else:
            binary_mask_raw, threshold_map = threshold_out
        artifacts.binary_mask_raw = binary_mask_raw
        artifacts.threshold_map = threshold_map
        artifacts.thresholding_metadata.update(
            self._compute_threshold_stats(
                binary_mask_raw,
                threshold_map,
                artifacts.thresholding_metadata,
            )
        )
        if not silent:
            self._print_threshold_stats(artifacts.thresholding_metadata)

        if not silent:
            self._print_resolved_postprocess_config(sample.pair_id)
        post_cfg = self._resolve_postprocessing_config()
        self._set_runtime_param(post_cfg, "valid_mask", valid_mask)
        signed_residual = np.asarray(ins, dtype=np.float32) - np.asarray(ref, dtype=np.float32)
        self._set_runtime_param(post_cfg, "signed_residual", signed_residual)
        self._set_runtime_param(post_cfg, "pair_id", sample.pair_id)
        ar_inter = artifacts.artifact_residual_intermediates
        if ar_inter and isinstance(ar_inter, dict):
            if "edge_mask_dilated" in ar_inter:
                em = np.asarray(ar_inter["edge_mask_dilated"], dtype=np.float32)
                self._set_runtime_param(post_cfg, "edge_exclude_mask", em > 0.5)
            # Pre-normalization residual energy avoids flat saturated peaks in robust-normalized anomaly [0,1].
            if "combined_after_edge" in ar_inter:
                self._set_runtime_param(
                    post_cfg,
                    "peak_score_map",
                    np.asarray(ar_inter["combined_after_edge"], dtype=np.float32),
                )
        gt_pts = get_ground_truth_points_for_pair(sample.pair_id)
        if gt_pts:
            self._set_runtime_param(post_cfg, "gt_points", gt_pts)
        self._set_runtime_param(post_cfg, "threshold_map", threshold_map)
        binary_mask_final, decision_metadata = self.postprocessor.run(
            binary_mask_raw,
            anomaly_map,
            post_cfg,
        )
        self._clear_runtime_param(post_cfg, "signed_residual")
        self._clear_runtime_param(post_cfg, "gt_points")
        self._clear_runtime_param(post_cfg, "pair_id")
        self._clear_runtime_param(post_cfg, "edge_exclude_mask")
        self._clear_runtime_param(post_cfg, "peak_score_map")
        self._clear_runtime_param(post_cfg, "threshold_map")
        self._clear_runtime_param(post_cfg, "valid_mask")
        artifacts.binary_mask_final = binary_mask_final
        artifacts.decision_metadata.update(decision_metadata)
        if not silent:
            self._print_postprocessing_stats(binary_mask_final, artifacts.decision_metadata)
            self._print_gt_audit(sample.pair_id, artifacts.decision_metadata)
            self._print_postprocess_sanity(sample.pair_id, artifacts.decision_metadata)
            if gt_pts:
                self._print_gt_point_coverage(sample.pair_id, binary_mask_final, gt_pts)
        if save_outputs:
            self._save_postprocess_audit(sample.pair_id, artifacts)
            self._save_compact_pipeline_figure(sample.pair_id, artifacts)
            self._save_ecc_affine_log(sample.pair_id, artifacts, alignment_cfg)
            self._save_search_euclidean_log(sample.pair_id, artifacts, alignment_cfg)
            self._save_debug_visualizations(sample.pair_id, artifacts)
            if self.cfg.output.save_intermediate:
                self._save_artifacts(sample.pair_id, artifacts)

        return DetectionResult(
            pair_id=sample.pair_id,
            defect_mask=binary_mask_final,
            artifacts=artifacts if self.cfg.output.return_artifacts else PipelineArtifacts(),
        )

    def _validate_inputs(self, ref: np.ndarray, ins: np.ndarray) -> None:
        if not isinstance(ref, np.ndarray) or not isinstance(ins, np.ndarray):
            raise TypeError("Both reference_image and inspected_image must be numpy arrays.")

        if self.cfg.fail_on_shape_mismatch and ref.shape != ins.shape:
            raise ValueError(
                f"Shape mismatch between reference and inspected images: "
                f"{ref.shape} vs {ins.shape}"
            )

    def _save_artifacts(self, pair_id: str, artifacts: PipelineArtifacts) -> None:
        save_dir = self.cfg.output.save_dir
        if save_dir is None:
            raise ValueError("output.save_dir must be provided if save_intermediate=True")
        _ = Path(save_dir) / pair_id
        raise NotImplementedError("_save_artifacts is not implemented yet.")

    def _save_debug_visualizations(self, pair_id: str, artifacts: PipelineArtifacts) -> None:
        if not self.cfg.debug.enable_debug_visualization:
            return
        if not self.cfg.debug.save_debug_images:
            return

        if self.cfg.debug.debug_dir is None:
            debug_root = _REPO_ROOT / "outs" / "debug"
        else:
            debug_root = Path(self.cfg.debug.debug_dir)

        try:
            save_stage_visualizations(
                artifacts=artifacts,
                pair_id=pair_id,
                output_dir=debug_root / pair_id,
            )
        except Exception as exc:
            print(f"[debug-viz] pair_id={pair_id} status=SKIPPED reason={exc}")

    def _save_compact_pipeline_figure(self, pair_id: str, artifacts: PipelineArtifacts) -> None:
        out_path = Path(__file__).resolve().parent / "outs" / "detection_results" / f"{pair_id}_pipeline.png"
        try:
            save_compact_pipeline_figure(
                pair_id=pair_id,
                artifacts=artifacts,
                comparator=self.comparator,
                comparator_cfg=self._resolve_comparison_config(),
                output_path=out_path,
            )
        except Exception as exc:
            print(f"[pipeline-figure] pair_id={pair_id} status=SKIPPED reason={exc}")

    def _save_ecc_affine_log(self, pair_id: str, artifacts: PipelineArtifacts, alignment_cfg) -> None:
        meta = artifacts.alignment_metadata or {}
        if str(meta.get("method", "")) not in {"ecc_affine", "ecc_affine_projected_euclidean"}:
            return

        out_path = _REPO_ROOT / "outs" / "detection_results" / f"{pair_id}_ecc_affine_log.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _safe(v, d="NA"):
            return d if v is None else v

        def _arr_stats(name: str, arr: np.ndarray | None) -> list[str]:
            if arr is None:
                return [f"- {name}: N/A"]
            a = np.asarray(arr)
            af = np.asarray(a, dtype=np.float32)
            return [
                f"- {name}: dtype={a.dtype}, min={float(np.min(af)):.6f}, max={float(np.max(af)):.6f}, "
                f"mean={float(np.mean(af)):.6f}, std={float(np.std(af)):.6f}, range={float(np.max(af)-np.min(af)):.6f}"
            ]

        def _format_warp(warp) -> list[str]:
            if warp is None:
                return ["  [NA NA NA]", "  [NA NA NA]"]
            wm = np.asarray(warp, dtype=np.float32)
            if wm.shape != (2, 3):
                return [f"  shape={wm.shape}"]
            return [
                f"  [{wm[0,0]: .6f} {wm[0,1]: .6f} {wm[0,2]: .6f}]",
                f"  [{wm[1,0]: .6f} {wm[1,1]: .6f} {wm[1,2]: .6f}]",
            ]

        before_abs = None
        after_abs = None
        vm = artifacts.valid_mask
        if (
            artifacts.reference_preprocessed is not None
            and artifacts.inspected_preprocessed is not None
            and artifacts.reference_aligned is not None
            and artifacts.inspected_aligned is not None
        ):
            before_abs = np.abs(
                np.asarray(artifacts.inspected_preprocessed, dtype=np.float32)
                - np.asarray(artifacts.reference_preprocessed, dtype=np.float32)
            )
            after_abs = np.abs(
                np.asarray(artifacts.inspected_aligned, dtype=np.float32)
                - np.asarray(artifacts.reference_aligned, dtype=np.float32)
            )

        def _masked_vals(x: np.ndarray | None):
            if x is None:
                return None
            arr = np.asarray(x, dtype=np.float32)
            if vm is None:
                return arr.reshape(-1)
            m = np.asarray(vm).astype(bool)
            if m.shape != arr.shape or not np.any(m):
                return arr.reshape(-1)
            return arr[m]

        before_vals = _masked_vals(before_abs)
        after_vals = _masked_vals(after_abs)
        before_mean_abs = float(np.mean(before_vals)) if before_vals is not None else None
        after_mean_abs = float(np.mean(after_vals)) if after_vals is not None else None
        after_med_abs = float(np.median(after_vals)) if after_vals is not None else None
        after_mad_abs = (
            float(np.median(np.abs(after_vals - np.median(after_vals))))
            if after_vals is not None
            else None
        )

        comparator_before_mean = None
        comparator_after_mean = None
        try:
            if (
                artifacts.reference_preprocessed is not None
                and artifacts.inspected_preprocessed is not None
                and artifacts.reference_aligned is not None
                and artifacts.inspected_aligned is not None
            ):
                before_map = self._compute_anomaly_with_comparator(
                    np.asarray(artifacts.reference_preprocessed, dtype=np.float32),
                    np.asarray(artifacts.inspected_preprocessed, dtype=np.float32),
                    vm,
                )
                after_map = self._compute_anomaly_with_comparator(
                    np.asarray(artifacts.reference_aligned, dtype=np.float32),
                    np.asarray(artifacts.inspected_aligned, dtype=np.float32),
                    vm,
                )
                comparator_before_mean = float(self._compute_masked_map_stats(before_map, vm)["mean"])
                comparator_after_mean = float(self._compute_masked_map_stats(after_map, vm)["mean"])
        except Exception:
            pass

        warnings: list[str] = []
        aff = meta.get("affine_params_final", {}) or {}
        sx = aff.get("scale_x")
        sy = aff.get("scale_y")
        shear = aff.get("shear_proxy")
        tx = float(meta.get("translation_x", 0.0))
        ty = float(meta.get("translation_y", 0.0))
        valid_fraction = float(meta.get("valid_pixel_fraction", 0.0))
        ecc_i = meta.get("ecc_initial_correlation")
        ecc_f = meta.get("ecc_correlation")

        if sx is not None and (float(sx) < 0.8 or float(sx) > 1.2):
            warnings.append(f"scale_x looks extreme: {float(sx):.4f}")
        if sy is not None and (float(sy) < 0.8 or float(sy) > 1.2):
            warnings.append(f"scale_y looks extreme: {float(sy):.4f}")
        if shear is not None and abs(float(shear)) > 0.2:
            warnings.append(f"shear_proxy magnitude is high: {float(shear):.4f}")
        if valid_fraction < 0.90:
            warnings.append(f"valid overlap fraction is low: {valid_fraction:.4f}")
        if ecc_i is not None and ecc_f is not None and (float(ecc_f) - float(ecc_i)) < 1e-4:
            warnings.append("ECC correlation improvement is very small.")
        if before_mean_abs is not None and after_mean_abs is not None and after_mean_abs >= before_mean_abs * 0.98:
            warnings.append("Absolute residual improved very little after affine alignment.")
        if abs(tx) < 1.0 and abs(ty) < 1.0 and after_mean_abs is not None and before_mean_abs is not None and after_mean_abs > before_mean_abs * 0.9:
            warnings.append("Transform near identity but residual remains high.")
        if (abs(tx) > 0.2 * max(1.0, np.sqrt(np.prod(np.asarray(meta.get('valid_mask', np.ones((1, 1))).shape[:2]))))) and before_mean_abs is not None and after_mean_abs is not None and after_mean_abs > before_mean_abs * 0.95:
            warnings.append("Large translation with little residual improvement.")

        lines: list[str] = []
        lines.append("=== ECC AFFINE ALIGNMENT LOG ===")
        lines.append("")
        lines.append("1) Header / config")
        lines.append(f"- pair_id: {pair_id}")
        lines.append(f"- aligner: {meta.get('method', 'NA')}")
        lines.append(f"- image_shape: {np.asarray(artifacts.inspected_aligned).shape if artifacts.inspected_aligned is not None else 'NA'}")
        lines.append(f"- timestamp: {datetime.now().isoformat(timespec='seconds')}")
        lines.append(f"- motion_model: {meta.get('motion_model_name', 'NA')} (warp_mode={meta.get('warp_mode', 'NA')})")
        if alignment_cfg is not None and hasattr(alignment_cfg, "params"):
            p = alignment_cfg.params or {}
            lines.append(f"- max_iterations: {_safe(p.get('number_of_iterations'))}")
            lines.append(f"- termination_eps: {_safe(p.get('termination_eps'))}")
            lines.append(f"- gaussian_filter_size: {_safe(p.get('gaussian_filter_size'))}")
            lines.append(f"- use_gradient_images: {_safe(p.get('use_gradient_images'))}")
            lines.append(f"- allow_fallback_to_identity: {_safe(p.get('allow_fallback_to_identity'))}")
        lines.append("")
        lines.append("2) Input statistics")
        lines.extend(_arr_stats("reference passed into ECC", artifacts.reference_preprocessed))
        lines.extend(_arr_stats("inspected passed into ECC", artifacts.inspected_preprocessed))
        lines.append(f"- internal ECC preprocessing: min-max normalization; gradient_mode={meta.get('use_gradient_images', False)}")
        lines.append("")
        lines.append("3) Initialization")
        lines.append("- initial_warp_matrix:")
        lines.extend(_format_warp(meta.get("initial_warp_matrix")))
        aff_init = meta.get("affine_params_initial", {}) or {}
        lines.append(
            f"- initial params: tx={float(aff_init.get('matrix_tx', 0.0)):.4f}, ty={float(aff_init.get('matrix_ty', 0.0)):.4f}, "
            f"rot={float(aff_init.get('rotation_deg_from_matrix', 0.0)):.4f} deg, sx={float(aff_init.get('scale_x', 1.0)):.4f}, "
            f"sy={float(aff_init.get('scale_y', 1.0)):.4f}, shear_proxy={float(aff_init.get('shear_proxy', 0.0)):.4f}"
        )
        lines.append("- initialization source: identity warp")
        lines.append("")
        lines.append("4) ECC optimization process")
        lines.append(f"- ecc_score_initial: {_safe(meta.get('ecc_initial_correlation'))}")
        lines.append(f"- ecc_score_final: {_safe(meta.get('ecc_correlation'))}")
        lines.append(f"- requested_max_iterations: {_safe(meta.get('requested_number_of_iterations'))}")
        lines.append(f"- requested_eps: {_safe(meta.get('requested_termination_eps'))}")
        lines.append(f"- returned_iteration_count: {_safe(meta.get('ecc_iteration_count_exposed'), 'not exposed by OpenCV API')}")
        lines.append(f"- converged_flag: {_safe(meta.get('ecc_converged'))}")
        lines.append("")
        lines.append("5) Final warp / transform")
        lines.append("- final_warp_matrix:")
        lines.extend(_format_warp(meta.get("warp_matrix")))
        lines.append(
            f"- final params (applied-transform convention): tx={float(meta.get('translation_x', 0.0)):.4f}, "
            f"ty={float(meta.get('translation_y', 0.0)):.4f}, rot={float(aff.get('rotation_deg_from_matrix', 0.0)):.4f} deg, "
            f"sx={float(aff.get('scale_x', 1.0)):.4f}, sy={float(aff.get('scale_y', 1.0)):.4f}, "
            f"shear_proxy={float(aff.get('shear_proxy', 0.0)):.4f}"
        )
        lines.append(f"- warp convention note: {meta.get('warp_convention_note', 'NA')}")
        if meta.get("method") == "ecc_affine_projected_euclidean":
            lines.append("")
            lines.append("5b) Affine raw vs projected-euclidean")
            lines.append("- affine_raw_warp_matrix:")
            lines.extend(_format_warp(meta.get("affine_raw_warp_matrix")))
            lines.append(f"- projected_theta_deg: {_safe(meta.get('projected_theta_deg'))}")
            lines.append(
                f"- projected tx/ty init: tx={float(meta.get('projected_tx_init', 0.0)):.4f}, "
                f"ty={float(meta.get('projected_ty_init', 0.0)):.4f}"
            )
            lines.append(
                f"- projected tx/ty refined: tx={float(meta.get('projected_refined_tx', 0.0)):.4f}, "
                f"ty={float(meta.get('projected_refined_ty', 0.0)):.4f}"
            )
            lines.append(
                f"- projected score seed/final: seed={_safe(meta.get('projected_score_seed'))}, "
                f"final={_safe(meta.get('projected_score_final'))}"
            )
            lines.append(
                f"- projected translation refinement improvement: {_safe(meta.get('projected_translation_refinement_improvement'))}"
            )
            flags = meta.get("affine_suspicious_flags", []) or []
            lines.append(f"- affine suspicious flags: {flags}")
        lines.append("")
        lines.append("6) Alignment quality metrics")
        valid_count = int(np.count_nonzero(np.asarray(vm).astype(bool))) if vm is not None else int(np.asarray(artifacts.inspected_aligned).size if artifacts.inspected_aligned is not None else 0)
        lines.append(f"- valid_pixel_count: {valid_count}")
        lines.append(f"- valid_pixel_fraction: {valid_fraction:.6f}")
        lines.append(f"- residual_before_mean_abs: {_safe(None if before_mean_abs is None else f'{before_mean_abs:.6f}')}")
        lines.append(f"- residual_after_mean_abs: {_safe(None if after_mean_abs is None else f'{after_mean_abs:.6f}')}")
        lines.append(f"- residual_after_median_abs: {_safe(None if after_med_abs is None else f'{after_med_abs:.6f}')}")
        lines.append(f"- residual_after_MAD: {_safe(None if after_mad_abs is None else f'{after_mad_abs:.6f}')}")
        lines.append(f"- comparator_anomaly_before_mean: {_safe(None if comparator_before_mean is None else f'{comparator_before_mean:.6f}')}")
        lines.append(f"- comparator_anomaly_after_mean: {_safe(None if comparator_after_mean is None else f'{comparator_after_mean:.6f}')}")
        lines.append("")
        lines.append("7) Warnings / anomalies")
        if warnings:
            for w in warnings:
                lines.append(f"- WARNING: {w}")
        else:
            lines.append("- none")
        lines.append("")
        lines.append("=== ECC AFFINE ALIGNMENT SUMMARY ===")
        lines.append(f"pair_id: {pair_id}")
        lines.append(
            f"config: iterations={_safe(meta.get('requested_number_of_iterations'))}, eps={_safe(meta.get('requested_termination_eps'))}, "
            f"gaussian_filter_size={_safe(meta.get('gaussian_filter_size_used'))}"
        )
        lines.append(
            f"initial_warp: tx={float(aff_init.get('matrix_tx', 0.0)):.4f}, ty={float(aff_init.get('matrix_ty', 0.0)):.4f}, "
            f"rot={float(aff_init.get('rotation_deg_from_matrix', 0.0)):.4f}, sx={float(aff_init.get('scale_x', 1.0)):.4f}, sy={float(aff_init.get('scale_y', 1.0)):.4f}"
        )
        lines.append(
            f"final_warp: tx={float(meta.get('translation_x', 0.0)):.4f}, ty={float(meta.get('translation_y', 0.0)):.4f}, "
            f"rot={float(aff.get('rotation_deg_from_matrix', 0.0)):.4f}, sx={float(aff.get('scale_x', 1.0)):.4f}, sy={float(aff.get('scale_y', 1.0)):.4f}"
        )
        lines.append(
            f"ecc_score: initial={_safe(meta.get('ecc_initial_correlation'))}, final={_safe(meta.get('ecc_correlation'))}"
        )
        lines.append(
            f"residual_mean_abs: before={_safe(None if before_mean_abs is None else f'{before_mean_abs:.6f}')}, "
            f"after={_safe(None if after_mean_abs is None else f'{after_mean_abs:.6f}')}"
        )
        lines.append(f"valid_fraction: {valid_fraction:.6f}")
        lines.append("warnings:")
        if warnings:
            for w in warnings:
                lines.append(f"- {w}")
        else:
            lines.append("- none")
        lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved ECC affine log to {out_path}")

    def _save_search_euclidean_log(self, pair_id: str, artifacts: PipelineArtifacts, alignment_cfg) -> None:
        meta = artifacts.alignment_metadata or {}
        if str(meta.get("method", "")) != "search_euclidean":
            return

        out_path = Path(__file__).resolve().parent / "outs" / "detection_results" / f"{pair_id}_alignment_log.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        score_mode = str(meta.get("score_mode", "NA"))
        edge_mode = score_mode == "edge_distance"
        candidate_records = list(meta.get("candidate_records", []) or [])
        iteration_summaries = list(meta.get("iteration_summaries", []) or [])
        valid_mask = meta.get("valid_mask", artifacts.valid_mask)
        vm = None if valid_mask is None else np.asarray(valid_mask).astype(bool)

        def _safe(v, d="NA"):
            return d if v is None else v

        lines: list[str] = []
        lines.append("=== SEARCH EUCLIDEAN ALIGNMENT LOG ===")
        lines.append("")
        lines.append("HEADER / CONFIG")
        lines.append(f"- pair_id: {pair_id}")
        lines.append(f"- aligner: {meta.get('method', 'NA')}")
        lines.append(f"- score_mode: {score_mode}")
        lines.append(f"- image_shape: {np.asarray(artifacts.inspected_aligned).shape if artifacts.inspected_aligned is not None else 'NA'}")
        if alignment_cfg is not None and hasattr(alignment_cfg, "params"):
            p = alignment_cfg.params or {}
            lines.append(f"- coarse_angle_min: {_safe(p.get('coarse_angle_min'))}")
            lines.append(f"- coarse_angle_max: {_safe(p.get('coarse_angle_max'))}")
            lines.append(f"- coarse_steps: {_safe(p.get('coarse_steps'))}")
            lines.append(f"- refine_half_width: {_safe(p.get('refine_half_width'))}")
            lines.append(f"- refine_steps: {_safe(p.get('refine_steps'))}")
            lines.append(f"- overlap_threshold: {_safe(p.get('overlap_threshold'))}")
            lines.append(f"- upsample_factor: {_safe(p.get('upsample_factor'))}")
            if edge_mode:
                lines.append(f"- edge_method: {_safe(p.get('edge_method'))}")
                lines.append(f"- canny_threshold1: {_safe(p.get('canny_threshold1'))}")
                lines.append(f"- canny_threshold2: {_safe(p.get('canny_threshold2'))}")
                lines.append(f"- sobel_edge_percentile: {_safe(p.get('sobel_edge_percentile'))}")
                lines.append(f"- min_edge_pixels_for_score: {_safe(p.get('min_edge_pixels_for_score'))}")
        lines.append("")
        lines.append("INPUT STATISTICS")
        if artifacts.reference_preprocessed is not None:
            rr = np.asarray(artifacts.reference_preprocessed, dtype=np.float32)
            lines.append(
                f"- reference_preprocessed: dtype={rr.dtype}, min={float(np.min(rr)):.6f}, max={float(np.max(rr)):.6f}, mean={float(np.mean(rr)):.6f}, std={float(np.std(rr)):.6f}"
            )
        if artifacts.inspected_preprocessed is not None:
            ii = np.asarray(artifacts.inspected_preprocessed, dtype=np.float32)
            lines.append(
                f"- inspected_preprocessed: dtype={ii.dtype}, min={float(np.min(ii)):.6f}, max={float(np.max(ii)):.6f}, mean={float(np.mean(ii)):.6f}, std={float(np.std(ii)):.6f}"
            )

        if edge_mode:
            ins_edge = meta.get("inspected_edge_map")
            ins_dist = meta.get("inspected_edge_distance_map")
            lines.append("")
            lines.append("EDGE PRECOMPUTE")
            if ins_edge is not None:
                e = np.asarray(ins_edge).astype(bool)
                lines.append(f"- inspected_edge_count: {int(np.count_nonzero(e))}")
                lines.append(f"- inspected_edge_fraction: {float(np.mean(e.astype(np.float32))):.6f}")
            if ins_dist is not None:
                d = np.asarray(ins_dist, dtype=np.float32)
                lines.append(f"- distance_transform_min: {float(np.min(d)):.6f}")
                lines.append(f"- distance_transform_mean: {float(np.mean(d)):.6f}")
                lines.append(f"- distance_transform_max: {float(np.max(d)):.6f}")
            if artifacts.reference_preprocessed is not None:
                rp = np.asarray(artifacts.reference_preprocessed, dtype=np.float32)
                lo = float(np.min(rp))
                hi = float(np.max(rp))
                if hi > lo:
                    rn = (rp - lo) / (hi - lo)
                else:
                    rn = np.zeros_like(rp, dtype=np.float32)
                r8 = np.clip(rn * 255.0, 0, 255).astype(np.uint8)
                ref_edges = cv2.Canny(r8, 50.0, 150.0)
                lines.append(f"- reference_edge_count_before_warp: {int(np.count_nonzero(ref_edges > 0))}")

        lines.append("")
        lines.append("PER-ITERATION SEARCH")
        rec_by_it: dict[int, list[dict]] = {}
        for r in candidate_records:
            it = int(r.get("iteration", -1))
            if it < 0:
                continue
            rec_by_it.setdefault(it, []).append(r)
        prev_best_score = None
        prev_theta = None
        prev_tx = None
        prev_ty = None
        for s in iteration_summaries:
            it = int(s.get("iteration", -1))
            if it < 0:
                continue
            lines.append(f"- iteration={it} theta_range=[{float(s.get('theta_search_min',0.0)):.4f},{float(s.get('theta_search_max',0.0)):.4f}]")
            it_recs = rec_by_it.get(it, [])
            it_recs = sorted(it_recs, key=lambda x: float(x.get("score", float("inf"))))
            for rank, r in enumerate(it_recs, start=1):
                st = r.get("score_terms", {}) or {}
                base = (
                    f"  cand#{rank:02d} theta={float(r.get('theta_deg',0.0)):+.4f} "
                    f"tx={float(r.get('tx',0.0)):+.4f} ty={float(r.get('ty',0.0)):+.4f} "
                    f"overlap={float(r.get('overlap_fraction',0.0)):.4f} score={float(r.get('score',float('inf'))):.6f} "
                    f"accepted={bool(r.get('valid',False))}"
                )
                if edge_mode:
                    edge_score = float(st.get("edge_distance_term", r.get("score", float("inf"))))
                    edge_used = int(st.get("warped_reference_edge_count", 0.0))
                    edge_valid_fraction = float(st.get("edge_coverage_fraction", 0.0))
                    base += (
                        f" edge_score={edge_score:.6f} edge_pixels_used={edge_used} "
                        f"edge_valid_fraction={edge_valid_fraction:.6f}"
                    )
                lines.append(base)

            cur_best_score = float(s.get("best_score", s.get("score", float("inf"))))
            cur_theta = float(s.get("chosen_theta_deg", 0.0))
            cur_tx = float(s.get("chosen_tx", 0.0))
            cur_ty = float(s.get("chosen_ty", 0.0))
            lines.append(
                f"  winner: theta={cur_theta:+.4f}, tx={cur_tx:+.4f}, ty={cur_ty:+.4f}, score={cur_best_score:.6f}"
            )
            if prev_best_score is not None:
                lines.append(
                    f"  improvement_vs_prev: dscore={cur_best_score - prev_best_score:+.6f}, "
                    f"dtheta={cur_theta - prev_theta:+.6f}, dtx={cur_tx - prev_tx:+.6f}, dty={cur_ty - prev_ty:+.6f}"
                )
            prev_best_score = cur_best_score
            prev_theta = cur_theta
            prev_tx = cur_tx
            prev_ty = cur_ty

        if edge_mode:
            lines.append("")
            lines.append("EDGE MATCH (BEST)")
            st_best = meta.get("score_terms_best", {}) or {}
            lines.append(f"- warped_ref_edge_count: {int(st_best.get('warped_reference_edge_count', 0.0))}")
            lines.append(f"- pixels_scored: {int(st_best.get('edge_coverage_count', 0.0))}")
            lines.append(f"- mean_distance: {float(st_best.get('edge_distance_term', meta.get('best_score', float('nan')))):.6f}")
            recs = [r for r in candidate_records if bool(r.get("valid", False))]
            if recs:
                dvals = []
                for r in recs:
                    st = r.get("score_terms", {}) or {}
                    if np.isfinite(float(st.get("edge_distance_term", float("nan")))):
                        dvals.append(float(st.get("edge_distance_term")))
                if dvals:
                    arr = np.asarray(dvals, dtype=np.float32)
                    lines.append(f"- median_distance: {float(np.median(arr)):.6f}")
                    lines.append(f"- p90_distance: {float(np.percentile(arr, 90.0)):.6f}")

        lines.append("")
        lines.append("FINAL TRANSFORM")
        lines.append(
            f"- final_theta_deg={float(meta.get('final_theta_deg',0.0)):+.6f}, "
            f"final_tx={float(meta.get('final_tx',0.0)):+.6f}, "
            f"final_ty={float(meta.get('final_ty',0.0)):+.6f}, "
            f"final_score={float(meta.get('final_score', float('nan'))):.6f}"
        )
        lines.append(
            f"- overlap_fraction={float(meta.get('overlap_fraction',0.0)):.6f}, "
            f"valid_fraction={float(meta.get('valid_pixel_fraction',0.0)):.6f}, "
            f"iterations_used={_safe(meta.get('iterations_used',2))}, converged={_safe(meta.get('converged','NA'))}"
        )

        warn: list[str] = []
        if edge_mode:
            warn.extend(list(meta.get("edge_score_warnings", []) or []))
            st_best = meta.get("score_terms_best", {}) or {}
            if int(st_best.get("warped_reference_edge_count", 0.0)) < 50:
                warn.append("Too few warped-reference edges used by best candidate.")
            ins_count = int(meta.get("inspected_edge_pixel_count", 0))
            best_count = int(st_best.get("warped_reference_edge_count", 0.0))
            if ins_count > 0 and best_count > 0 and (best_count / max(1.0, float(ins_count))) < 0.1:
                warn.append("Severe edge-count imbalance between inspected and warped-reference edges.")
            if float(st_best.get("edge_coverage_fraction", 0.0)) < 0.005:
                warn.append("Edge-valid fraction is very low for best candidate.")
            if vm is not None and np.any(vm):
                border = np.zeros_like(vm, dtype=bool)
                border[:3, :] = True
                border[-3:, :] = True
                border[:, :3] = True
                border[:, -3:] = True
                wrm = np.asarray(meta.get("best_warped_reference_edge_map")) if meta.get("best_warped_reference_edge_map") is not None else None
                if wrm is not None and wrm.shape == vm.shape:
                    edge_pts = np.asarray(wrm).astype(bool)
                    used = np.count_nonzero(edge_pts)
                    if used > 0:
                        border_ratio = float(np.count_nonzero(np.logical_and(edge_pts, border)) / used)
                        if border_ratio > 0.7:
                            warn.append("Most scored edges are near borders.")

        lines.append("")
        lines.append("WARNINGS")
        if warn:
            for w in warn:
                lines.append(f"- WARNING: {w}")
        else:
            lines.append("- none")

        lines.append("")
        lines.append("=== ALIGNMENT SUMMARY FOR DIAGNOSIS ===")
        lines.append(f"pair_id: {pair_id}")
        lines.append(
            f"config: score_mode={score_mode}, coarse_steps={_safe(meta.get('coarse_steps'))}, "
            f"refine_steps={_safe(meta.get('refine_steps'))}, overlap_threshold={float(meta.get('overlap_threshold',0.0)):.4f}"
        )
        winners = []
        for s in iteration_summaries:
            winners.append(
                f"it{s.get('iteration')}:(theta={float(s.get('chosen_theta_deg',0.0)):+.3f},"
                f"tx={float(s.get('chosen_tx',0.0)):+.2f},ty={float(s.get('chosen_ty',0.0)):+.2f},"
                f"score={float(s.get('best_score', s.get('score', float('nan')))):.4f})"
            )
        lines.append("per-iteration winners: " + ("; ".join(winners) if winners else "NA"))
        if edge_mode:
            st_best = meta.get("score_terms_best", {}) or {}
            lines.append(
                f"edge summary: edge_score_best={float(st_best.get('edge_distance_term', meta.get('best_score', float('nan')))):.6f}, "
                f"edge_pixels_used={int(st_best.get('warped_reference_edge_count',0.0))}"
            )
        lines.append(
            f"final transform: theta={float(meta.get('final_theta_deg',0.0)):+.4f}, "
            f"tx={float(meta.get('final_tx',0.0)):+.4f}, ty={float(meta.get('final_ty',0.0)):+.4f}, "
            f"score={float(meta.get('final_score', float('nan'))):.6f}"
        )
        lines.append("warnings: " + (" | ".join(warn) if warn else "none"))
        lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Saved search alignment log to {out_path}")

    def _print_ref_ins_stats(self, stage: str, ref: np.ndarray, ins: np.ndarray) -> None:
        residual = ins.astype(np.float32, copy=False) - ref.astype(np.float32, copy=False)
        print(
            f"[diag:{stage}] residual(ins-ref): "
            f"mean={float(np.mean(residual)):.6f}, "
            f"std={float(np.std(residual)):.6f}, "
            f"min={float(np.min(residual)):.6f}, "
            f"max={float(np.max(residual)):.6f}, "
            f"mean_abs={float(np.mean(np.abs(residual))):.6f}"
        )

    def _compute_anomaly_stats(self, anomaly_map: np.ndarray) -> dict:
        arr = anomaly_map.astype(np.float32, copy=False)
        percentiles = np.percentile(arr, [50, 90, 95, 99, 99.5, 99.9])
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p50": float(percentiles[0]),
            "p90": float(percentiles[1]),
            "p95": float(percentiles[2]),
            "p99": float(percentiles[3]),
            "p99_5": float(percentiles[4]),
            "p99_9": float(percentiles[5]),
        }

    def _print_anomaly_stats(self, stats: dict) -> None:
        print(
            "[diag:comparison] "
            f"min={stats['min']:.6f}, max={stats['max']:.6f}, "
            f"mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
            f"p50={stats['p50']:.6f}, p90={stats['p90']:.6f}, p95={stats['p95']:.6f}, "
            f"p99={stats['p99']:.6f}, p99.5={stats['p99_5']:.6f}, p99.9={stats['p99_9']:.6f}"
        )

    def _compute_threshold_stats(
        self,
        binary_mask_raw: np.ndarray,
        threshold_map: np.ndarray,
        thresholding_metadata: dict | None = None,
    ) -> dict:
        thr = float(np.mean(threshold_map.astype(np.float32, copy=False)))
        positive_count = int(np.count_nonzero(binary_mask_raw))
        total_count = int(binary_mask_raw.size)
        ratio = float((positive_count / total_count) * 100.0) if total_count else 0.0
        out = {
            "threshold": thr,
            "positive_count": positive_count,
            "total_count": total_count,
            "positive_percent": ratio,
        }
        md = thresholding_metadata or {}
        if "median" in md:
            out["median"] = float(md["median"])
        if "mad" in md:
            out["mad"] = float(md["mad"])
        if "valid_pixel_count" in md:
            out["valid_pixel_count"] = int(md["valid_pixel_count"])
        return out

    def _print_threshold_stats(self, stats: dict) -> None:
        line = (
            "[diag:thresholding] "
            f"threshold={stats['threshold']:.6f}, "
            f"positive_pixels={stats['positive_count']}/{stats['total_count']} "
            f"({stats['positive_percent']:.4f}%)"
        )
        if "median" in stats:
            line += f", median={stats['median']:.6f}"
        if "mad" in stats:
            line += f", mad={stats['mad']:.6f}"
        if "valid_pixel_count" in stats:
            line += f", valid_pixels={stats['valid_pixel_count']}"
        print(line)

    def _print_postprocessing_stats(self, binary_mask_final: np.ndarray, decision_metadata: dict) -> None:
        final_count = int(np.count_nonzero(binary_mask_final))
        total_count = int(binary_mask_final.size)
        final_percent = float((final_count / total_count) * 100.0) if total_count else 0.0
        num_kept_contours = int(
            decision_metadata.get(
                "num_kept_contours",
                decision_metadata.get(
                    "num_contours_kept",
                    decision_metadata.get("num_components", 0),
                ),
            )
        )
        contour_areas = decision_metadata.get("contour_areas_sorted_desc")
        if contour_areas is None:
            contour_areas = decision_metadata.get("contour_areas_kept")
        if contour_areas is None:
            contour_areas = []
            for comp in decision_metadata.get("components", []):
                area = comp.get("area")
                if isinstance(area, (int, float)):
                    contour_areas.append(float(area))
            contour_areas.sort(reverse=True)
        top_areas = [float(a) for a in list(contour_areas)[:5]]
        total_kept_area = decision_metadata.get("total_kept_area")
        if total_kept_area is None:
            total_kept_area = float(sum(float(a) for a in contour_areas))
        num_centers_drawn = int(decision_metadata.get("num_centers_drawn", num_kept_contours))
        reject_counts = decision_metadata.get("reject_counts")
        rc_str = ""
        if isinstance(reject_counts, dict) and reject_counts:
            rc_str = ", reject_counts={" + ", ".join(f"{k}:{int(reject_counts.get(k, 0))}" for k in sorted(reject_counts.keys())) + "}"
        nb_topk = decision_metadata.get("num_contours_after_score_threshold")
        if nb_topk is None:
            nb_topk = decision_metadata.get("num_kept_contours_before_topk")
        na_topk = decision_metadata.get("num_contours_after_topk")
        if na_topk is None:
            na_topk = decision_metadata.get("num_kept_contours_after_topk")
        top_scores = decision_metadata.get("top_scores_ranked") or decision_metadata.get("top_contour_scores") or decision_metadata.get("ranking_scores_sorted_desc") or []
        top_scores_5: list[float] = []
        if isinstance(top_scores, list):
            top_scores_5 = [float(x) for x in top_scores[:5]]
        topk_str = ""
        if nb_topk is not None and na_topk is not None:
            topk_str = (
                f", after_score_threshold={int(nb_topk)}, after_topk_cap={int(na_topk)}, "
                f"top_scores[:5]={top_scores_5}"
            )
        morph_str = ""
        pbm = decision_metadata.get("positive_pixels_before_morph")
        pam = decision_metadata.get("positive_pixels_after_morph")
        if pbm is not None and pam is not None:
            morph_str = f", positive_pixels_before_morph={int(pbm)}, positive_pixels_after_morph={int(pam)}"
        print(
            "[diag:postprocessing] "
            f"final_positive_pixels={final_count}/{total_count} ({final_percent:.4f}%), "
            f"num_kept_contours={num_kept_contours}, total_kept_area={float(total_kept_area):.1f}, "
            f"num_centers_drawn={num_centers_drawn}, top_contour_areas={top_areas}"
            f"{topk_str}{morph_str}{rc_str}"
        )
        if str(decision_metadata.get("method", "")) in {
            "contour_filter_postprocess",
            "peak_nms_postprocess",
        }:
            self._print_postprocessing_stage_counts(decision_metadata)
            self._print_postprocessing_rank_lists(decision_metadata)

    def _print_postprocessing_stage_counts(self, decision_metadata: dict) -> None:
        """Geom filters -> score threshold -> top-K cap (max K, not a quota)."""
        g = decision_metadata.get("num_contours_after_geom_filters")
        s = decision_metadata.get("num_contours_after_score_threshold")
        t = decision_metadata.get("num_contours_after_topk")
        st = decision_metadata.get("score_threshold_used")
        if g is None and s is None and t is None:
            return
        print(
            "[diag:postprocess_stages] "
            f"num_contours_after_geom_filters={g}, "
            f"num_contours_after_score_threshold={s}, "
            f"num_contours_after_topk={t}, "
            f"score_threshold_used={st}"
        )

    def _print_postprocessing_rank_lists(self, decision_metadata: dict) -> None:
        """Rank-ordered survivors (highest ranking score first; may be fewer than K)."""
        def _take10(key: str) -> list:
            v = decision_metadata.get(key)
            if not isinstance(v, list):
                return []
            return [float(x) for x in v[:10]]

        ts = _take10("top_scores_ranked") or _take10("top_contour_scores")
        ta = _take10("top_areas_ranked")
        tm = _take10("top_mean_inside_ranked") or _take10("top_mean_anomalies_ranked")
        tp = _take10("top_p95_inside_ranked") or _take10("top_p95_anomalies_ranked")
        tr = _take10("top_ring_mean_ranked")
        tsc = _take10("top_sign_consistency_ranked")
        tds = decision_metadata.get("top_dominant_sign_ranked")
        if isinstance(tds, list):
            tds = [str(x) for x in tds[:10]]
        else:
            tds = []
        if not ts and not ta and not tm and not tp and not tr and not tsc and not tds:
            return
        parts = [
            f"top_scores[:10]={ts}",
            f"top_areas[:10]={ta}",
            f"top_p95_inside[:10]={tp}",
            f"top_ring_mean[:10]={tr}",
            f"top_sign_consistency[:10]={tsc if tsc else []}",
            f"top_dominant_sign[:10]={tds}",
        ]
        print("[diag:postprocessing_ranks] " + ", ".join(parts))

    def _save_postprocess_audit(self, pair_id: str, artifacts: PipelineArtifacts) -> None:
        """Write contour candidate audit (CSV + overlay) under repo outs/postprocess_audit/."""
        meta = artifacts.decision_metadata or {}
        if str(meta.get("method", "")) not in {"contour_filter_postprocess", "peak_nms_postprocess"}:
            return
        rows = meta.get("contour_audit_rows")
        specs = meta.get("contour_audit_specs")
        gt_audit = meta.get("gt_audit_rows") or []
        peak_full = meta.get("peak_nms_full_audit_rows") or []
        peak_scored = meta.get("peak_nms_candidate_rows") or []
        if not rows and not specs and not gt_audit and not peak_full and not peak_scored:
            return
        root = _REPO_ROOT / "outs" / "postprocess_audit" / pair_id
        root.mkdir(parents=True, exist_ok=True)
        ins = artifacts.inspected_normalized
        if ins is None:
            ins = artifacts.inspected_aligned
        if ins is None:
            ins = artifacts.inspected_raw
        if ins is None:
            ins = artifacts.inspected_input
        try:
            from visualization.debug import (
                save_contour_audit_csv,
                save_contour_postprocess_audit,
                save_gt_audit_csv,
            )

            if rows:
                save_contour_audit_csv(rows, root / "contour_audit.csv")
            if gt_audit:
                save_gt_audit_csv(gt_audit, root / "gt_audit.csv")
            if str(meta.get("method", "")) == "peak_nms_postprocess":
                from modules.postprocessing.peak_nms_postprocess import save_peak_nms_audit_csv

                if peak_full:
                    save_peak_nms_audit_csv(peak_full, root / "peak_nms_full_audit.csv")
                elif peak_scored:
                    save_peak_nms_audit_csv(peak_scored, root / "peak_nms_candidates.csv")
            if ins is not None and specs:
                save_contour_postprocess_audit(pair_id, np.asarray(ins), specs, root)
            print(f"[audit:postprocess] pair_id={pair_id} dir={root}")
        except Exception as exc:
            print(f"[audit:postprocess] pair_id={pair_id} status=SKIPPED reason={exc}")

    def _print_gt_point_coverage(
        self,
        pair_id: str,
        defect_mask: np.ndarray,
        gt_points_xy: list,
        *,
        radius_px: float = 5.0,
    ) -> None:
        """Summarize GT point coverage vs final defect mask (see ``utils.gt_coverage``)."""
        if not gt_points_xy:
            return
        m = compute_gt_point_coverage_metrics(defect_mask, gt_points_xy, radius_px=radius_px)
        print(
            "[diag:gt_point_coverage] "
            f"pair_id={pair_id} "
            f"gt_total={m.gt_total} "
            f"covered_exact={m.gt_covered_exact} "
            f"covered_within_r{radius_px:g}px={m.gt_covered_within_radius} "
            f"fraction_exact={m.coverage_fraction_exact:.4f} "
            f"fraction_within_r={m.coverage_fraction_within_radius:.4f}"
        )

    def _print_gt_audit(self, pair_id: str, decision_metadata: dict) -> None:
        if str(decision_metadata.get("method", "")) not in {"contour_filter_postprocess", "peak_nms_postprocess"}:
            return
        rows = decision_metadata.get("gt_audit_rows")
        if not rows:
            return
        print(f"[diag:gt_audit] pair_id={pair_id}")
        for r in rows:
            nid = r.get("nearest_candidate_id")
            dist = r.get("distance_px")
            st = r.get("status", "")
            print(
                f"  - defect#{r.get('defect_id')}: nearest_candidate_id={nid}, "
                f"dist_px={dist if dist is None else round(float(dist), 2)}, status={st}, "
                f"inside_contour={r.get('inside_contour')}, inside_bbox={r.get('inside_bbox')}, "
                f"on_raw_mask={r.get('gt_on_threshold_mask_raw')}, on_morph_mask={r.get('gt_on_mask_after_morph')}, "
                f"score={r.get('candidate_score')}, area={r.get('candidate_area')}, "
                f"reject_reason={r.get('reject_reason')!r}, kept_final={r.get('kept_final')}"
            )

    def _print_postprocess_sanity(self, pair_id: str, decision_metadata: dict) -> None:
        if str(decision_metadata.get("method", "")) != "contour_filter_postprocess":
            return
        print(
            "[diag:postprocess_sanity] "
            f"pair_id={pair_id} "
            f"after_score_threshold={decision_metadata.get('num_contours_after_score_threshold', decision_metadata.get('num_kept_contours_before_topk'))} "
            f"after_topk={decision_metadata.get('num_contours_after_topk', decision_metadata.get('num_kept_contours_after_topk'))} "
            f"final_num_contours={decision_metadata.get('final_num_contours', decision_metadata.get('num_kept_contours'))} "
            f"final_num_centers={decision_metadata.get('final_num_centers', decision_metadata.get('num_centers_drawn'))} "
            f"score_threshold_used={decision_metadata.get('score_threshold_used')}"
        )

    def _set_runtime_param(self, cfg_obj, key: str, value) -> None:
        if cfg_obj is None or not hasattr(cfg_obj, "params"):
            return
        if cfg_obj.params is None:
            cfg_obj.params = {}
        cfg_obj.params[key] = value

    def _clear_runtime_param(self, cfg_obj, key: str) -> None:
        if cfg_obj is None or not hasattr(cfg_obj, "params"):
            return
        if cfg_obj.params is None:
            return
        cfg_obj.params.pop(key, None)

    def _resolve_alignment_config(self):
        method = str(self.cfg.choices.alignment)
        if method == "orb_affine":
            return self.cfg.orb_affine_alignment
        if method == "ecc_translation":
            return self.cfg.ecc_translation_alignment
        if method == "ecc_euclidean":
            return self.cfg.ecc_euclidean_alignment
        if method == "ecc_affine":
            return self.cfg.ecc_affine_alignment
        if method == "ecc_affine_projected_euclidean":
            return self.cfg.ecc_affine_projected_euclidean_alignment
        if method == "search_euclidean":
            return self.cfg.search_euclidean_alignment
        return self.cfg.alignment

    def _resolve_comparison_config(self):
        """Config object whose .params are passed to the active comparator (matches choices.comparison)."""
        method = str(self.cfg.choices.comparison)
        if method == "gradient_difference":
            return self.cfg.gradient_difference
        if method == "artifact_residual":
            return self.cfg.artifact_residual
        if method == "ssim_comparator":
            return self.cfg.ssim_comparator
        return self.cfg.comparison

    def _resolve_postprocessing_config(self):
        """Config object whose .params are passed to the active postprocessor (matches choices.postprocessing)."""
        method = str(self.cfg.choices.postprocessing)
        if method == "contour_filter_postprocess":
            return self.cfg.contour_filter_postprocess
        if method == "peak_nms_postprocess":
            return self.cfg.peak_nms_postprocess
        return self.cfg.postprocessing

    def _print_resolved_postprocess_config(self, pair_id: str) -> None:
        mod = str(self.cfg.choices.postprocessing)
        if mod == "contour_filter_postprocess":
            p = self.cfg.contour_filter_postprocess.params
            print(
                "[resolved:postprocess]\n"
                f"  pair_id = {pair_id}\n"
                f"  module = contour_filter_postprocess\n"
                f"  top_k_keep = {p.get('top_k_keep')}\n"
                f"  ranking_mode = {p.get('ranking_mode')!r}\n"
                f"  min_area = {p.get('min_area')}\n"
                f"  max_area = {p.get('max_area')}\n"
                f"  max_aspect_ratio = {p.get('max_aspect_ratio')}\n"
                f"  min_fill_ratio = {p.get('min_fill_ratio')}\n"
                f"  exclude_border_touching = {p.get('exclude_border_touching')}\n"
                f"  border_margin_px = {p.get('border_margin_px')}\n"
                f"  morph_open_kernel = {p.get('morph_open_kernel')}\n"
                f"  morph_open_iterations = {p.get('morph_open_iterations')}\n"
                f"  morph_close_kernel = {p.get('morph_close_kernel')}\n"
                f"  morph_close_iterations = {p.get('morph_close_iterations')}\n"
                f"  min_contour_score = {p.get('min_contour_score')}\n"
                f"  contour_score_threshold_mode = {p.get('contour_score_threshold_mode')!r}\n"
                f"  ring_radius_px = {p.get('ring_radius_px')}\n"
                f"  min_sign_consistency = {p.get('min_sign_consistency')}\n"
                f"  reject_on_low_sign_consistency = {p.get('reject_on_low_sign_consistency', False)}"
            )
        elif mod == "peak_nms_postprocess":
            pp = self.cfg.peak_nms_postprocess
            p = pp.params
            print(
                "[resolved:postprocess]\n"
                f"  pair_id = {pair_id}\n"
                f"  module = peak_nms_postprocess\n"
                f"  gaussian_sigma = {pp.gaussian_sigma}\n"
                f"  peak_min_distance = {pp.peak_min_distance}\n"
                f"  peak_threshold_percentile = {pp.peak_threshold_percentile}\n"
                f"  edge_reject_radius = {pp.edge_reject_radius}\n"
                f"  min_peakness = {pp.min_peakness}\n"
                f"  top_k_keep = {pp.top_k_keep}\n"
                f"  min_best_score = {pp.min_best_score}\n"
                f"  render_radius_px = {pp.render_radius_px}\n"
                f"  case3_return_empty = {pp.case3_return_empty}\n"
                f"  (params overrides) = {p!r}"
            )
        else:
            print(
                "[resolved:postprocess]\n"
                f"  pair_id = {pair_id}\n"
                f"  module = {mod!r}\n"
                f"  (using cfg.postprocessing.params; not contour_filter_postprocess)"
            )

    def _print_resolved_comparison_config(self, pair_id: str) -> None:
        choice = str(self.cfg.choices.comparison)
        if choice == "gradient_difference":
            p = self.cfg.gradient_difference.params
            print(
                "[resolved:comparison] "
                f"pair_id={pair_id} "
                f"comparison_choice={choice!r} "
                f"edge_suppression_enabled={bool(p.get('edge_suppression_enabled', False))} "
                f"edge_percentile={float(p.get('edge_percentile', 85.0))} "
                f"edge_weight_on_edges={float(p.get('edge_weight_on_edges', 0.35))}"
            )
        elif choice == "artifact_residual":
            p = self.cfg.artifact_residual.params
            k = p.get("top_hat_kernel_size", p.get("tophat_kernel_size", 9))
            print(
                "[resolved:comparison] "
                f"pair_id={pair_id} "
                f"comparison_choice={choice!r} "
                f"pre_blur_sigma={float(p.get('pre_blur_sigma', 1.0))} "
                f"top_hat_kernel_size={int(k)} "
                f"top_hat_iterations={int(p.get('top_hat_iterations', 1))} "
                f"combine_mode={p.get('combine_mode', 'max')!r} "
                f"edge_mode={p.get('edge_mode', 'hard')!r} "
                f"edge_percentile={float(p.get('edge_percentile', 90.0))} "
                f"edge_dilate_kernel={int(p.get('edge_dilate_kernel', 5))} "
                f"edge_dilate_iterations={int(p.get('edge_dilate_iterations', 1))} "
                f"edge_weight_on_edges={float(p.get('edge_weight_on_edges', 0.25))} "
                f"debug_save_intermediates={bool(p.get('debug_save_intermediates', True))}"
            )
        else:
            print(
                "[resolved:comparison] "
                f"pair_id={pair_id} "
                f"comparison_choice={choice!r} "
                "edge_suppression_enabled=n/a edge_percentile=n/a edge_weight_on_edges=n/a"
            )

    def _print_edge_suppression_diag(self, pair_id: str, meta: dict) -> None:
        if str(meta.get("method", "")) != "gradient_difference":
            return
        print(
            "[diag:comparison_edge_suppression] "
            f"pair_id={pair_id} "
            f"enabled={bool(meta.get('edge_suppression_enabled', False))} "
            f"strong_edge_fraction={float(meta.get('strong_edge_fraction', 0.0)):.6f} "
            f"mean_before={float(meta.get('anomaly_mean_before_edge_suppression', 0.0)):.6f} "
            f"mean_after={float(meta.get('anomaly_mean_after_edge_suppression', 0.0)):.6f} "
            f"delta={float(meta.get('anomaly_mean_delta_edge_suppression', 0.0)):.6f}"
        )

    def _run_normalization_check(
        self,
        pair_id: str,
        reference_before: np.ndarray,
        inspected_before: np.ndarray,
        reference_after: np.ndarray,
        inspected_after: np.ndarray,
        valid_mask,
        *,
        quiet: bool = False,
    ) -> dict:
        try:
            core_mask = self._build_normalization_core_mask(reference_before.shape, valid_mask)
            before_abs = np.abs(
                np.asarray(reference_before, dtype=np.float32)
                - np.asarray(inspected_before, dtype=np.float32)
            )
            after_abs = np.abs(
                np.asarray(reference_after, dtype=np.float32)
                - np.asarray(inspected_after, dtype=np.float32)
            )
            stats_before = self._compute_masked_map_stats(before_abs, core_mask)
            stats_after = self._compute_masked_map_stats(after_abs, core_mask)
            delta = float(stats_after["mean"] - stats_before["mean"])
            status = "IMPROVED" if stats_after["mean"] < stats_before["mean"] else "WORSE"

            if not quiet:
                print(f"[NORMALIZATION CHECK] pair_id={pair_id}")
                print(
                    f"  before_mean={stats_before['mean']:.4f} "
                    f"after_mean={stats_after['mean']:.4f} "
                    f"delta={delta:.4f}"
                )
                print(
                    f"  before_median={stats_before['median']:.4f} "
                    f"after_median={stats_after['median']:.4f}"
                )
                print(f"  status={status}")

            return {
                "before_mean": float(stats_before["mean"]),
                "after_mean": float(stats_after["mean"]),
                "delta": float(delta),
                "before_median": float(stats_before["median"]),
                "after_median": float(stats_after["median"]),
                "before_std": float(stats_before["std"]),
                "after_std": float(stats_after["std"]),
                "core_pixels_used": int(np.count_nonzero(core_mask)),
                "core_overlap_fraction": float(np.mean(core_mask.astype(np.float32))),
                "status": status,
            }
        except Exception as exc:
            if not quiet:
                print(f"[NORMALIZATION CHECK] pair_id={pair_id} status=SKIPPED reason={exc}")
            return {"status": "SKIPPED", "reason": str(exc)}

    def _build_normalization_core_mask(self, shape: tuple[int, ...], valid_mask) -> np.ndarray:
        if valid_mask is None:
            vm = np.ones(shape, dtype=bool)
        else:
            vm_cand = np.asarray(valid_mask).astype(bool)
            vm = vm_cand if vm_cand.shape == shape else np.ones(shape, dtype=bool)
        core = cv2.erode(vm.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
        if np.any(core):
            return core
        return vm

    def _compute_anomaly_with_comparator(
        self,
        reference_image: np.ndarray,
        inspected_image: np.ndarray,
        valid_mask,
    ) -> np.ndarray:
        comp_cfg = self._resolve_comparison_config()
        self._set_runtime_param(comp_cfg, "valid_mask", valid_mask)
        out = self.comparator.run(reference_image, inspected_image, comp_cfg)
        self._clear_runtime_param(comp_cfg, "valid_mask")
        if isinstance(out, tuple) and len(out) == 2:
            return np.asarray(out[0], dtype=np.float32)
        return np.asarray(out, dtype=np.float32)

    def _compute_masked_map_stats(self, x: np.ndarray, valid_mask) -> dict:
        arr = np.asarray(x, dtype=np.float32)
        if valid_mask is None:
            vals = arr.reshape(-1)
        else:
            vm = np.asarray(valid_mask).astype(bool)
            if vm.shape != arr.shape:
                vals = arr.reshape(-1)
            elif np.any(vm):
                vals = arr[vm]
            else:
                vals = arr.reshape(-1)
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
        }