from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from dd_types import PipelineArtifacts
from utils.ground_truth_defects import get_ground_truth_points_for_pair

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

DEBUG_PLOTS_ORDER = [
    "raw_input_check",
    "alignment_effect",
    "alignment_shift",
    "normalization_effect",
    "normalization_scatter",
    "residual_histogram",
    "comparison_effect",
    "threshold_decision",
    "postprocessing_effect",
    "final_result",
    "alignment_transform_summary",
    "alignment_overlay_rgb",
    "alignment_checkerboard",
    "alignment_difference_triptych",
    "alignment_blink_frame_ref",
    "alignment_blink_frame_inspected",
    "alignment_valid_mask_overlay",
    "search_alignment_summary",
    "coarse_search_scores",
    "refined_search_scores",
    "search_iteration_theta_scores",
    "search_final_theta_neighborhood",
    "search_translation_sensitivity_heatmap",
    "search_edge_distance_diagnostics",
    "best_transform_overlay",
    "best_transform_checkerboard",
    "best_transform_diff_triptych",
    "orb_keypoint_matches",
    "valid_overlap_mask",
    "alignment_border_effect",
    "ssim_map",
    "ssim_threshold_histogram",
    "ssim_local_examples",
    "contour_candidates",
    "contour_area_histogram",
    "contour_boxes_overlay",
]


def get_debug_filename(stage_name: str, ext: str = ".png") -> str:
    if stage_name not in DEBUG_PLOTS_ORDER:
        raise ValueError(f"Unknown debug stage: {stage_name}")
    idx = DEBUG_PLOTS_ORDER.index(stage_name) + 1
    return f"{idx:02d}_{stage_name}{ext}"


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Debug visualization requires 'matplotlib'. Please install a compatible "
            "matplotlib build for your numpy version."
        ) from exc
    return plt


def normalize_for_display(
    img: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    lo = float(np.percentile(arr, p_low))
    hi = float(np.percentile(arr, p_high))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _normalize_signed(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    vmax = float(np.max(np.abs(arr)))
    if vmax <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / vmax, -1.0, 1.0)


def _to_gray(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return arr[..., 0]
        return np.mean(arr[..., :3], axis=-1, dtype=np.float32)
    raise ValueError(f"Unsupported image rank for debug visualization: {arr.ndim}")


def _abs_diff(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is None or b is None:
        return None
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    if aa.shape != bb.shape:
        return None
    return np.abs(aa - bb)


def save_three_panel_image(
    path: str | Path,
    panels: list[tuple[str, np.ndarray | None, str]],
    annotation: str | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt = _get_plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, image, mode) in zip(axes, panels):
        ax.set_title(title)
        ax.axis("off")
        if image is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            continue
        arr = np.asarray(image, dtype=np.float32)
        if mode == "signed":
            ax.imshow(_normalize_signed(arr), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        elif mode == "magma":
            ax.imshow(normalize_for_display(arr), cmap="magma", vmin=0.0, vmax=1.0)
        elif mode == "binary":
            ax.imshow((arr > 0).astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(normalize_for_display(_to_gray(arr)), cmap="gray", vmin=0.0, vmax=1.0)
    if annotation:
        fig.text(0.02, 0.02, annotation, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_histogram_with_threshold(
    path: str | Path,
    values: np.ndarray,
    title: str,
    threshold: float | None = None,
    annotation: str | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    ax.hist(vals, bins=120, color="steelblue", alpha=0.9)
    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"threshold={threshold:.4f}")
        ax.legend(loc="upper right")
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Pixel count")
    if annotation:
        ax.text(0.02, 0.95, annotation, transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_shift_figure(
    path: str | Path,
    inspected: np.ndarray,
    shift_x: float,
    shift_y: float,
    extra_annotation: str | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 6))
    bg = normalize_for_display(_to_gray(inspected))
    h, w = bg.shape
    cx, cy = w / 2.0, h / 2.0
    ax.imshow(bg, cmap="gray")
    ax.arrow(cx, cy, shift_x, shift_y, color="yellow", width=1.5, head_width=8, head_length=10)
    ax.set_title("Estimated alignment shift")
    ax.text(
        0.02,
        0.95,
        f"dx={shift_x:.3f}, dy={shift_y:.3f}" + (f"\n{extra_annotation}" if extra_annotation else ""),
        transform=ax.transAxes,
        color="yellow",
        fontsize=10,
        va="top",
        bbox={"facecolor": "black", "alpha": 0.4, "pad": 4},
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_scatter_with_fit(path: str | Path, x: np.ndarray, y: np.ndarray, gain: float | None, offset: float | None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt = _get_plt()
    xx = np.asarray(x, dtype=np.float32).reshape(-1)
    yy = np.asarray(y, dtype=np.float32).reshape(-1)
    n = min(xx.size, yy.size)
    xx = xx[:n]
    yy = yy[:n]
    if n > 20000:
        idx = np.random.default_rng(0).choice(n, size=20000, replace=False)
        xx = xx[idx]
        yy = yy[idx]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xx, yy, s=2, alpha=0.2, color="tab:blue", rasterized=True)
    if gain is not None and offset is not None:
        lx = np.array([float(np.min(xx)), float(np.max(xx))], dtype=np.float32)
        ly = gain * lx + offset
        ax.plot(lx, ly, color="red", linewidth=2, label=f"y={gain:.3f}x+{offset:.3f}")
        ax.legend(loc="upper left")
    ax.set_title("Normalization scatter (aligned ref vs inspected)")
    ax.set_xlabel("Aligned reference intensity")
    ax.set_ylabel("Inspected intensity")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_mask_overlay(inspected: np.ndarray, mask: np.ndarray, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt = _get_plt()
    ins = normalize_for_display(_to_gray(inspected))
    m = np.asarray(mask).astype(bool)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(ins, cmap="gray")
    overlay = np.zeros((*m.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = m.astype(np.float32) * 0.35
    ax.imshow(overlay)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_orb_keypoint_matches(path: Path, ref: np.ndarray, ins: np.ndarray, alignment_metadata: dict) -> None:
    try:
        import cv2
    except Exception:
        return

    r = normalize_for_display(_to_gray(ref))
    i = normalize_for_display(_to_gray(ins))
    r8 = np.clip(r * 255.0, 0, 255).astype(np.uint8)
    i8 = np.clip(i * 255.0, 0, 255).astype(np.uint8)
    orb = cv2.ORB_create(nfeatures=1000)
    kp_r, des_r = orb.detectAndCompute(r8, None)
    kp_i, des_i = orb.detectAndCompute(i8, None)
    if des_r is None or des_i is None:
        return
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des_r, des_i)
    matches = sorted(matches, key=lambda m: m.distance)[:40]
    vis = cv2.drawMatches(r8, kp_r, i8, kp_i, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(vis, cmap="gray")
    ax.axis("off")
    ax.set_title("ORB keypoint matches")
    txt = (
        f"kp_ref={alignment_metadata.get('num_keypoints_ref', 'NA')}, "
        f"kp_ins={alignment_metadata.get('num_keypoints_inspected', 'NA')}, "
        f"matches_total={alignment_metadata.get('num_matches_total', 'NA')}, "
        f"matches_used={alignment_metadata.get('num_matches_used', 'NA')}, "
        f"inliers={alignment_metadata.get('inlier_count', 'NA')}"
    )
    fig.text(0.01, 0.01, txt, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_valid_overlap_mask(path: Path, valid_mask: np.ndarray, inspected: np.ndarray | None) -> None:
    plt = _get_plt()
    vm = np.asarray(valid_mask).astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].set_title("Valid overlap region after warp")
    axes[0].imshow(vm.astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].axis("off")

    axes[1].set_title("Valid overlap overlay on inspected")
    if inspected is None:
        axes[1].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[1].transAxes)
    else:
        ins = normalize_for_display(_to_gray(inspected))
        axes[1].imshow(ins, cmap="gray")
        overlay = np.zeros((*vm.shape, 4), dtype=np.float32)
        overlay[..., 1] = 1.0
        overlay[..., 3] = vm.astype(np.float32) * 0.25
        axes[1].imshow(overlay)
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_alignment_border_effect(path: Path, aligned_diff: np.ndarray | None, valid_mask: np.ndarray) -> None:
    if aligned_diff is None:
        return
    vm = np.asarray(valid_mask).astype(bool)
    masked = np.asarray(aligned_diff, dtype=np.float32).copy()
    if masked.shape != vm.shape:
        return
    masked[~vm] = 0.0
    save_three_panel_image(
        path,
        [
            ("Post-alignment |diff| (all pixels)", aligned_diff, "magma"),
            ("Valid overlap mask", vm.astype(np.float32), "binary"),
            ("Post-alignment |diff| (valid only)", masked, "magma"),
        ],
    )


def _shared_normalize_pair_for_display(
    a: np.ndarray,
    b: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    combined = np.concatenate([aa.reshape(-1), bb.reshape(-1)])
    lo = float(np.percentile(combined, p_low))
    hi = float(np.percentile(combined, p_high))
    if hi <= lo:
        z = np.zeros_like(aa, dtype=np.float32)
        return z, z.copy(), lo, hi
    a_norm = np.clip((aa - lo) / (hi - lo), 0.0, 1.0)
    b_norm = np.clip((bb - lo) / (hi - lo), 0.0, 1.0)
    return a_norm.astype(np.float32, copy=False), b_norm.astype(np.float32, copy=False), lo, hi


def _save_text_figure(path: Path, lines: list[str], title: str | None = None) -> None:
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    y = 0.95
    if title:
        fig.text(0.02, y, title, fontsize=14, va="top")
        y -= 0.06
    for line in lines:
        fig.text(0.02, y, line, fontsize=11, va="top")
        y -= 0.045
        if y < 0.05:
            break
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_alignment_transform_summary(
    path: Path,
    alignment_metadata: dict | None,
) -> None:
    meta = alignment_metadata or {}
    method = meta.get("method", "NA")
    warp_mode = meta.get("warp_mode")
    warp_matrix = meta.get("warp_matrix")
    affine_matrix = meta.get("affine_matrix")
    fallback_used = meta.get("fallback_used")
    valid_pixel_fraction = meta.get("valid_pixel_fraction")
    ecc_correlation = meta.get("ecc_correlation")
    ecc_converged = meta.get("ecc_converged")
    translation_x = meta.get("translation_x", meta.get("shift_x", None))
    translation_y = meta.get("translation_y", meta.get("shift_y", None))
    rotation_degrees_estimated = meta.get("rotation_degrees_estimated", None)
    peak_response = meta.get("peak_response", None)
    warp_convention_note = meta.get("warp_convention_note", None)

    lines: list[str] = [
        f"method: {method}",
    ]
    if warp_mode is not None:
        lines.append(f"warp_mode: {int(warp_mode)}")
    if translation_x is not None and translation_y is not None:
        lines.append(f"translation_x: {float(translation_x):.4f}")
        lines.append(f"translation_y: {float(translation_y):.4f}")
    if rotation_degrees_estimated is not None:
        lines.append(f"rotation_deg: {float(rotation_degrees_estimated):.4f}")
    if peak_response is not None:
        lines.append(f"phase peak_response: {float(peak_response):.6f}")
    if ecc_correlation is not None:
        lines.append(f"ecc_correlation: {float(ecc_correlation):.6f}")
    if ecc_converged is not None:
        lines.append(f"ecc_converged: {bool(ecc_converged)}")
    if fallback_used is not None:
        lines.append(f"fallback_used: {bool(fallback_used)}")
    if valid_pixel_fraction is not None:
        lines.append(f"valid_pixel_fraction: {float(valid_pixel_fraction):.4f}")

    if warp_matrix is not None:
        try:
            wm = np.asarray(warp_matrix, dtype=np.float32)
            if wm.shape == (2, 3):
                lines.append("warp_matrix:")
                lines.append(
                    f"  [{wm[0,0]:.6f} {wm[0,1]:.6f} {wm[0,2]:.6f}]"
                )
                lines.append(
                    f"  [{wm[1,0]:.6f} {wm[1,1]:.6f} {wm[1,2]:.6f}]"
                )
            else:
                lines.append(f"warp_matrix_shape: {wm.shape}")
        except Exception:
            lines.append("warp_matrix: <unavailable>")
    elif affine_matrix is not None:
        try:
            am = np.asarray(affine_matrix, dtype=np.float32)
            if am.shape == (2, 3):
                lines.append("affine_matrix:")
                lines.append(f"  [{am[0,0]:.6f} {am[0,1]:.6f} {am[0,2]:.6f}]")
                lines.append(f"  [{am[1,0]:.6f} {am[1,1]:.6f} {am[1,2]:.6f}]")
            else:
                lines.append(f"affine_matrix_shape: {am.shape}")
        except Exception:
            lines.append("affine_matrix: <unavailable>")

    if warp_convention_note:
        lines.append(f"warp_convention_note: {warp_convention_note}")

    _save_text_figure(path, lines, title="Estimated alignment transform")


def _save_search_alignment_summary(
    path: Path,
    alignment_metadata: dict | None,
) -> None:
    meta = alignment_metadata or {}
    method = meta.get("method", "NA")

    best_theta_deg = meta.get("best_theta_deg")
    best_tx = meta.get("best_tx")
    best_ty = meta.get("best_ty")
    best_score = meta.get("best_score")
    best_stage = meta.get("best_stage")

    coarse_total = meta.get("coarse_candidates_total")
    coarse_valid = meta.get("coarse_candidates_valid")
    refined_total = meta.get("refined_candidates_total")
    refined_valid = meta.get("refined_candidates_valid")
    max_iterations = meta.get("max_iterations")
    iterations_used = meta.get("iterations_used")
    converged = meta.get("converged")
    tol_theta = meta.get("convergence_tol_theta_deg")
    tol_tx = meta.get("convergence_tol_tx")
    tol_ty = meta.get("convergence_tol_ty")

    valid_pixel_fraction = meta.get("valid_pixel_fraction")
    min_valid_overlap_fraction = meta.get("min_valid_overlap_fraction")
    coarse_best_score = meta.get("coarse_best_score")
    refined_best_score = meta.get("refined_best_score")

    lines: list[str] = [
        f"method: {method}",
    ]
    if best_stage is not None:
        lines.append(f"best_stage: {best_stage}")
    if best_theta_deg is not None:
        lines.append(f"best_theta_deg: {float(best_theta_deg):.4f}")
    if best_tx is not None and best_ty is not None:
        lines.append(f"best_tx: {float(best_tx):.4f}")
        lines.append(f"best_ty: {float(best_ty):.4f}")
    if best_score is not None:
        lines.append(f"best_score: {float(best_score):.6f}")

    if coarse_total is not None:
        lines.append(f"coarse_candidates_total: {int(coarse_total)}")
    if coarse_valid is not None:
        lines.append(f"coarse_candidates_valid: {int(coarse_valid)}")
    if refined_total is not None:
        lines.append(f"refined_candidates_total: {int(refined_total)}")
    if refined_valid is not None:
        lines.append(f"refined_candidates_valid: {int(refined_valid)}")
    if max_iterations is not None:
        lines.append(f"max_iterations: {int(max_iterations)}")
    if iterations_used is not None:
        lines.append(f"iterations_used: {int(iterations_used)}")
    if converged is not None:
        lines.append(f"converged: {bool(converged)}")
    if tol_theta is not None and tol_tx is not None and tol_ty is not None:
        lines.append(
            f"convergence_tols: theta={float(tol_theta):.4f}, tx={float(tol_tx):.4f}, ty={float(tol_ty):.4f}"
        )

    if min_valid_overlap_fraction is not None:
        lines.append(f"min_valid_overlap_fraction: {float(min_valid_overlap_fraction):.4f}")
    if valid_pixel_fraction is not None:
        lines.append(f"valid_pixel_fraction: {float(valid_pixel_fraction):.4f}")

    if coarse_best_score is not None and np.isfinite(float(coarse_best_score)):
        lines.append(f"coarse_best_score: {float(coarse_best_score):.6f}")
    if refined_best_score is not None and np.isfinite(float(refined_best_score)):
        lines.append(f"refined_best_score: {float(refined_best_score):.6f}")

    _save_text_figure(path, lines, title="Search alignment summary")


def _save_search_scores_grid(
    path: Path,
    candidate_records: list[dict],
    title: str,
) -> None:
    if not candidate_records:
        return

    plt = _get_plt()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Coarse-to-fine search is now theta-sweep (translation estimated per-theta),
    # so we plot theta vs score directly.
    valid_scores: list[float] = []
    records: list[dict] = []
    for r in candidate_records:
        if "theta_deg" not in r:
            continue
        records.append(r)
        if r.get("valid"):
            s = r.get("score")
            if s is not None:
                try:
                    fs = float(s)
                except Exception:
                    continue
                if np.isfinite(fs):
                    valid_scores.append(fs)

    if not records:
        return

    # Shared scale: valid scores only (keeps plot readable when invalid exist).
    if valid_scores:
        vmin = float(min(valid_scores))
        vmax = float(max(valid_scores))
    else:
        # If nothing is valid, fall back to finite scores.
        finite_scores: list[float] = []
        for r in records:
            s = r.get("score")
            try:
                fs = float(s)
            except Exception:
                continue
            if np.isfinite(fs):
                finite_scores.append(fs)
        if not finite_scores:
            return
        vmin = float(min(finite_scores))
        vmax = float(max(finite_scores))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title)
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("score (gradient MAD)")
    ax.grid(True, alpha=0.2)

    # Scatter valid points (filled) and invalid points (x).
    xs_valid: list[float] = []
    ys_valid: list[float] = []
    cs_valid: list[float] = []
    xs_invalid: list[float] = []
    ys_invalid: list[float] = []

    for r in records:
        theta = float(r.get("theta_deg"))
        score = float(r.get("score", float("nan")))
        if r.get("valid"):
            xs_valid.append(theta)
            ys_valid.append(score)
            cs_valid.append(score)
        else:
            xs_invalid.append(theta)
            ys_invalid.append(score)

    if xs_valid:
        sc = ax.scatter(xs_valid, ys_valid, c=cs_valid, cmap="viridis", vmin=vmin, vmax=vmax, s=50, alpha=0.95, label="valid")
        fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, label="score")
    if xs_invalid:
        ax.scatter(xs_invalid, ys_invalid, c="gray", marker="x", s=70, alpha=0.8, label="invalid")
    ax.legend(loc="best", fontsize=9)

    # Small table-like annotation: theta + estimated translation + overlap + score.
    lines: list[str] = []
    for r in sorted(records, key=lambda x: float(x.get("theta_deg", 0.0))):
        theta = float(r.get("theta_deg"))
        tx = float(r.get("tx", r.get("estimated_tx", 0.0)))
        ty = float(r.get("ty", r.get("estimated_ty", 0.0)))
        ov = r.get("overlap_fraction", None)
        if ov is None:
            ov = float("nan")
        else:
            ov = float(ov)
        score = float(r.get("score", float("nan")))
        valid = bool(r.get("valid", False))
        lines.append(f"theta={theta:+.2f} tx={tx:+.1f} ty={ty:+.1f} ov={ov:.3f} valid={valid}")

    # Place annotation inside axes for compactness.
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        family="monospace",
        bbox={"facecolor": "black", "alpha": 0.35, "pad": 4},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_search_iteration_theta_scores(
    path: Path,
    candidate_records: list[dict],
    iteration_summaries: list[dict],
) -> None:
    if not candidate_records:
        return
    plt = _get_plt()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Per-iteration theta vs score")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.2)

    rec_by_iter: dict[int, list[dict]] = {}
    for r in candidate_records:
        it = int(r.get("iteration", -1))
        if it < 0:
            continue
        rec_by_iter.setdefault(it, []).append(r)
    if not rec_by_iter:
        plt.close(fig)
        return

    for it in sorted(rec_by_iter.keys()):
        recs = sorted(rec_by_iter[it], key=lambda rr: float(rr.get("theta_deg", 0.0)))
        xs = [float(rr.get("theta_deg", 0.0)) for rr in recs]
        ys = [float(rr.get("score", np.nan)) for rr in recs]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"iter {it}")

    for s in iteration_summaries or []:
        it = int(s.get("iteration", -1))
        theta = s.get("chosen_theta_deg")
        score = s.get("best_score", s.get("score"))
        if theta is None or score is None:
            continue
        ax.scatter([float(theta)], [float(score)], marker="*", s=150, color="black")
        ax.text(float(theta), float(score), f"  win {it}", fontsize=8, va="bottom")

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_search_final_theta_neighborhood(
    path: Path,
    neighborhood_records: list[dict],
) -> None:
    if not neighborhood_records:
        return
    plt = _get_plt()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recs = sorted(neighborhood_records, key=lambda r: float(r.get("theta_deg", 0.0)))

    xs = np.asarray([float(r.get("theta_deg", 0.0)) for r in recs], dtype=np.float32)
    ys = np.asarray([float(r.get("score", np.nan)) for r in recs], dtype=np.float32)
    txs = np.asarray([float(r.get("tx", 0.0)) for r in recs], dtype=np.float32)
    tys = np.asarray([float(r.get("ty", 0.0)) for r in recs], dtype=np.float32)

    finite = np.isfinite(ys)
    if ys.size == 0 or not np.any(finite):
        return
    best_idx = int(np.nanargmin(ys))
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_title("Final-theta neighborhood diagnostic")
    ax1.plot(xs, ys, "-o", color="tab:blue", label="score")
    ax1.scatter([xs[best_idx]], [ys[best_idx]], marker="*", s=160, color="black", label="best")
    ax1.set_xlabel("theta (deg)")
    ax1.set_ylabel("score", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(xs, txs, "--", color="tab:orange", linewidth=1.2, label="tx")
    ax2.plot(xs, tys, ":", color="tab:green", linewidth=1.2, label="ty")
    ax2.set_ylabel("estimated translation (px)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_search_translation_sensitivity_heatmap(
    path: Path,
    records: list[dict],
) -> None:
    if not records:
        return
    plt = _get_plt()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    txs = sorted({float(r.get("tx", 0.0)) for r in records})
    tys = sorted({float(r.get("ty", 0.0)) for r in records})
    if not txs or not tys:
        return
    grid = np.full((len(tys), len(txs)), np.nan, dtype=np.float32)
    tx_to_j = {v: j for j, v in enumerate(txs)}
    ty_to_i = {v: i for i, v in enumerate(tys)}
    for r in records:
        i = ty_to_i.get(float(r.get("ty", 0.0)))
        j = tx_to_j.get(float(r.get("tx", 0.0)))
        if i is None or j is None:
            continue
        grid[i, j] = float(r.get("score", np.nan))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap="viridis", origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="score")
    ax.set_title("Translation sensitivity at final theta")
    ax.set_xlabel("tx (px)")
    ax.set_ylabel("ty (px)")
    x_ticks = np.linspace(0, max(len(txs) - 1, 0), min(len(txs), 7), dtype=int)
    y_ticks = np.linspace(0, max(len(tys) - 1, 0), min(len(tys), 7), dtype=int)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f"{txs[j]:+.2f}" for j in x_ticks], rotation=30, ha="right")
    ax.set_yticklabels([f"{tys[i]:+.2f}" for i in y_ticks])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_search_edge_distance_diagnostics(
    path: Path,
    inspected_edge_map: np.ndarray | None,
    warped_reference_edge_map: np.ndarray | None,
    inspected_edge_distance_map: np.ndarray | None,
) -> None:
    if inspected_edge_map is None or warped_reference_edge_map is None:
        return
    plt = _get_plt()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ins_e = np.asarray(inspected_edge_map).astype(np.float32)
    ref_e = np.asarray(warped_reference_edge_map).astype(np.float32)
    dist = None if inspected_edge_distance_map is None else np.asarray(inspected_edge_distance_map, dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(ins_e, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Inspected edge map")
    axes[0].axis("off")

    axes[1].imshow(ref_e, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Warped reference edge map (best)")
    axes[1].axis("off")

    if dist is None:
        axes[2].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[2].transAxes)
    else:
        axes[2].imshow(normalize_for_display(dist), cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("Inspected edge distance transform")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_alignment_overlay_rgb(
    path: Path,
    aligned_ref: np.ndarray,
    inspected: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> None:
    aligned_ref_g = _to_gray(aligned_ref)
    ins_g = _to_gray(inspected)
    ref_norm, ins_norm, _, _ = _shared_normalize_pair_for_display(aligned_ref_g, ins_g)

    vm = None
    if valid_mask is not None:
        vm = np.asarray(valid_mask).astype(bool)
        if vm.shape != ref_norm.shape:
            vm = None

    # Important: do not mask/dim the inspected image panel.
    # The inspected panel should always show the true inspected data so that
    # any border wedges are attributable to the reference warp only.
    if vm is not None:
        vm_f = vm.astype(np.float32)
        ref_norm_plot = ref_norm * vm_f
        ins_norm_plot = ins_norm
    else:
        ref_norm_plot = ref_norm
        ins_norm_plot = ins_norm

    overlay = np.zeros((*ref_norm.shape, 3), dtype=np.float32)
    overlay[..., 0] = ref_norm_plot  # R (optionally masked)
    overlay[..., 1] = ins_norm_plot  # G (never masked)
    overlay[..., 2] = 0.0  # B

    if vm is not None:
        # Subtle green tint for valid overlap (keeps invalid regions readable).
        overlay[..., 1] = np.clip(overlay[..., 1] + 0.10 * vm.astype(np.float32), 0.0, 1.0)

    plt = _get_plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(ref_norm_plot, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Aligned reference (R channel)")
    axes[0].axis("off")

    axes[1].imshow(ins_norm_plot, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Inspected image (G channel)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay RGB (R=ref, G=ins, B=0)")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_checkerboard(ref_norm: np.ndarray, ins_norm: np.ndarray, tile_size: int) -> np.ndarray:
    h, w = ref_norm.shape
    ty = np.arange(h, dtype=np.int32) // int(tile_size)
    tx = np.arange(w, dtype=np.int32) // int(tile_size)
    take_ref = (ty[:, None] + tx[None, :]) % 2 == 0
    return np.where(take_ref, ref_norm, ins_norm).astype(np.float32, copy=False)


def _save_alignment_checkerboard(
    path: Path,
    aligned_ref: np.ndarray,
    inspected: np.ndarray,
) -> None:
    aligned_ref_g = _to_gray(aligned_ref)
    ins_g = _to_gray(inspected)
    ref_norm, ins_norm, _, _ = _shared_normalize_pair_for_display(aligned_ref_g, ins_g)

    coarse_tile = 80
    fine_tile = 24
    coarse = _make_checkerboard(ref_norm, ins_norm, tile_size=coarse_tile)
    fine = _make_checkerboard(ref_norm, ins_norm, tile_size=fine_tile)

    plt = _get_plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(coarse, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Checkerboard coarse (tile={coarse_tile}px)")
    axes[0].axis("off")

    axes[1].imshow(fine, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Checkerboard fine (tile={fine_tile}px)")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_alignment_difference_triptych(
    path: Path,
    pre_diff: np.ndarray,
    aligned_diff: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> None:
    before = np.asarray(pre_diff, dtype=np.float32)
    after = np.asarray(aligned_diff, dtype=np.float32)

    vm = None
    if valid_mask is not None:
        vm = np.asarray(valid_mask).astype(bool)
        if vm.shape != before.shape:
            vm = None

    if vm is not None and vm.any():
        before_disp = before.copy()
        after_disp = after.copy()
        before_disp[~vm] = 0.0
        after_disp[~vm] = 0.0
        before_stats = before[vm]
        after_stats = after[vm]
    else:
        before_disp = before
        after_disp = after
        before_stats = before.reshape(-1)
        after_stats = after.reshape(-1)

    combined = np.concatenate([before_stats.reshape(-1), after_stats.reshape(-1)])
    lo = float(np.percentile(combined, 1.0))
    hi = float(np.percentile(combined, 99.0))
    if hi <= lo:
        before_norm = np.zeros_like(before_disp, dtype=np.float32)
        after_norm = np.zeros_like(after_disp, dtype=np.float32)
    else:
        before_norm = np.clip((before_disp - lo) / (hi - lo), 0.0, 1.0)
        after_norm = np.clip((after_disp - lo) / (hi - lo), 0.0, 1.0)

    improvement = before_disp - after_disp
    improvement_pos = np.maximum(improvement, 0.0)
    if vm is not None and vm.any():
        imp_stats = improvement_pos[vm].reshape(-1)
    else:
        imp_stats = improvement_pos.reshape(-1)
    if imp_stats.size == 0:
        imp_norm = np.zeros_like(improvement_pos, dtype=np.float32)
    else:
        imp_lo = float(np.percentile(imp_stats, 1.0))
        imp_hi = float(np.percentile(imp_stats, 99.0))
        if imp_hi <= imp_lo:
            imp_norm = np.zeros_like(improvement_pos, dtype=np.float32)
        else:
            imp_norm = np.clip((improvement_pos - imp_lo) / (imp_hi - imp_lo), 0.0, 1.0)

    plt = _get_plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(before_norm, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0].set_title("Abs diff BEFORE alignment")
    axes[0].axis("off")

    axes[1].imshow(after_norm, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("Abs diff AFTER alignment")
    axes[1].axis("off")

    axes[2].imshow(imp_norm, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("Improvement (bright = alignment helped)")
    axes[2].axis("off")

    if vm is not None:
        fig.text(
            0.02,
            0.01,
            "Masked diffs to valid overlap for fair comparison.",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_blink_frame(path: Path, image_norm: np.ndarray, title: str) -> None:
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.asarray(image_norm, dtype=np.float32), cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_alignment_blink_frames(
    ref_path: Path,
    inspected_path: Path,
    aligned_ref: np.ndarray,
    inspected: np.ndarray,
) -> None:
    aligned_ref_g = _to_gray(aligned_ref)
    ins_g = _to_gray(inspected)
    ref_norm, ins_norm, _, _ = _shared_normalize_pair_for_display(aligned_ref_g, ins_g)

    _save_blink_frame(ref_path, ref_norm, "Aligned reference")
    _save_blink_frame(inspected_path, ins_norm, "Inspected image")


def _save_alignment_valid_mask_overlay(
    path: Path,
    inspected: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    ins_g = _to_gray(inspected)
    ins_norm, _, _, _ = _shared_normalize_pair_for_display(ins_g, ins_g)
    vm = np.asarray(valid_mask).astype(bool)

    plt = _get_plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(ins_norm, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Inspected (aligned stage)")
    axes[0].axis("off")

    axes[1].imshow(vm.astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Valid overlap mask")
    axes[1].axis("off")

    overlay_rgb = np.zeros((*ins_norm.shape, 3), dtype=np.float32)
    overlay_rgb[..., 0] = ins_norm  # R
    overlay_rgb[..., 1] = ins_norm  # G
    overlay_rgb[..., 2] = ins_norm  # B

    # Light green tint for valid pixels.
    alpha = 0.35
    overlay_rgb[..., 1] = overlay_rgb[..., 1] * (1.0 - alpha) + alpha * vm.astype(np.float32)
    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Inspected + valid overlay")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_ssim_map(path: Path, ssim_map: np.ndarray, anomaly_map: np.ndarray, global_ssim: float | None) -> None:
    annotation = f"global_ssim_score={global_ssim:.6f}" if global_ssim is not None else None
    save_three_panel_image(
        path,
        [
            ("SSIM similarity map (brighter = more similar)", ssim_map, "gray"),
            ("Anomaly map = 1 - SSIM (brighter = more anomalous)", anomaly_map, "magma"),
            ("Comparator added signal", np.asarray(anomaly_map, dtype=np.float32), "magma"),
        ],
        annotation=annotation,
    )


def _save_ssim_threshold_histogram(
    path: Path,
    anomaly_map: np.ndarray,
    threshold_value: float | None,
    positive_pct: float | None,
) -> None:
    ann = None
    if threshold_value is not None and positive_pct is not None:
        ann = f"threshold={threshold_value:.6f}, positive={positive_pct:.4f}%"
    save_histogram_with_threshold(
        path,
        values=anomaly_map,
        title="SSIM anomaly histogram with threshold",
        threshold=threshold_value,
        annotation=ann,
    )


def _save_ssim_local_examples(path: Path, ref: np.ndarray, ins: np.ndarray, anomaly: np.ndarray) -> None:
    arr = np.asarray(anomaly, dtype=np.float32)
    if arr.ndim != 2:
        return
    h, w = arr.shape
    if h < 16 or w < 16:
        return
    k = 4
    flat_idx = np.argpartition(arr.reshape(-1), -k)[-k:]
    coords = [np.unravel_index(int(idx), arr.shape) for idx in flat_idx]
    coords.append((h // 2, w // 2))
    patch = 24
    plt = _get_plt()
    fig, axes = plt.subplots(len(coords), 3, figsize=(9, 3 * len(coords)))
    refg = _to_gray(ref)
    insg = _to_gray(ins)
    for r, (yy, xx) in enumerate(coords):
        y0 = max(0, yy - patch // 2)
        y1 = min(h, y0 + patch)
        x0 = max(0, xx - patch // 2)
        x1 = min(w, x0 + patch)
        refp = normalize_for_display(refg[y0:y1, x0:x1])
        insp = normalize_for_display(insg[y0:y1, x0:x1])
        anp = normalize_for_display(arr[y0:y1, x0:x1])
        axes[r, 0].imshow(refp, cmap="gray")
        axes[r, 1].imshow(insp, cmap="gray")
        axes[r, 2].imshow(anp, cmap="magma")
        axes[r, 0].set_title(f"Ref patch ({x0},{y0})")
        axes[r, 1].set_title("Inspected patch")
        axes[r, 2].set_title("Anomaly patch")
        for c in range(3):
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_contour_candidates(
    path: Path,
    raw_mask: np.ndarray,
    metadata: dict,
    mask_final: np.ndarray | None = None,
) -> None:
    try:
        import cv2
    except Exception:
        return
    from modules.postprocessing.contour_filter_postprocess import (
        apply_pre_contour_morphology,
        contour_keep_decision,
    )

    raw = np.asarray(raw_mask).astype(bool).astype(np.uint8) * 255
    mok = int(metadata.get("morph_open_kernel", 0))
    moi = int(metadata.get("morph_open_iterations", 0))
    mck = int(metadata.get("morph_close_kernel", 0))
    mci = int(metadata.get("morph_close_iterations", 0))
    morphed = apply_pre_contour_morphology(
        raw,
        morph_open_kernel=mok,
        morph_open_iterations=moi,
        morph_close_kernel=mck,
        morph_close_iterations=mci,
    )
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_draw = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(all_draw, contours, -1, (0, 255, 255), 1)

    kept_draw = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)
    min_area = float(metadata.get("min_area", 1.0))
    max_area = metadata.get("max_area", None)
    max_area_f = None if max_area is None else float(max_area)
    mar = metadata.get("max_aspect_ratio", None)
    max_aspect_ratio_f = None if mar is None else float(mar)
    mfr = metadata.get("min_fill_ratio", None)
    min_fill_ratio_f = None if mfr is None else float(mfr)
    exclude_border = bool(metadata.get("exclude_border_touching", False))
    border_margin_px = int(metadata.get("border_margin_px", 2))
    img_h, img_w = int(raw.shape[0]), int(raw.shape[1])

    kept_geo = 0
    for cnt in contours:
        ok, _ = contour_keep_decision(
            cnt,
            img_h,
            img_w,
            min_area=min_area,
            max_area_f=max_area_f,
            max_aspect_ratio=max_aspect_ratio_f,
            min_fill_ratio=min_fill_ratio_f,
            exclude_border_touching=exclude_border,
            border_margin_px=border_margin_px,
        )
        if not ok:
            continue
        cv2.drawContours(kept_draw, [cnt], -1, (0, 255, 0), 1)
        kept_geo += 1

    plt = _get_plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw, cmap="gray")
    axes[0].set_title("Raw threshold mask (pre-morph)")
    axes[1].imshow(all_draw)
    axes[1].set_title("After morph: all external contours")
    if mask_final is not None:
        final_u8 = np.asarray(mask_final).astype(bool).astype(np.uint8) * 255
        final_draw = cv2.cvtColor(final_u8, cv2.COLOR_GRAY2BGR)
        fc, _ = cv2.findContours(final_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(final_draw, fc, -1, (0, 255, 0), 2)
        axes[2].imshow(final_draw)
        axes[2].set_title("Final kept (postprocessor output / top-K)")
    else:
        axes[2].imshow(kept_draw)
        axes[2].set_title("Geometric filter survivors (no final mask passed)")
    for ax in axes:
        ax.axis("off")
    rc = metadata.get("reject_counts", {})
    rc_txt = ", ".join(f"{k}={rc.get(k, 0)}" for k in ("min_area", "max_area", "aspect_ratio", "fill_ratio", "border_touch", "degenerate_bbox"))
    nb = metadata.get("num_kept_contours_before_topk", kept_geo)
    na = metadata.get("num_kept_contours_after_topk", metadata.get("num_contours_kept", kept_geo))
    ts = metadata.get("top_contour_scores") or []
    ts5 = ts[:5] if isinstance(ts, list) else []
    pbm = metadata.get("positive_pixels_before_morph")
    pam = metadata.get("positive_pixels_after_morph")
    morph_px = ""
    if pbm is not None and pam is not None:
        morph_px = f"morph_pixels before/after={int(pbm)}/{int(pam)} | "
    fig.text(
        0.02,
        0.02,
        f"{morph_px}"
        f"total={metadata.get('num_contours_total', len(contours))}, "
        f"after_geom={nb}, after_topk={na}, meta_kept={metadata.get('num_contours_kept', 'NA')} | "
        f"top_scores[:5]={ts5} | rejected: {rc_txt}",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_contour_area_histogram(path: Path, metadata: dict) -> None:
    areas = metadata.get("contour_areas_all", [])
    vals = np.asarray(areas, dtype=np.float32)
    if vals.size == 0:
        return
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals, bins=50, color="steelblue", alpha=0.9)
    min_area = metadata.get("min_area", None)
    max_area = metadata.get("max_area", None)
    if min_area is not None:
        ax.axvline(float(min_area), color="red", linestyle="--", linewidth=2, label=f"min_area={float(min_area):.2f}")
    if max_area is not None:
        ax.axvline(float(max_area), color="orange", linestyle="--", linewidth=2, label=f"max_area={float(max_area):.2f}")
    if min_area is not None or max_area is not None:
        ax.legend(loc="upper right")
    ax.set_title("Contour area histogram")
    ax.set_xlabel("Contour area")
    ax.set_ylabel("Count")
    rc = metadata.get("reject_counts") or {}
    rc_line = ""
    if isinstance(rc, dict) and rc:
        rc_line = " | rejected: " + ", ".join(f"{k}={rc.get(k, 0)}" for k in sorted(rc.keys()))
    rm = metadata.get("ranking_mode", "")
    nb = metadata.get("num_kept_contours_before_topk")
    na = metadata.get("num_kept_contours_after_topk")
    tk = metadata.get("top_k_keep")
    extra = ""
    if rm:
        extra += f" | ranking_mode={rm}"
    if nb is not None and na is not None:
        extra += f" | before_topk={nb} after_topk={na}"
    if tk is not None:
        extra += f" | top_k_keep={tk}"
    ax.text(
        0.02,
        0.95,
        f"kept={metadata.get('num_contours_kept', 'NA')} / total={metadata.get('num_contours_total', 'NA')}{rc_line}{extra}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_contour_boxes_overlay(path: Path, inspected: np.ndarray, metadata: dict) -> None:
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(normalize_for_display(_to_gray(inspected)), cmap="gray")
    boxes = metadata.get("bounding_boxes", [])
    for b in boxes:
        x, y, w, h = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=1.5)
        ax.add_patch(rect)
    ax.set_title("Accepted contour boxes on inspected image")
    ax.text(
        0.02,
        0.95,
        f"kept contours={metadata.get('num_contours_kept', len(boxes))}",
        transform=ax.transAxes,
        va="top",
        color="lime",
        fontsize=9,
        bbox={"facecolor": "black", "alpha": 0.4, "pad": 3},
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_gt_audit_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Ground-truth vs nearest post-geom candidate audit (one row per GT defect)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "defect_id",
        "gt_x",
        "gt_y",
        "nearest_candidate_id",
        "distance_px",
        "inside_contour",
        "inside_bbox",
        "gt_on_threshold_mask_raw",
        "gt_on_mask_after_morph",
        "candidate_area",
        "candidate_score",
        "sign_consistency",
        "reject_reason",
        "kept_final",
        "status",
    ]
    if not rows:
        path.write_text(",".join(fieldnames) + "\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({
                "defect_id": r.get("defect_id"),
                "gt_x": r.get("gt_x"),
                "gt_y": r.get("gt_y"),
                "nearest_candidate_id": r.get("nearest_candidate_id"),
                "distance_px": r.get("distance_px"),
                "inside_contour": r.get("inside_contour"),
                "inside_bbox": r.get("inside_bbox"),
                "gt_on_threshold_mask_raw": r.get("gt_on_threshold_mask_raw"),
                "gt_on_mask_after_morph": r.get("gt_on_mask_after_morph"),
                "candidate_area": r.get("candidate_area"),
                "candidate_score": r.get("candidate_score"),
                "sign_consistency": r.get("sign_consistency"),
                "reject_reason": r.get("reject_reason"),
                "kept_final": r.get("kept_final"),
                "status": r.get("status"),
            })


def save_contour_audit_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """One row per geometrically valid scored contour; columns match postprocess audit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id",
        "area",
        "score",
        "kept_final",
        "reject_reason",
        "reject_stage",
        "mean_inside",
        "p95_inside",
        "ring_mean",
        "sign_consistency",
        "dominant_sign",
    ]
    if not rows:
        path.write_text(",".join(fieldnames) + "\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "candidate_id": r["candidate_id"],
                "area": r["area"],
                "score": r["ranking_score"],
                "kept_final": r["kept_final"],
                "reject_reason": r.get("reject_reason", ""),
                "reject_stage": r.get("reject_stage", ""),
                "mean_inside": r["mean_inside"],
                "p95_inside": r["p95_inside"],
                "ring_mean": r["ring_mean"],
                "sign_consistency": r.get("sign_consistency", ""),
                "dominant_sign": r.get("dominant_sign", ""),
            })


def _draw_gt_markers_matplotlib(
    ax,
    gt_points: list[tuple[int, int]] | None,
    img_w: int,
    img_h: int,
    *,
    color: str = "cyan",
) -> None:
    """Draw crosshairs + 1-based index on an image axis (pixel coords, origin top-left)."""
    if not gt_points:
        return
    c = max(3, min(12, max(img_w, img_h) // 48))
    for i, (x, y) in enumerate(gt_points, start=1):
        xi, yi = int(x), int(y)
        if not (0 <= xi < img_w and 0 <= yi < img_h):
            continue
        ax.plot([xi - c, xi + c], [yi, yi], color=color, linewidth=1.25, zorder=20)
        ax.plot([xi, xi], [yi - c, yi + c], color=color, linewidth=1.25, zorder=20)
        ax.text(
            min(img_w - 1, xi + 4),
            max(0, yi - 4),
            str(i),
            color=color,
            fontsize=7,
            fontweight="bold",
            zorder=21,
            clip_on=True,
        )


def _draw_gt_markers_cv2(
    bgr: np.ndarray,
    gt_points: list[tuple[int, int]] | None,
) -> None:
    """Cyan-like crosshairs in BGR (255,255,0) + small labels; mutates bgr in place."""
    if not gt_points:
        return
    try:
        import cv2
    except Exception:
        return
    h, w = int(bgr.shape[0]), int(bgr.shape[1])
    c = max(3, min(12, max(w, h) // 48))
    col = (255, 255, 0)  # cyan in BGR
    for i, (x, y) in enumerate(gt_points, start=1):
        xi, yi = int(x), int(y)
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        cv2.line(bgr, (xi - c, yi), (xi + c, yi), col, 1, cv2.LINE_AA)
        cv2.line(bgr, (xi, yi - c), (xi, yi + c), col, 1, cv2.LINE_AA)
        cv2.putText(
            bgr,
            str(i),
            (min(w - 8, xi + 4), max(12, yi - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            col,
            1,
            cv2.LINE_AA,
        )


def save_contour_postprocess_audit(
    pair_id: str,
    inspected: np.ndarray,
    audit_specs: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    gt_points: list[tuple[int, int]] | None = None,
) -> None:
    """
    Overlay all scored candidates on inspected image.
    Green = kept final; yellow = failed score threshold; red = passed score but cut by top-K cap.
    """
    try:
        import cv2
    except Exception:
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = normalize_for_display(_to_gray(np.asarray(inspected, dtype=np.float32)))
    u8 = (np.clip(g, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    colors = {
        "kept": (0, 220, 0),
        "score_threshold": (0, 220, 255),
        "top_k_cap": (0, 60, 255),
    }
    for spec in audit_specs:
        cnt = spec["cnt"]
        status = str(spec.get("status", ""))
        cid = int(spec["candidate_id"])
        col = colors.get(status, (180, 180, 180))
        cv2.drawContours(bgr, [cnt], -1, col, 2)
        M = cv2.moments(cnt)
        if M["m00"] > 1e-6:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
        cv2.putText(
            bgr,
            str(cid),
            (max(0, cx - 6), max(12, cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            col,
            1,
            cv2.LINE_AA,
        )
    pts = gt_points if gt_points is not None else get_ground_truth_points_for_pair(pair_id)
    _draw_gt_markers_cv2(bgr, pts)
    legend_y = 18
    cap = f"{pair_id} green=kept yellow=score_thr red=topk_cap cyan=GT"
    cv2.putText(bgr, cap, (8, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    out_path = out_dir / "contour_audit_overlay.png"
    cv2.imwrite(str(out_path), bgr)


def compute_mean_anomaly(anomaly_map: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    arr = np.asarray(anomaly_map, dtype=np.float32)
    if valid_mask is None:
        return float(np.mean(arr))
    vm = np.asarray(valid_mask).astype(bool)
    if vm.shape != arr.shape or not np.any(vm):
        return float(np.mean(arr))
    return float(np.mean(arr[vm]))


def compute_gain(before_map: np.ndarray, after_map: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    before_mean = compute_mean_anomaly(before_map, valid_mask=valid_mask)
    after_mean = compute_mean_anomaly(after_map, valid_mask=valid_mask)
    return float(before_mean - after_mean)


def _run_comparator_map(
    comparator,
    comparator_cfg,
    reference_image: np.ndarray,
    inspected_image: np.ndarray,
    valid_mask: np.ndarray | None,
) -> np.ndarray:
    if comparator_cfg is not None and hasattr(comparator_cfg, "params"):
        if comparator_cfg.params is None:
            comparator_cfg.params = {}
        comparator_cfg.params["valid_mask"] = valid_mask
    out = comparator.run(reference_image, inspected_image, comparator_cfg)
    if comparator_cfg is not None and hasattr(comparator_cfg, "params") and comparator_cfg.params is not None:
        comparator_cfg.params.pop("valid_mask", None)
    if isinstance(out, tuple) and len(out) == 2:
        return np.asarray(out[0], dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def _gray_with_invalid(
    img: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    invalid_level: float = 0.55,
) -> np.ndarray:
    """Normalize to [0,1] gray; outside valid overlap set to neutral gray."""
    g = normalize_for_display(_to_gray(np.asarray(img, dtype=np.float32)))
    if valid_mask is None:
        return g
    vm = np.asarray(valid_mask).astype(bool)
    if vm.shape != g.shape:
        return g
    return np.where(vm, g, invalid_level).astype(np.float32)


def _ensure_artifact_residual_debug_maps(
    artifacts: PipelineArtifacts,
    comparator,
    comparator_cfg: Any,
) -> dict[str, Any] | None:
    """
    Return artifact_residual intermediate maps from artifacts, or re-run comparator once
    with debug_save_intermediates=True (does not mutate comparator_cfg).
    """
    comp_meta = getattr(artifacts, "comparison_metadata", None) or {}
    is_art = comp_meta.get("method") == "artifact_residual" or getattr(comparator, "name", None) == "artifact_residual"
    if not is_art:
        return None

    inter = getattr(artifacts, "artifact_residual_intermediates", None)
    if isinstance(inter, dict) and "residual_signed" in inter and inter["residual_signed"] is not None:
        return inter

    if comparator_cfg is None:
        return None
    cfg = copy.deepcopy(comparator_cfg)
    if not hasattr(cfg, "params") or cfg.params is None:
        cfg.params = {}
    cfg.params["debug_save_intermediates"] = True
    vm = getattr(artifacts, "valid_mask", None)
    if vm is not None:
        cfg.params["valid_mask"] = np.asarray(vm)

    ref = artifacts.reference_normalized
    ins = artifacts.inspected_normalized
    if ref is None or ins is None:
        return None

    out = comparator.run(
        np.asarray(ref, dtype=np.float32),
        np.asarray(ins, dtype=np.float32),
        cfg,
    )
    if isinstance(out, tuple) and len(out) == 2:
        _amap, meta = out
        dbg = meta.get("artifact_residual_debug_maps") if isinstance(meta, dict) else None
        if isinstance(dbg, dict) and dbg.get("residual_signed") is not None:
            return dbg
    return None


def save_artifact_residual_diagnostic_figure(
    pair_id: str,
    artifacts: PipelineArtifacts,
    comparator,
    comparator_cfg: Any,
    output_path: str | Path,
) -> None:
    """
    3×3 diagnostic figure for artifact_residual: normalized inputs, residuals, top-hat channels,
    edge mask, final anomaly, threshold, postprocess. Saves under repo output paths only.

    Uses ``artifacts.artifact_residual_intermediates`` when present; otherwise re-runs the
    comparator once with ``debug_save_intermediates=True``.
    """
    comp_meta = getattr(artifacts, "comparison_metadata", None) or {}
    if comp_meta.get("method") != "artifact_residual" and getattr(comparator, "name", None) != "artifact_residual":
        return

    dbg = _ensure_artifact_residual_debug_maps(artifacts, comparator, comparator_cfg)
    if dbg is None:
        return

    norm_ins = artifacts.inspected_normalized
    norm_ref = artifacts.reference_normalized
    if norm_ins is None or norm_ref is None:
        return

    vm = None if artifacts.valid_mask is None else np.asarray(artifacts.valid_mask).astype(bool)
    gt_pts = get_ground_truth_points_for_pair(pair_id)

    residual_signed = np.asarray(dbg["residual_signed"], dtype=np.float32)
    enh_pos = np.asarray(dbg["enhanced_positive"], dtype=np.float32)
    enh_neg = np.asarray(dbg["enhanced_negative"], dtype=np.float32)
    edge_mask = np.asarray(dbg.get("edge_mask_dilated", 0.0), dtype=np.float32)

    anomaly_map = artifacts.anomaly_map
    if anomaly_map is None:
        return
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)

    mask_raw = artifacts.binary_mask_raw
    thr_mask = np.asarray(
        mask_raw if mask_raw is not None else np.zeros_like(anomaly_map, dtype=bool),
        dtype=np.float32,
    )

    post_base = np.asarray(norm_ins, dtype=np.float32)
    mask_final = artifacts.binary_mask_final
    post_mask = mask_final if mask_final is not None else (mask_raw if mask_raw is not None else np.zeros_like(post_base, dtype=bool))
    contour_vis, num_contours, total_area = draw_contours_and_centers(post_base, np.asarray(post_mask))

    edge_mode = str(comp_meta.get("edge_mode", "?"))
    frac_edge = float(comp_meta.get("fraction_anomaly_touched_by_edge_mask", 0.0))

    # Signed residual: diverging colormap; invalid overlap → NaN (matplotlib shows as bad color)
    rs = residual_signed.astype(np.float32).copy()
    if vm is not None and rs.shape == vm.shape:
        rs[~vm] = np.nan
    finite = np.isfinite(rs)
    if np.any(finite):
        vmax_abs = float(np.nanmax(np.abs(rs)))
    else:
        vmax_abs = 0.0
    if vmax_abs <= 1e-12:
        rs_disp = np.zeros_like(rs, dtype=np.float32)
    else:
        rs_disp = np.clip(rs / vmax_abs, -1.0, 1.0)
    if vm is not None and rs_disp.shape == vm.shape:
        rs_disp[~vm] = np.nan

    plt = _get_plt()
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    _cms = getattr(plt, "colormaps", None)
    if _cms is not None:
        cmap_div = _cms["coolwarm"].copy()
    else:
        cmap_div = plt.cm.coolwarm.copy()
    cmap_div.set_bad((0.78, 0.78, 0.82, 1.0))

    h0, w0 = int(norm_ins.shape[0]), int(norm_ins.shape[1])

    axes[0, 0].imshow(_gray_with_invalid(norm_ins, vm), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Inspected (normalized)\n(outside overlap = neutral gray)")
    axes[0, 0].axis("off")
    _draw_gt_markers_matplotlib(axes[0, 0], gt_pts, w0, h0)

    axes[0, 1].imshow(_gray_with_invalid(norm_ref, vm), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("Reference (normalized, aligned)")
    axes[0, 1].axis("off")
    _draw_gt_markers_matplotlib(axes[0, 1], gt_pts, w0, h0)

    axes[0, 2].imshow(rs_disp, cmap=cmap_div, vmin=-1.0, vmax=1.0, interpolation="nearest")
    axes[0, 2].set_title("Signed residual (ins − ref)\nscaled to ±1 by |max| in valid overlap")
    axes[0, 2].axis("off")
    _draw_gt_markers_matplotlib(axes[0, 2], gt_pts, residual_signed.shape[1], residual_signed.shape[0])

    def _heatmap_top_hat(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float32).copy()
        if vm is not None and a.shape == vm.shape:
            a[~vm] = np.nan
        if not np.any(np.isfinite(a)):
            return np.zeros_like(a, dtype=np.float32)
        lo = float(np.nanpercentile(a, 1.0))
        hi = float(np.nanpercentile(a, 99.0))
        if hi <= lo:
            out = np.zeros_like(a, dtype=np.float32)
        else:
            out = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
        if vm is not None and out.shape == vm.shape:
            out = np.where(vm, out, 0.35)
        return out

    axes[1, 0].imshow(_heatmap_top_hat(enh_pos), cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Bright residual → white top-hat\n(enhanced Δ+)")
    axes[1, 0].axis("off")
    _draw_gt_markers_matplotlib(axes[1, 0], gt_pts, enh_pos.shape[1], enh_pos.shape[0])

    axes[1, 1].imshow(_heatmap_top_hat(enh_neg), cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title("Dark residual → white top-hat\n(enhanced Δ−)")
    axes[1, 1].axis("off")
    _draw_gt_markers_matplotlib(axes[1, 1], gt_pts, enh_neg.shape[1], enh_neg.shape[0])

    emax = float(np.max(edge_mask)) if edge_mask.size else 0.0
    axes[1, 2].imshow(np.clip(edge_mask, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)
    t_edge = f"Strong-edge mask (dilated)\nmode={edge_mode}, frac(anomaly on mask)={frac_edge:.3f}"
    if emax < 1e-6 and edge_mode == "off":
        t_edge = "Edge mask (edge_mode=off)\n(no suppression)"
    axes[1, 2].set_title(t_edge)
    axes[1, 2].axis("off")
    _draw_gt_markers_matplotlib(axes[1, 2], gt_pts, edge_mask.shape[1], edge_mask.shape[0])

    show_an = normalize_for_display(anomaly_map)
    if vm is not None and show_an.shape == vm.shape:
        show_an = np.where(vm, show_an, 0.35)
    axes[2, 0].imshow(show_an, cmap="magma", vmin=0.0, vmax=1.0)
    ks = int(comp_meta.get("top_hat_kernel_size", comp_meta.get("tophat_kernel_size", 9)))
    nit = int(comp_meta.get("top_hat_iterations", 1))
    axes[2, 0].set_title(f"Final anomaly [0,1]\nartifact_residual k={ks}×{nit}, edge={edge_mode}")
    axes[2, 0].axis("off")
    _draw_gt_markers_matplotlib(axes[2, 0], gt_pts, anomaly_map.shape[1], anomaly_map.shape[0])

    thr_val = None
    if artifacts.thresholding_metadata:
        thr_val = artifacts.thresholding_metadata.get("threshold")
        if thr_val is None:
            thr_val = artifacts.thresholding_metadata.get("threshold_value")
    thr_txt = "NA" if thr_val is None else f"{float(thr_val):.4f}"
    axes[2, 1].imshow(thr_mask, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2, 1].set_title(f"Threshold mask (raw)\nt = {thr_txt}")
    axes[2, 1].axis("off")
    _draw_gt_markers_matplotlib(axes[2, 1], gt_pts, thr_mask.shape[1], thr_mask.shape[0])

    axes[2, 2].imshow(contour_vis)
    axes[2, 2].set_title(f"Postprocess (kept)\nN={num_contours}, area={total_area:.1f}")
    axes[2, 2].axis("off")
    _draw_gt_markers_matplotlib(axes[2, 2], gt_pts, contour_vis.shape[1], contour_vis.shape[0])

    p_low = float(comp_meta.get("norm_percentile_low", 1.0))
    p_high = float(comp_meta.get("norm_percentile_high", 99.0))
    st = (
        f"{pair_id}  |  artifact_residual diagnostics  |  cyan = GT  |  "
        f"norm pct [{p_low},{p_high}]  |  edge p{comp_meta.get('edge_percentile', '?')}"
    )
    fig.suptitle(st, fontsize=10)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_contours_and_centers(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int, float]:
    try:
        import cv2
    except Exception:
        base = np.stack([normalize_for_display(_to_gray(image))] * 3, axis=-1)
        return (base * 255.0).astype(np.uint8), 0, 0.0

    img = normalize_for_display(_to_gray(image))
    rgb = np.stack([img, img, img], axis=-1)
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    m = np.asarray(mask).astype(bool).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = 0.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        total_area += area
        cv2.drawContours(rgb_u8, [cnt], -1, (0, 255, 0), 1)
        mm = cv2.moments(cnt)
        if abs(mm["m00"]) > 1e-12:
            cx = int(mm["m10"] / mm["m00"])
            cy = int(mm["m01"] / mm["m00"])
            cv2.circle(rgb_u8, (cx, cy), 2, (255, 0, 0), -1)

    return rgb_u8, int(len(contours)), float(total_area)


def save_compact_pipeline_figure(
    pair_id: str,
    artifacts: PipelineArtifacts,
    comparator,
    comparator_cfg,
    output_path: str | Path,
) -> None:
    raw_ref = artifacts.reference_raw
    raw_ins = artifacts.inspected_raw
    pre_ref = artifacts.reference_preprocessed
    pre_ins = artifacts.inspected_preprocessed
    aligned_ref = artifacts.reference_aligned
    aligned_ins = artifacts.inspected_aligned
    norm_ref = artifacts.reference_normalized
    norm_ins = artifacts.inspected_normalized
    anomaly_map = artifacts.anomaly_map
    mask_raw = artifacts.binary_mask_raw
    mask_final = artifacts.binary_mask_final
    valid_mask = artifacts.valid_mask

    if (
        raw_ref is None
        or raw_ins is None
        or pre_ref is None
        or pre_ins is None
        or aligned_ref is None
        or aligned_ins is None
        or norm_ref is None
        or norm_ins is None
    ):
        return

    vm = None if valid_mask is None else np.asarray(valid_mask).astype(bool)

    # Same comparator instance/config used in the pipeline.
    before_pre_map = _run_comparator_map(comparator, comparator_cfg, np.asarray(raw_ref, dtype=np.float32), np.asarray(raw_ins, dtype=np.float32), vm)
    after_pre_map = _run_comparator_map(comparator, comparator_cfg, np.asarray(pre_ref, dtype=np.float32), np.asarray(pre_ins, dtype=np.float32), vm)
    gain_pre = compute_gain(before_pre_map, after_pre_map, valid_mask=vm)

    before_norm_map = _run_comparator_map(comparator, comparator_cfg, np.asarray(aligned_ref, dtype=np.float32), np.asarray(aligned_ins, dtype=np.float32), vm)
    after_norm_map = _run_comparator_map(comparator, comparator_cfg, np.asarray(norm_ref, dtype=np.float32), np.asarray(norm_ins, dtype=np.float32), vm)
    gain_norm = compute_gain(before_norm_map, after_norm_map, valid_mask=vm)

    align_meta = artifacts.alignment_metadata or {}
    theta = align_meta.get("rotation_degrees_estimated", align_meta.get("best_theta_deg", 0.0))
    tx = align_meta.get("translation_x", align_meta.get("best_tx", align_meta.get("estimated_tx", align_meta.get("shift_x", 0.0))))
    ty = align_meta.get("translation_y", align_meta.get("best_ty", align_meta.get("estimated_ty", align_meta.get("shift_y", 0.0))))
    overlap = align_meta.get("valid_pixel_fraction", 1.0 if vm is None else float(np.mean(vm.astype(np.float32))))

    threshold_value = None
    if artifacts.thresholding_metadata:
        threshold_value = artifacts.thresholding_metadata.get("threshold")
        if threshold_value is None:
            threshold_value = artifacts.thresholding_metadata.get("threshold_value")
    if threshold_value is None and artifacts.threshold_map is not None:
        threshold_value = float(np.mean(np.asarray(artifacts.threshold_map, dtype=np.float32)))

    overlay_ref, overlay_ins, _, _ = _shared_normalize_pair_for_display(_to_gray(aligned_ref), _to_gray(aligned_ins))
    overlay_rgb = np.zeros((*overlay_ref.shape, 3), dtype=np.float32)
    overlay_rgb[..., 0] = overlay_ref
    overlay_rgb[..., 1] = overlay_ins

    post_base = np.asarray(norm_ins if norm_ins is not None else aligned_ins, dtype=np.float32)
    post_mask = mask_final if mask_final is not None else (mask_raw if mask_raw is not None else np.zeros_like(post_base, dtype=bool))
    contour_vis, num_contours, total_area = draw_contours_and_centers(post_base, np.asarray(post_mask))

    gt_pts = get_ground_truth_points_for_pair(pair_id)

    plt = _get_plt()
    fig, axes = plt.subplots(1, 7, figsize=(21, 4))

    raw_ins_arr = np.asarray(raw_ins, dtype=np.float32)
    axes[0].imshow(normalize_for_display(_to_gray(raw_ins_arr)), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Inspected Image")
    axes[0].axis("off")
    _draw_gt_markers_matplotlib(axes[0], gt_pts, raw_ins_arr.shape[1], raw_ins_arr.shape[0])

    axes[1].imshow(normalize_for_display(_to_gray(np.asarray(pre_ins, dtype=np.float32))), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Preprocessed Image (gain={gain_pre:+.4f})")
    axes[1].axis("off")

    axes[2].imshow(overlay_rgb)
    axes[2].set_title(
        f"Alignment Overlay (theta={float(theta):+.2f}, tx={float(tx):+.2f}, ty={float(ty):+.2f}, overlap={float(overlap):.3f})"
    )
    axes[2].axis("off")

    if anomaly_map is None:
        show_anomaly = after_norm_map
    else:
        show_anomaly = np.asarray(anomaly_map, dtype=np.float32)
    axes[3].imshow(normalize_for_display(show_anomaly), cmap="magma", vmin=0.0, vmax=1.0)
    comp_meta = getattr(artifacts, "comparison_metadata", None) or {}
    if comp_meta.get("method") == "gradient_difference" and "edge_suppression_enabled" in comp_meta:
        es = bool(comp_meta.get("edge_suppression_enabled"))
        edge_suppr = "on" if es else "off"
        frac = float(comp_meta.get("strong_edge_fraction", 0.0))
        ew = float(comp_meta.get("edge_weight_on_edges", 0.0))
        an_title = (
            f"Anomaly (edge_suppr={edge_suppr}, frac={frac:.2f}, w={ew:.2f}, norm_gain={gain_norm:+.4f})"
        )
    elif comp_meta.get("method") == "artifact_residual":
        ks = int(comp_meta.get("top_hat_kernel_size", comp_meta.get("tophat_kernel_size", 9)))
        nit = int(comp_meta.get("top_hat_iterations", 1))
        em = str(comp_meta.get("edge_mode", "off"))
        bmean = float(comp_meta.get("bright_artifact_mean", 0.0))
        dmean = float(comp_meta.get("dark_artifact_mean", 0.0))
        f_edge = float(comp_meta.get("fraction_anomaly_touched_by_edge_mask", 0.0))
        epct = comp_meta.get("edge_percentile", "?")
        an_title = (
            f"Anomaly: artifact_residual  |  top-hat k={ks}×{nit}  |  edge={em}  p={epct}\n"
            f"brightμ={bmean:.4f} darkμ={dmean:.4f}  |  frac on edge-mask={f_edge:.3f}  |  norm_gain={gain_norm:+.4f}"
        )
    else:
        an_title = f"Anomaly (norm_gain={gain_norm:+.4f})"
    axes[3].set_title(an_title, fontsize=8 if comp_meta.get("method") == "artifact_residual" else 10)
    axes[3].axis("off")
    _draw_gt_markers_matplotlib(axes[3], gt_pts, int(show_anomaly.shape[1]), int(show_anomaly.shape[0]))

    thr_mask = np.asarray(mask_raw if mask_raw is not None else np.zeros_like(show_anomaly, dtype=bool)).astype(np.float32)
    thr_txt = "NA" if threshold_value is None else f"{float(threshold_value):.4f}"
    axes[4].imshow(thr_mask, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[4].set_title(f"Thresholded Mask (Pre-Postprocess)\nt={thr_txt}")
    axes[4].axis("off")
    _draw_gt_markers_matplotlib(axes[4], gt_pts, int(thr_mask.shape[1]), int(thr_mask.shape[0]))

    # Final prediction mask (same as DetectionResult.defect_mask / binary_mask_final after postprocess).
    final_mask = mask_final if mask_final is not None else np.zeros_like(post_base, dtype=bool)
    final_u8 = np.asarray(final_mask, dtype=np.float32)
    axes[5].imshow(final_u8, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    axes[5].set_title("Final Binary Defect Mask")
    axes[5].axis("off")

    axes[6].imshow(contour_vis)
    axes[6].set_title(f"Final Detection + Ground Truth\n(N={num_contours}, area={total_area:.1f})")
    axes[6].axis("off")
    _draw_gt_markers_matplotlib(axes[6], gt_pts, int(contour_vis.shape[1]), int(contour_vis.shape[0]))

    st = f"{pair_id} pipeline progression"
    if gt_pts:
        st += "  (cyan = ground-truth defect locations)"
    summary_bits: list[str] = []
    cm = comp_meta.get("method") or getattr(comparator, "name", "")
    if cm:
        summary_bits.append(f"comparator={cm}")
    thr_m = getattr(artifacts, "thresholding_metadata", None) or {}
    if thr_m.get("k_mad") is not None:
        summary_bits.append(f"k_mad={float(thr_m['k_mad']):.2f}")
    if comp_meta.get("method") == "artifact_residual":
        summary_bits.append(
            f"edge={comp_meta.get('edge_mode', '?')}|p{comp_meta.get('edge_percentile', '?')}|k{comp_meta.get('top_hat_kernel_size', comp_meta.get('tophat_kernel_size', '?'))}"
        )
    if summary_bits:
        st += "  |  " + " | ".join(summary_bits)
    fig.suptitle(st, fontsize=10)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Extended artifact_residual diagnostics (separate file; does not change other comparators' outputs).
    if comp_meta.get("method") == "artifact_residual":
        dbg_path = out.parent / f"{pair_id}_artifact_residual_debug.png"
        try:
            save_artifact_residual_diagnostic_figure(
                pair_id=pair_id,
                artifacts=artifacts,
                comparator=comparator,
                comparator_cfg=comparator_cfg,
                output_path=dbg_path,
            )
        except Exception as exc:
            print(f"[artifact_residual_debug] pair_id={pair_id} status=SKIPPED reason={exc}")


def save_stage_visualizations(artifacts: PipelineArtifacts, pair_id: str, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_ref = artifacts.reference_raw
    raw_ins = artifacts.inspected_raw
    pre_ref = artifacts.reference_preprocessed
    pre_ins = artifacts.inspected_preprocessed
    aligned_ref = artifacts.reference_aligned
    aligned_ins = artifacts.inspected_aligned
    norm_ref = artifacts.reference_normalized
    norm_ins = artifacts.inspected_normalized
    anomaly = artifacts.anomaly_map
    threshold_map = artifacts.threshold_map
    mask_raw = artifacts.binary_mask_raw
    mask_final = artifacts.binary_mask_final
    valid_mask = artifacts.valid_mask

    raw_diff = _abs_diff(raw_ref, raw_ins)
    pre_diff = _abs_diff(pre_ref, pre_ins)
    aligned_diff = _abs_diff(aligned_ref, aligned_ins)
    norm_diff = _abs_diff(norm_ref, norm_ins)

    # Core figures 01-10
    save_three_panel_image(
        out / get_debug_filename("raw_input_check"),
        [("Reference", raw_ref, "gray"), ("Inspected", raw_ins, "gray"), ("|Inspected - Reference|", raw_diff, "magma")],
    )

    shift_x = float(artifacts.alignment_metadata.get("shift_x", 0.0)) if artifacts.alignment_metadata else 0.0
    shift_y = float(artifacts.alignment_metadata.get("shift_y", 0.0)) if artifacts.alignment_metadata else 0.0
    alignment_improvement = None if (pre_diff is None or aligned_diff is None or pre_diff.shape != aligned_diff.shape) else pre_diff - aligned_diff
    save_three_panel_image(
        out / get_debug_filename("alignment_effect"),
        [
            ("Pre-alignment |diff|", pre_diff, "magma"),
            ("Post-alignment |diff|", aligned_diff, "magma"),
            ("Improvement (pre - post)", alignment_improvement, "signed"),
        ],
        annotation=f"Estimated shift: dx={shift_x:.3f}, dy={shift_y:.3f}",
    )

    if raw_ins is not None:
        extra_shift_text = None
        if artifacts.alignment_metadata:
            method = str(artifacts.alignment_metadata.get("method", ""))
            if method in {"ecc_translation", "ecc_euclidean"}:
                corr = artifacts.alignment_metadata.get("ecc_correlation")
                rot = artifacts.alignment_metadata.get("rotation_degrees_estimated")
                chunks = []
                if corr is not None:
                    chunks.append(f"ecc_corr={float(corr):.5f}")
                if rot is not None:
                    chunks.append(f"rot={float(rot):.3f} deg")
                extra_shift_text = ", ".join(chunks) if chunks else None
        save_shift_figure(
            out / get_debug_filename("alignment_shift"),
            raw_ins,
            shift_x=shift_x,
            shift_y=shift_y,
            extra_annotation=extra_shift_text,
        )

    gain = artifacts.normalization_metadata.get("gain") if artifacts.normalization_metadata else None
    offset = artifacts.normalization_metadata.get("offset") if artifacts.normalization_metadata else None
    norm_improvement = None if (aligned_diff is None or norm_diff is None or aligned_diff.shape != norm_diff.shape) else aligned_diff - norm_diff
    save_three_panel_image(
        out / get_debug_filename("normalization_effect"),
        [
            ("Before normalization |diff|", aligned_diff, "magma"),
            ("After normalization |diff|", norm_diff, "magma"),
            ("Improvement (before - after)", norm_improvement, "signed"),
        ],
        annotation=(f"gain={float(gain):.4f}, offset={float(offset):.4f}" if gain is not None and offset is not None else None),
    )

    if aligned_ref is not None and aligned_ins is not None:
        save_scatter_with_fit(
            out / get_debug_filename("normalization_scatter"),
            x=aligned_ref,
            y=aligned_ins,
            gain=float(gain) if gain is not None else None,
            offset=float(offset) if offset is not None else None,
        )

    if norm_ref is not None and norm_ins is not None:
        residual = np.asarray(norm_ins, dtype=np.float32) - np.asarray(norm_ref, dtype=np.float32)
        save_histogram_with_threshold(
            out / get_debug_filename("residual_histogram"),
            values=residual,
            title="Residual histogram (inspected - normalized reference)",
            threshold=0.0,
            annotation=f"mean residual = {float(np.mean(residual)):.6f}",
        )

    abs_diff_component = _abs_diff(norm_ref, norm_ins)
    comparator_added_signal = (
        None
        if (anomaly is None or abs_diff_component is None or anomaly.shape != abs_diff_component.shape)
        else np.asarray(anomaly, dtype=np.float32) - np.asarray(abs_diff_component, dtype=np.float32)
    )
    save_three_panel_image(
        out / get_debug_filename("comparison_effect"),
        [
            ("Raw |diff| component", abs_diff_component, "magma"),
            ("Anomaly map (brighter = more anomalous)", anomaly, "magma"),
            ("Comparator added signal", comparator_added_signal, "signed"),
        ],
    )

    threshold_value = float(np.mean(np.asarray(threshold_map, dtype=np.float32))) if threshold_map is not None else None
    threshold_positive_pct = (
        float(np.mean(np.asarray(mask_raw, dtype=np.float32)) * 100.0)
        if mask_raw is not None and np.asarray(mask_raw).size > 0
        else None
    )
    plt = _get_plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].set_title("Anomaly map (brighter = more anomalous)")
    axes[0].axis("off")
    if anomaly is not None:
        axes[0].imshow(normalize_for_display(np.asarray(anomaly, dtype=np.float32)), cmap="magma", vmin=0.0, vmax=1.0)
    else:
        axes[0].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[0].transAxes)
    axes[1].set_title("Raw binary threshold mask")
    axes[1].axis("off")
    if mask_raw is not None:
        axes[1].imshow(np.asarray(mask_raw).astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
    else:
        axes[1].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[1].transAxes)
    axes[2].set_title("Anomaly histogram with threshold")
    if anomaly is not None:
        vals = np.asarray(anomaly, dtype=np.float32).reshape(-1)
        axes[2].hist(vals, bins=120, color="steelblue", alpha=0.9)
    if threshold_value is not None:
        axes[2].axvline(threshold_value, color="red", linestyle="--", linewidth=2, label=f"thr={threshold_value:.4f}")
        axes[2].legend(loc="upper right")
    axes[2].set_xlabel("Anomaly value")
    axes[2].set_ylabel("Pixel count")
    if threshold_value is not None and threshold_positive_pct is not None:
        fig.text(0.02, 0.02, f"threshold={threshold_value:.6f}, positive={threshold_positive_pct:.4f}%", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / get_debug_filename("threshold_decision"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    removed_pixels = None
    if mask_raw is not None and mask_final is not None:
        rb = np.asarray(mask_raw).astype(bool)
        fb = np.asarray(mask_final).astype(bool)
        if rb.shape == fb.shape:
            removed_pixels = np.logical_and(rb, np.logical_not(fb)).astype(np.float32)
    num_components = int(
        artifacts.decision_metadata.get(
            "num_kept_contours",
            artifacts.decision_metadata.get(
                "num_contours_kept",
                artifacts.decision_metadata.get("num_components", 0),
            ),
        )
    )
    save_three_panel_image(
        out / get_debug_filename("postprocessing_effect"),
        [
            ("Raw binary mask", mask_raw, "binary"),
            ("Final binary mask", mask_final, "binary"),
            ("Removed pixels", removed_pixels, "binary"),
        ],
        annotation=f"kept detections = {num_components}",
    )

    if raw_ins is not None:
        plt = _get_plt()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].set_title("Inspected image")
        axes[0].axis("off")
        axes[0].imshow(normalize_for_display(_to_gray(raw_ins)), cmap="gray", vmin=0.0, vmax=1.0)
        axes[1].set_title("Final binary mask")
        axes[1].axis("off")
        if mask_final is not None:
            axes[1].imshow(np.asarray(mask_final).astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            axes[1].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[1].transAxes)
        axes[2].set_title("Inspected + final mask overlay")
        axes[2].axis("off")
        insg = normalize_for_display(_to_gray(raw_ins))
        axes[2].imshow(insg, cmap="gray", vmin=0.0, vmax=1.0)
        if mask_final is not None:
            mb = np.asarray(mask_final).astype(bool)
            overlay = np.zeros((*mb.shape, 4), dtype=np.float32)
            overlay[..., 0] = 1.0
            overlay[..., 3] = mb.astype(np.float32) * 0.35
            axes[2].imshow(overlay)
        fig.tight_layout()
        fig.savefig(out / get_debug_filename("final_result"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Alignment-focused inspection figures (saved only, no plt.show()).
    if (
        aligned_ref is not None
        and aligned_ins is not None
        and pre_ref is not None
        and pre_ins is not None
    ):
        _save_alignment_transform_summary(
            out / get_debug_filename("alignment_transform_summary"),
            artifacts.alignment_metadata or {},
        )
        if aligned_ref.shape == aligned_ins.shape:
            _save_alignment_overlay_rgb(
                out / get_debug_filename("alignment_overlay_rgb"),
                aligned_ref=aligned_ref,
                inspected=aligned_ins,
                valid_mask=valid_mask,
            )
            _save_alignment_checkerboard(
                out / get_debug_filename("alignment_checkerboard"),
                aligned_ref=aligned_ref,
                inspected=aligned_ins,
            )
        if pre_diff is not None and aligned_diff is not None:
            _save_alignment_difference_triptych(
                out / get_debug_filename("alignment_difference_triptych"),
                pre_diff=pre_diff,
                aligned_diff=aligned_diff,
                valid_mask=valid_mask,
            )
        _save_alignment_blink_frames(
            ref_path=out / get_debug_filename("alignment_blink_frame_ref"),
            inspected_path=out / get_debug_filename("alignment_blink_frame_inspected"),
            aligned_ref=aligned_ref,
            inspected=aligned_ins,
        )

        if valid_mask is not None:
            _save_alignment_valid_mask_overlay(
                out / get_debug_filename("alignment_valid_mask_overlay"),
                inspected=aligned_ins,
                valid_mask=np.asarray(valid_mask),
            )

        # Search-specific inspection figures (coarse-to-fine Euclidean search).
        alignment_method = artifacts.alignment_metadata.get("method") if artifacts.alignment_metadata else None
        if alignment_method == "search_euclidean":
            _save_search_alignment_summary(
                out / get_debug_filename("search_alignment_summary"),
                artifacts.alignment_metadata or {},
            )
            coarse_records = artifacts.alignment_metadata.get("coarse_candidate_records", [])
            refined_records = artifacts.alignment_metadata.get("refined_candidate_records", [])
            _save_search_scores_grid(
                out / get_debug_filename("coarse_search_scores"),
                coarse_records,
                title="Coarse search",
            )
            _save_search_scores_grid(
                out / get_debug_filename("refined_search_scores"),
                refined_records,
                title="Refined search",
            )
            _save_search_iteration_theta_scores(
                out / get_debug_filename("search_iteration_theta_scores"),
                candidate_records=artifacts.alignment_metadata.get("candidate_records", []),
                iteration_summaries=artifacts.alignment_metadata.get("iteration_summaries", []),
            )
            _save_search_final_theta_neighborhood(
                out / get_debug_filename("search_final_theta_neighborhood"),
                neighborhood_records=artifacts.alignment_metadata.get("final_theta_neighborhood_records", []),
            )
            _save_search_translation_sensitivity_heatmap(
                out / get_debug_filename("search_translation_sensitivity_heatmap"),
                records=artifacts.alignment_metadata.get("translation_sensitivity_records", []),
            )
            _save_search_edge_distance_diagnostics(
                out / get_debug_filename("search_edge_distance_diagnostics"),
                inspected_edge_map=artifacts.alignment_metadata.get("inspected_edge_map"),
                warped_reference_edge_map=artifacts.alignment_metadata.get("best_warped_reference_edge_map"),
                inspected_edge_distance_map=artifacts.alignment_metadata.get("inspected_edge_distance_map"),
            )

            if aligned_ref is not None and aligned_ins is not None:
                _save_alignment_overlay_rgb(
                    out / get_debug_filename("best_transform_overlay"),
                    aligned_ref=aligned_ref,
                    inspected=aligned_ins,
                    valid_mask=valid_mask,
                )
                _save_alignment_checkerboard(
                    out / get_debug_filename("best_transform_checkerboard"),
                    aligned_ref=aligned_ref,
                    inspected=aligned_ins,
                )

            if pre_diff is not None and aligned_diff is not None:
                _save_alignment_difference_triptych(
                    out / get_debug_filename("best_transform_diff_triptych"),
                    pre_diff=pre_diff,
                    aligned_diff=aligned_diff,
                    valid_mask=valid_mask,
                )

    # Optional figures for new algorithmic options.
    alignment_method = artifacts.alignment_metadata.get("method") if artifacts.alignment_metadata else None
    comparison_method = artifacts.comparison_metadata.get("method")
    post_method = artifacts.decision_metadata.get("method")

    if alignment_method == "orb_affine" and pre_ref is not None and pre_ins is not None:
        _save_orb_keypoint_matches(out / get_debug_filename("orb_keypoint_matches"), pre_ref, pre_ins, artifacts.alignment_metadata or {})

    if valid_mask is not None:
        _save_valid_overlap_mask(out / get_debug_filename("valid_overlap_mask"), np.asarray(valid_mask), raw_ins)
        _save_alignment_border_effect(out / get_debug_filename("alignment_border_effect"), aligned_diff, np.asarray(valid_mask))

    if comparison_method == "ssim_comparator" and artifacts.ssim_map is not None and anomaly is not None:
        global_ssim = artifacts.comparison_metadata.get("global_ssim_score")
        _save_ssim_map(out / get_debug_filename("ssim_map"), np.asarray(artifacts.ssim_map), np.asarray(anomaly), None if global_ssim is None else float(global_ssim))
        _save_ssim_threshold_histogram(out / get_debug_filename("ssim_threshold_histogram"), np.asarray(anomaly), threshold_value, threshold_positive_pct)
        if norm_ref is not None and norm_ins is not None:
            _save_ssim_local_examples(out / get_debug_filename("ssim_local_examples"), np.asarray(norm_ref), np.asarray(norm_ins), np.asarray(anomaly))

    if post_method == "contour_filter_postprocess" and mask_raw is not None and raw_ins is not None:
        _save_contour_candidates(
            out / get_debug_filename("contour_candidates"),
            np.asarray(mask_raw),
            artifacts.decision_metadata,
            mask_final=artifacts.binary_mask_final,
        )
        _save_contour_area_histogram(out / get_debug_filename("contour_area_histogram"), artifacts.decision_metadata)
        _save_contour_boxes_overlay(out / get_debug_filename("contour_boxes_overlay"), np.asarray(raw_ins), artifacts.decision_metadata)
    elif post_method == "peak_nms_postprocess" and raw_ins is not None:
        _save_contour_boxes_overlay(out / get_debug_filename("contour_boxes_overlay"), np.asarray(raw_ins), artifacts.decision_metadata)

