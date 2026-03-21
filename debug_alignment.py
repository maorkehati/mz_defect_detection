import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.registration import phase_cross_correlation


def read_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def robust_normalize(img: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(img, [1, 99])
    if p99 <= p1:
        return np.zeros_like(img, dtype=np.float32)
    x = np.clip((img - p1) / (p99 - p1), 0.0, 1.0)
    return x.astype(np.float32)


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.magnitude(gx, gy)
    return cv2.GaussianBlur(g, (0, 0), 1.0)


def hanning2d(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def checkerboard(a: np.ndarray, b: np.ndarray, tile: int = 32) -> np.ndarray:
    h, w = a.shape
    yy, xx = np.indices((h, w))
    patt = ((yy // tile) + (xx // tile)) % 2
    return np.where(patt == 0, a, b)


def overlay_rgb(ref: np.ndarray, ins: np.ndarray) -> np.ndarray:
    zr = np.zeros_like(ref)
    return np.dstack([ref, ins, zr])


def affine_about_center(shape: tuple[int, int], angle_deg: float, tx: float, ty: float) -> np.ndarray:
    h, w = shape
    center = (w / 2.0, h / 2.0)
    rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0).astype(np.float32)
    rot[:, 2] += [tx, ty]
    return rot


def warp_image(img: np.ndarray, M: np.ndarray, shape: tuple[int, int], interp: int) -> np.ndarray:
    h, w = shape
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


@dataclass
class AlignmentResult:
    angle_deg: float
    tx: float
    ty: float
    matrix: np.ndarray
    aligned_reference: np.ndarray
    valid_mask: np.ndarray
    coarse_scores: list[dict[str, Any]]
    refined_scores: list[dict[str, Any]]
    score: float


def _estimate_shift_with_masks(
    inspected_feat: np.ndarray,
    rotated_ref_feat: np.ndarray,
    rotated_ref_mask: np.ndarray,
    overlap_ratio: float,
    upsample_factor: int,
) -> tuple[float, float]:
    shift_rc, _, _ = phase_cross_correlation(
        reference_image=inspected_feat,
        moving_image=rotated_ref_feat,
        reference_mask=np.ones_like(inspected_feat, dtype=bool),
        moving_mask=rotated_ref_mask > 0.5,
        overlap_ratio=overlap_ratio,
        upsample_factor=upsample_factor,
        normalization=None,
    )
    dy, dx = float(shift_rc[0]), float(shift_rc[1])
    return dx, dy


def _score_candidate(
    ref_img: np.ndarray,
    ins_img: np.ndarray,
    ref_feat: np.ndarray,
    ins_feat: np.ndarray,
    angle_deg: float,
    overlap_threshold: float,
    upsample_factor: int,
) -> dict[str, Any]:
    shape = ref_img.shape
    M_rot = affine_about_center(shape, angle_deg, 0.0, 0.0)
    rotated_ref_feat = warp_image(ref_feat, M_rot, shape, cv2.INTER_LINEAR)
    rotated_ref_img = warp_image(ref_img, M_rot, shape, cv2.INTER_LINEAR)
    rotated_ref_mask = warp_image(np.ones(shape, np.float32), M_rot, shape, cv2.INTER_NEAREST)

    tx, ty = _estimate_shift_with_masks(
        inspected_feat=ins_feat,
        rotated_ref_feat=rotated_ref_feat,
        rotated_ref_mask=rotated_ref_mask,
        overlap_ratio=0.3,
        upsample_factor=upsample_factor,
    )

    M = affine_about_center(shape, angle_deg, tx, ty)
    aligned_ref_feat = warp_image(ref_feat, M, shape, cv2.INTER_LINEAR)
    aligned_ref_img = warp_image(ref_img, M, shape, cv2.INTER_LINEAR)
    valid_mask = warp_image(np.ones(shape, np.float32), M, shape, cv2.INTER_NEAREST) > 0.5

    overlap = float(valid_mask.mean())
    if overlap < overlap_threshold:
        score = np.inf
    else:
        diff = np.abs(aligned_ref_feat - ins_feat)
        core = valid_mask.copy().astype(np.uint8)
        core = cv2.erode(core, np.ones((3, 3), np.uint8), iterations=1) > 0
        score = float(np.median(diff[core])) if np.any(core) else np.inf

    return {
        "angle_deg": float(angle_deg),
        "tx": float(tx),
        "ty": float(ty),
        "score": float(score),
        "overlap": overlap,
        "matrix": M,
        "aligned_ref_img": aligned_ref_img,
        "valid_mask": valid_mask,
    }


def align_rigid_by_structure(
    reference: np.ndarray,
    inspected: np.ndarray,
    coarse_angle_min: float = -4.0,
    coarse_angle_max: float = 4.0,
    coarse_steps: int = 17,
    refine_half_width: float = 0.75,
    refine_steps: int = 15,
    overlap_threshold: float = 0.92,
    upsample_factor: int = 20,
) -> AlignmentResult:
    ref = robust_normalize(reference)
    ins = robust_normalize(inspected)

    ref_feat = gradient_magnitude(ref) * hanning2d(ref.shape)
    ins_feat = gradient_magnitude(ins) * hanning2d(ins.shape)

    coarse_angles = np.linspace(coarse_angle_min, coarse_angle_max, coarse_steps)
    coarse_scores = [
        _score_candidate(ref, ins, ref_feat, ins_feat, a, overlap_threshold, upsample_factor)
        for a in coarse_angles
    ]
    coarse_best = min(coarse_scores, key=lambda d: d["score"])

    refined_angles = np.linspace(
        coarse_best["angle_deg"] - refine_half_width,
        coarse_best["angle_deg"] + refine_half_width,
        refine_steps,
    )
    refined_scores = [
        _score_candidate(ref, ins, ref_feat, ins_feat, a, overlap_threshold, upsample_factor)
        for a in refined_angles
    ]
    best = min(refined_scores, key=lambda d: d["score"])

    return AlignmentResult(
        angle_deg=float(best["angle_deg"]),
        tx=float(best["tx"]),
        ty=float(best["ty"]),
        matrix=best["matrix"],
        aligned_reference=best["aligned_ref_img"],
        valid_mask=best["valid_mask"].astype(np.uint8),
        coarse_scores=coarse_scores,
        refined_scores=refined_scores,
        score=float(best["score"]),
    )


def save_visualization(reference: np.ndarray, inspected: np.ndarray, result: AlignmentResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = robust_normalize(reference)
    ins = robust_normalize(inspected)
    aligned = robust_normalize(result.aligned_reference)
    valid = result.valid_mask.astype(bool)

    before_diff = np.abs(ref - ins)
    after_diff = np.abs(aligned - ins)
    after_diff_masked = after_diff.copy()
    after_diff_masked[~valid] = 0.0

    # 1. Raw pair
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(ref, cmap="gray")
    axes[0].set_title("Reference")
    axes[1].imshow(ins, cmap="gray")
    axes[1].set_title("Inspected")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "01_raw_pair.png", dpi=160)
    plt.close(fig)

    # 2. Search scores
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, items, title in [
        (axes[0], result.coarse_scores, "Coarse angle search"),
        (axes[1], result.refined_scores, "Refined angle search"),
    ]:
        angles = [d["angle_deg"] for d in items]
        scores = [d["score"] for d in items]
        ax.plot(angles, scores, marker="o")
        best_idx = int(np.argmin(scores))
        ax.axvline(angles[best_idx], linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("angle (deg)")
        ax.set_ylabel("median |grad diff|")
    fig.suptitle(
        f"best angle={result.angle_deg:.4f}°, tx={result.tx:.3f}, ty={result.ty:.3f}, score={result.score:.5f}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "02_search_scores.png", dpi=160)
    plt.close(fig)

    # 3. Overlay before/after
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(overlay_rgb(ref, ins))
    axes[0].set_title("Before alignment\nR=reference, G=inspected")
    axes[1].imshow(overlay_rgb(aligned, ins))
    axes[1].set_title("After alignment\nR=aligned reference, G=inspected")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "03_overlay_before_after.png", dpi=160)
    plt.close(fig)

    # 4. Checkerboard
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(checkerboard(ref, ins), cmap="gray")
    axes[0].set_title("Checkerboard before")
    axes[1].imshow(checkerboard(aligned, ins), cmap="gray")
    axes[1].set_title("Checkerboard after")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "04_checkerboard_before_after.png", dpi=160)
    plt.close(fig)

    # 5. Difference triptych
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(before_diff, cmap="magma")
    axes[0].set_title("|ref - inspected|")
    axes[1].imshow(after_diff, cmap="magma")
    axes[1].set_title("|aligned ref - inspected|")
    axes[2].imshow(after_diff_masked, cmap="magma")
    axes[2].set_title("Masked post-align diff")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "05_difference_triptych.png", dpi=160)
    plt.close(fig)

    # 6. Valid overlap mask
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(valid, cmap="gray")
    axes[0].set_title("Valid overlap mask")
    axes[1].imshow(ins, cmap="gray")
    axes[1].imshow(np.ma.masked_where(valid, ~valid), cmap="autumn", alpha=0.5)
    axes[1].set_title("Invalid border highlighted")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "06_valid_overlap.png", dpi=160)
    plt.close(fig)


def process_pair(reference_path: Path, inspected_path: Path, out_dir: Path) -> None:
    reference = read_gray(reference_path)
    inspected = read_gray(inspected_path)
    result = align_rigid_by_structure(reference, inspected)
    save_visualization(reference, inspected, result, out_dir)
    print(f"pair={reference_path.stem}")
    print(f"best angle_deg={result.angle_deg:.6f}")
    print(f"best tx={result.tx:.6f}")
    print(f"best ty={result.ty:.6f}")
    print(f"score={result.score:.6f}")
    print(f"saved to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--inspected", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    process_pair(args.reference, args.inspected, args.out_dir)


if __name__ == "__main__":
    main()
