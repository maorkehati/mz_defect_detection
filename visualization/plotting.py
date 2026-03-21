from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from dd_types import DetectionResult

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Visualization requires 'matplotlib'. Please install a compatible "
            "matplotlib build for your numpy version."
        ) from exc
    return plt


def score_to_display_map(
    score_map: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """
    Convert an anomaly score map to [0, 1] for visualization only.
    """
    x = np.asarray(score_map, dtype=np.float32)
    lo = float(np.percentile(x, low_percentile))
    hi = float(np.percentile(x, high_percentile))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    out = (x - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _pick_reference_for_display(result: DetectionResult) -> Optional[np.ndarray]:
    artifacts = result.artifacts
    if artifacts.reference_normalized is not None:
        return artifacts.reference_normalized
    if artifacts.reference_aligned is not None:
        return artifacts.reference_aligned
    if artifacts.reference_preprocessed is not None:
        return artifacts.reference_preprocessed
    return artifacts.reference_input


def _show_image(ax, image: Optional[np.ndarray], title: str, cmap: str = "gray") -> None:
    ax.set_title(title)
    ax.axis("off")
    if image is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        return
    ax.imshow(image, cmap=cmap)


def _show_mask_overlay(
    ax,
    inspected: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    title: str = "Final Mask Overlay",
) -> None:
    ax.set_title(title)
    ax.axis("off")
    if inspected is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        return

    ax.imshow(inspected, cmap="gray")
    if mask is not None:
        mask_bool = np.asarray(mask).astype(bool)
        overlay = np.zeros((*mask_bool.shape, 4), dtype=np.float32)
        overlay[..., 0] = 1.0
        overlay[..., 3] = mask_bool.astype(np.float32) * 0.35
        ax.imshow(overlay)


def plot_prediction_panels(
    inspected_image: np.ndarray,
    reference_image: np.ndarray | None = None,
    anomaly_map: np.ndarray | None = None,
    binary_mask: np.ndarray | None = None,
    threshold_map: np.ndarray | None = None,
    binary_mask_raw: np.ndarray | None = None,
    figsize: tuple[float, float] = (14, 10),
    suptitle: str | None = None,
    show: bool = False,
) -> "plt.Figure":
    """
    Plot core prediction and debugging panels.
    """
    plt = _get_plt()
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axs = axes.ravel()

    _show_image(axs[0], inspected_image, "Inspected", cmap="gray")
    _show_image(axs[1], reference_image, "Reference (Best Available)", cmap="gray")
    _show_image(axs[2], anomaly_map, "Raw Anomaly Score", cmap="magma")
    _show_image(
        axs[3],
        score_to_display_map(anomaly_map) if anomaly_map is not None else None,
        "Display-Normalized Score",
        cmap="magma",
    )
    _show_image(axs[4], binary_mask_raw, "Binary Raw Threshold", cmap="gray")
    _show_image(axs[5], binary_mask, "Final Binary Mask", cmap="gray")
    _show_image(axs[6], threshold_map, "Threshold Map", cmap="viridis")

    residual = None
    if inspected_image is not None and reference_image is not None:
        if np.asarray(inspected_image).shape == np.asarray(reference_image).shape:
            residual = np.abs(
                np.asarray(inspected_image, dtype=np.float32)
                - np.asarray(reference_image, dtype=np.float32)
            )
    _show_image(axs[7], residual, "Absolute Residual", cmap="magma")

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_detection_result(
    result: DetectionResult,
    figsize: tuple[float, float] = (14, 10),
    suptitle: str | None = None,
    show: bool = False,
) -> "plt.Figure":
    """
    Plot standard panels from a DetectionResult.
    """
    artifacts = result.artifacts
    inspected = artifacts.inspected_input
    reference = _pick_reference_for_display(result)
    anomaly = artifacts.anomaly_map
    binary_raw = artifacts.binary_mask_raw
    binary_final = artifacts.binary_mask_final if artifacts.binary_mask_final is not None else result.defect_mask
    threshold_map = artifacts.threshold_map

    fig = plot_prediction_panels(
        inspected_image=inspected,
        reference_image=reference,
        anomaly_map=anomaly,
        binary_mask=binary_final,
        threshold_map=threshold_map,
        binary_mask_raw=binary_raw,
        figsize=figsize,
        suptitle=suptitle or f"Detection Result: {result.pair_id}",
        show=False,
    )

    overlay_ax = fig.add_axes([0.72, 0.02, 0.25, 0.25])
    _show_mask_overlay(overlay_ax, inspected=inspected, mask=binary_final)

    return fig

