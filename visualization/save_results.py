from __future__ import annotations

from pathlib import Path

import numpy as np

from dd_types import DetectionResult
from visualization.plotting import plot_detection_result, score_to_display_map


def _get_plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Saving visualization PNGs requires 'matplotlib'. Please install a "
            "compatible matplotlib build for your numpy version."
        ) from exc
    return plt


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_detection_figure(
    result: DetectionResult,
    output_path: str | Path,
    figsize: tuple[float, float] = (14, 10),
) -> Path:
    """Save a multi-panel detection figure to disk."""
    out = Path(output_path)
    _ensure_parent(out)
    plt = _get_plt()
    fig = plot_detection_result(result=result, figsize=figsize, show=False)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def save_binary_mask(mask: np.ndarray, output_path: str | Path) -> Path:
    """Save a binary mask as PNG/TIFF based on extension."""
    out = Path(output_path)
    _ensure_parent(out)
    mask_bool = np.asarray(mask).astype(bool)

    if out.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError(
                "Saving TIFF requires 'tifffile' to be installed."
            ) from exc
        tifffile.imwrite(str(out), mask_bool.astype(np.uint8))
        return out

    plt = _get_plt()
    plt.imsave(out, mask_bool.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
    return out


def save_anomaly_map(
    anomaly_map: np.ndarray,
    output_path: str | Path,
    normalize_for_view: bool = True,
) -> Path:
    """Save anomaly score map, preserving float for TIFF when feasible."""
    out = Path(output_path)
    _ensure_parent(out)
    score = np.asarray(anomaly_map)

    if out.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError(
                "Saving TIFF requires 'tifffile' to be installed."
            ) from exc
        tifffile.imwrite(str(out), score.astype(np.float32))
        return out

    plt = _get_plt()
    view = score_to_display_map(score) if normalize_for_view else np.asarray(score, dtype=np.float32)
    plt.imsave(out, view, cmap="magma", vmin=0.0, vmax=1.0)
    return out

