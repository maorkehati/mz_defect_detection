from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable

import numpy as np

from data.pairing import (
    build_case_map,
    pair_case_maps,
    sanitize_id,
)
from dd_types import SamplePair


def load_tiff_image(path: str | Path) -> np.ndarray:
    """
    Load a TIFF image as a numpy array.

    Tries tifffile first, then falls back to PIL.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file does not exist: '{p}'.")

    tifffile_error: Exception | None = None
    try:
        import tifffile

        arr = tifffile.imread(str(p))
        return np.asarray(arr)
    except Exception as exc:  # pragma: no cover - environment/codec dependent
        tifffile_error = exc

    try:
        from PIL import Image
    except ImportError as exc:
        if tifffile_error is not None:
            raise ImportError(
                "Failed to read TIFF image using tifffile, and Pillow is not installed. "
                f"tifffile error: {tifffile_error!r}"
            ) from exc
        raise ImportError(
            "TIFF loading requires either 'tifffile' or 'Pillow' to be installed."
        ) from exc

    try:
        with Image.open(p) as img:
            return np.asarray(img)
    except Exception as exc:
        if tifffile_error is not None:
            raise RuntimeError(
                f"Failed to load TIFF '{p}' with both tifffile and Pillow. "
                f"tifffile error: {tifffile_error!r}; Pillow error: {exc!r}"
            ) from exc
        raise


def _expand_roots(root_pattern: str) -> list[Path]:
    roots = [Path(p) for p in glob.glob(root_pattern)]
    if not roots:
        raise FileNotFoundError(
            f"No paths matched root_pattern='{root_pattern}'."
        )
    return roots


def _iter_files(folder: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    return folder.rglob(pattern) if recursive else folder.glob(pattern)


def _iter_tiff_files(folder: Path, pattern: str, recursive: bool) -> list[Path]:
    files = list(_iter_files(folder, pattern, recursive))

    lower = pattern.lower()
    alt_pattern: str | None = None
    if lower.endswith(".tif"):
        alt_pattern = pattern[:-4] + ".tiff"
    elif lower.endswith(".tiff"):
        alt_pattern = pattern[:-5] + ".tif"

    if alt_pattern:
        files.extend(_iter_files(folder, alt_pattern, recursive))

    return [p for p in files if p.is_file()]


def _build_pair_id(folder: Path, case_id: str) -> str:
    parent = sanitize_id(folder.name or str(folder))
    case = sanitize_id(case_id)
    return f"{parent}__{case}" if parent else case


def load_sample_pairs(
    root_pattern: str = r"C:\Users\mayoa\Desktop\home exercise\*",
    inspected_pattern: str = "case*_inspected_image.tif",
    reference_pattern: str = "case*_reference_image.tif",
    recursive: bool = False,
    sort_results: bool = True,
) -> list[SamplePair]:
    """
    Load all TIFF inspected/reference pairs from folders matched by root_pattern.
    """
    roots = _expand_roots(root_pattern)
    folders: list[Path] = []
    seen_folders: set[Path] = set()
    for root in roots:
        folder = root if root.is_dir() else root.parent
        resolved = folder.resolve()
        if resolved in seen_folders:
            continue
        seen_folders.add(resolved)
        folders.append(folder)

    samples: list[SamplePair] = []
    for folder in folders:
        if not folder.exists() or not folder.is_dir():
            continue

        inspected_files = _iter_tiff_files(folder, inspected_pattern, recursive)
        reference_files = _iter_tiff_files(folder, reference_pattern, recursive)

        if not inspected_files and not reference_files:
            continue

        inspected_map = build_case_map(inspected_files, side_name="inspected")
        reference_map = build_case_map(reference_files, side_name="reference")
        pairs = pair_case_maps(inspected_map, reference_map, folder_hint=str(folder))

        for case_id, inspected_path, reference_path in pairs:
            sample = SamplePair(
                reference_image=load_tiff_image(reference_path),
                inspected_image=load_tiff_image(inspected_path),
                pair_id=_build_pair_id(folder, case_id),
            )
            samples.append(sample)

    if not samples:
        raise FileNotFoundError(
            "No paired samples found. "
            f"root_pattern='{root_pattern}', inspected_pattern='{inspected_pattern}', "
            f"reference_pattern='{reference_pattern}'."
        )

    if sort_results:
        samples.sort(key=lambda s: s.pair_id)

    return samples

