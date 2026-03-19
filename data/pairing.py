from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Tuple


INSPECTED_SUFFIX = "_inspected_image"
REFERENCE_SUFFIX = "_reference_image"


def extract_case_id(path: str | Path) -> str:
    """
    Extract a stable case id from an inspected/reference filename.

    Examples:
        case12_inspected_image.tif -> case12
        case12_reference_image.tiff -> case12
    """
    p = Path(path)
    stem = p.stem

    for suffix in (INSPECTED_SUFFIX, REFERENCE_SUFFIX):
        if stem.endswith(suffix):
            case_id = stem[: -len(suffix)]
            if case_id:
                return case_id
            break

    raise ValueError(
        f"Cannot extract case id from filename '{p.name}'. "
        f"Expected stem ending in '{INSPECTED_SUFFIX}' or '{REFERENCE_SUFFIX}'."
    )


def build_case_map(paths: Iterable[Path], side_name: str) -> Dict[str, Path]:
    """Build case_id -> file path mapping and reject duplicates."""
    case_map: Dict[str, Path] = {}
    for p in paths:
        case_id = extract_case_id(p)
        if case_id in case_map:
            raise ValueError(
                f"Duplicate {side_name} file for case '{case_id}': "
                f"'{case_map[case_id]}' and '{p}'."
            )
        case_map[case_id] = p
    return case_map


def pair_case_maps(
    inspected_map: Dict[str, Path],
    reference_map: Dict[str, Path],
    folder_hint: str,
) -> list[Tuple[str, Path, Path]]:
    """
    Pair inspected and reference maps by case id.

    Returns tuples: (case_id, inspected_path, reference_path)
    """
    inspected_ids = set(inspected_map)
    reference_ids = set(reference_map)

    missing_reference = sorted(inspected_ids - reference_ids)
    missing_inspected = sorted(reference_ids - inspected_ids)

    if missing_reference or missing_inspected:
        parts: list[str] = []
        if missing_reference:
            parts.append(
                f"missing reference for cases: {', '.join(missing_reference)}"
            )
        if missing_inspected:
            parts.append(
                f"missing inspected for cases: {', '.join(missing_inspected)}"
            )
        raise ValueError(
            f"Pairing mismatch in '{folder_hint}': " + "; ".join(parts) + "."
        )

    pairs: list[Tuple[str, Path, Path]] = []
    for case_id in sorted(inspected_ids):
        pairs.append((case_id, inspected_map[case_id], reference_map[case_id]))
    return pairs


def sanitize_id(text: str) -> str:
    """Convert text into a filesystem-friendly identifier chunk."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")

