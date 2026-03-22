"""
Parse ground-truth defect pixel locations from the exercise text file.

Coordinates are in the original inspected-image frame (same grid as pipeline arrays).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default path: sibling folder `defective_examples` under the parent of `defect_detection`.
def _default_locations_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root.parent / "defective_examples" / "defects locations.txt"


def parse_defect_locations_txt(path: Path | str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parse lines like:
        case 1:
        defect #1 at x=149, y=334
    into {"case1": [(149, 334), ...], ...}.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    cases: Dict[str, List[Tuple[int, int]]] = {}
    current: Optional[str] = None

    case_header = re.compile(r"^\s*case\s+(\d+)\s*:\s*$", re.IGNORECASE)
    defect_line = re.compile(
        r"^\s*defect\s*#(\d+)\s+at\s+x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)\s*$",
        re.IGNORECASE,
    )

    for line in text.splitlines():
        m = case_header.match(line)
        if m:
            current = f"case{int(m.group(1))}"
            cases.setdefault(current, [])
            continue
        m = defect_line.match(line)
        if m and current is not None:
            x, y = int(m.group(2)), int(m.group(3))
            cases[current].append((x, y))

    return cases


def pair_id_to_case_key(pair_id: str) -> Optional[str]:
    """
    Map pipeline pair ids to case keys used in the GT file.
    Examples:
      defective_examples__case1 -> case1
      case2 -> case2
    """
    if not pair_id:
        return None
    m = re.search(r"case\s*(\d+)", pair_id, re.IGNORECASE)
    if not m:
        return None
    return f"case{int(m.group(1))}"


_CACHED_PATH: Optional[str] = None
_CACHED_DATA: Optional[Dict[str, List[Tuple[int, int]]]] = None


def load_defect_locations(path: Path | str | None = None) -> Dict[str, List[Tuple[int, int]]]:
    """
    Load and parse the defect locations file. Returns empty dict if missing or unreadable.
    Parsed once per process per resolved path (cached).
    """
    global _CACHED_PATH, _CACHED_DATA
    if path is None:
        path = os.environ.get("DEFECT_GT_LOCATIONS_PATH")
        if path:
            path = Path(path)
        else:
            path = _default_locations_path()
    else:
        path = Path(path)

    resolved = str(path.resolve())
    if _CACHED_PATH == resolved and _CACHED_DATA is not None:
        return dict(_CACHED_DATA)

    if not path.is_file():
        _CACHED_PATH = resolved
        _CACHED_DATA = {}
        return {}

    try:
        data = parse_defect_locations_txt(path)
    except Exception:
        _CACHED_PATH = resolved
        _CACHED_DATA = {}
        return {}

    _CACHED_PATH = resolved
    _CACHED_DATA = {k: list(v) for k, v in data.items()}
    return dict(_CACHED_DATA)


def get_ground_truth_points_for_pair(
    pair_id: str,
    *,
    path: Path | str | None = None,
) -> List[Tuple[int, int]]:
    """
    Return list of (x, y) for this pair_id, or empty list if no GT or unknown case.
    """
    key = pair_id_to_case_key(pair_id)
    if key is None:
        return []
    data = load_defect_locations(path)
    return list(data.get(key, []))
