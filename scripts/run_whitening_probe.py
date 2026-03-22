"""
Deprecated alias: residual whitening was replaced by local patch NCC.

Use ``scripts/run_ncc_probe.py`` (same relaxed postprocess + contrast_area_log k=3).

This entry point forwards to the NCC probe for backward-compatible invocations.
"""

from __future__ import annotations

import sys
import warnings

warnings.warn(
    "run_whitening_probe is deprecated; use scripts/run_ncc_probe.py",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    from pathlib import Path

    _sd = Path(__file__).resolve().parent
    if str(_sd) not in sys.path:
        sys.path.insert(0, str(_sd))
    from run_ncc_probe import main

    main()
