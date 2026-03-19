"""Main repository entry point for running the full defect-detection pipeline.

This script reads input images from the dataset glob path (input-only) and writes
all outputs under the repository root.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from defect_detection.config import build_default_config
from defect_detection.data import load_sample_pairs
from defect_detection.pipeline import DefectDetectionPipeline
from defect_detection.visualization import (
    plot_detection_result,
    save_anomaly_map,
    save_binary_mask,
    save_detection_figure,
)

# Input dataset location (read-only source path).
ROOT_PATTERN = r"C:\Users\mayoa\Desktop\home exercise\*"
INSPECTED_PATTERN = "case*_inspected_image.tif"
REFERENCE_PATTERN = "case*_reference_image.tif"

SHOW_PLOTS = True
SAVE_PLOTS = True
SAVE_MASKS = True
SAVE_SCORE_MAPS = True
RECURSIVE = False

# Outputs are intentionally rooted inside this repository.
OUTPUT_DIR = REPO_ROOT / "outs" / "detection_results"


def run_pipeline(
    root_pattern: str = ROOT_PATTERN,
    inspected_pattern: str = INSPECTED_PATTERN,
    reference_pattern: str = REFERENCE_PATTERN,
    recursive: bool = RECURSIVE,
    show_plots: bool = SHOW_PLOTS,
    save_plots: bool = SAVE_PLOTS,
    save_masks: bool = SAVE_MASKS,
    save_score_maps: bool = SAVE_SCORE_MAPS,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    samples = load_sample_pairs(
        root_pattern=root_pattern,
        inspected_pattern=inspected_pattern,
        reference_pattern=reference_pattern,
        recursive=recursive,
        sort_results=True,
    )

    pipeline = DefectDetectionPipeline(build_default_config())

    if save_plots or save_masks or save_score_maps:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(samples)} sample pairs from pattern: {root_pattern}")

    for idx, sample in enumerate(samples, start=1):
        result = pipeline.run(sample)
        defect_pixels = int(result.defect_mask.astype(bool).sum())
        total_pixels = int(result.defect_mask.size)
        ratio = (defect_pixels / total_pixels) if total_pixels else 0.0

        print(
            f"[{idx:03d}/{len(samples):03d}] {result.pair_id} | "
            f"defect_pixels={defect_pixels} ({ratio:.4%})"
        )

        case_dir = output_dir / result.pair_id
        if save_plots or save_masks or save_score_maps:
            case_dir.mkdir(parents=True, exist_ok=True)

        if show_plots:
            plot_detection_result(result, suptitle=f"Case: {result.pair_id}", show=True)

        if save_plots:
            save_detection_figure(result, case_dir / "detection_panels.png")

        if save_masks:
            save_binary_mask(result.defect_mask, case_dir / "defect_mask.png")
            save_binary_mask(result.defect_mask, case_dir / "defect_mask.tiff")

        if save_score_maps and result.artifacts.anomaly_map is not None:
            save_anomaly_map(
                result.artifacts.anomaly_map,
                case_dir / "anomaly_score.tiff",
                normalize_for_view=False,
            )
            save_anomaly_map(
                result.artifacts.anomaly_map,
                case_dir / "anomaly_score_display.png",
                normalize_for_view=True,
            )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run defect detection pipeline over all paired TIFF samples."
    )
    parser.add_argument("--root-pattern", default=ROOT_PATTERN)
    parser.add_argument("--inspected-pattern", default=INSPECTED_PATTERN)
    parser.add_argument("--reference-pattern", default=REFERENCE_PATTERN)
    parser.add_argument("--recursive", action="store_true", default=RECURSIVE)
    parser.add_argument("--show-plots", action="store_true", default=SHOW_PLOTS)
    parser.add_argument("--no-show-plots", action="store_false", dest="show_plots")
    parser.add_argument("--save-plots", action="store_true", default=SAVE_PLOTS)
    parser.add_argument("--no-save-plots", action="store_false", dest="save_plots")
    parser.add_argument("--save-masks", action="store_true", default=SAVE_MASKS)
    parser.add_argument("--no-save-masks", action="store_false", dest="save_masks")
    parser.add_argument("--save-score-maps", action="store_true", default=SAVE_SCORE_MAPS)
    parser.add_argument("--no-save-score-maps", action="store_false", dest="save_score_maps")
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Output root (defaults to <repo_root>/outs/detection_results).",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run_pipeline(
        root_pattern=args.root_pattern,
        inspected_pattern=args.inspected_pattern,
        reference_pattern=args.reference_pattern,
        recursive=args.recursive,
        show_plots=args.show_plots,
        save_plots=args.save_plots,
        save_masks=args.save_masks,
        save_score_maps=args.save_score_maps,
        output_dir=Path(args.output_dir),
    )

