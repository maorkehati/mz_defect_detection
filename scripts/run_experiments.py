"""
Batch runner for multiple experiments in ``configs/experiment_matrix.py``.

For a **single** standard run of the primary artifact_residual pipeline on all pairs, prefer::

    python scripts/run_pipeline.py

See that script for ``--config`` / dataset flags.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.experiment_matrix import (
    EXPERIMENTS,
    EXPERIMENT_OUTPUT_ROOT_NAME,
    INSPECTED_PATTERN,
    REFERENCE_PATTERN,
    ROOT_PATTERN,
    SAVE_MASKS,
    SAVE_PLOTS,
    SAVE_SCORE_MAPS,
    SHOW_PLOTS,
)
from data import load_sample_pairs
from experiment_config import (
    apply_overrides,
    build_pipeline_config_from_variant,
    config_to_pretty_text,
)
from pipeline import DefectDetectionPipeline
from visualization import save_anomaly_map, save_binary_mask, save_detection_figure


def _write_summary(path: Path, lines: list[str], defect_pcts: list[float]) -> None:
    total = len(defect_pcts)
    avg_pct = (sum(defect_pcts) / total) if total else 0.0
    max_pct = max(defect_pcts) if defect_pcts else 0.0
    min_pct = min(defect_pcts) if defect_pcts else 0.0

    content = "\n".join(lines + [
        "",
        "aggregate:",
        f"total_samples={total}",
        f"average_defect_pct={avg_pct:.4f}%",
        f"max_defect_pct={max_pct:.4f}%",
        f"min_defect_pct={min_pct:.4f}%",
        "",
    ])
    path.write_text(content, encoding="utf-8")


def run_experiments() -> None:
    out_root = REPO_ROOT / "outs" / EXPERIMENT_OUTPUT_ROOT_NAME
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_info = {
        "root_pattern": ROOT_PATTERN,
        "inspected_pattern": INSPECTED_PATTERN,
        "reference_pattern": REFERENCE_PATTERN,
        "show_plots": SHOW_PLOTS,
        "save_plots": SAVE_PLOTS,
        "save_masks": SAVE_MASKS,
        "save_score_maps": SAVE_SCORE_MAPS,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    print("Loading dataset once for all experiments...")
    samples = load_sample_pairs(
        root_pattern=ROOT_PATTERN,
        inspected_pattern=INSPECTED_PATTERN,
        reference_pattern=REFERENCE_PATTERN,
        recursive=False,
        sort_results=True,
    )
    print(f"Loaded {len(samples)} sample pairs.")

    index_lines: list[str] = []

    for i, exp in enumerate(EXPERIMENTS, start=1):
        if "orb" in exp.variant:
            print(f"[SKIP] {exp.name} (ORB disabled)")
            continue

        exp_dir = out_root / exp.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[experiment {i}/{len(EXPERIMENTS)}] {exp.name}")
        print(f"  {exp.description}")

        cfg = build_pipeline_config_from_variant(exp.variant)
        apply_overrides(cfg, exp.overrides)

        print("[resolved pipeline]")
        print(f"  preprocessing = {cfg.choices.preprocessing}")
        print(f"  alignment = {cfg.choices.alignment}")
        print(f"  normalization = {cfg.choices.normalization}")
        print(f"  comparison = {cfg.choices.comparison}")
        print(f"  thresholding = {cfg.choices.thresholding}")
        print(f"  postprocessing = {cfg.choices.postprocessing}")

        cfg.output.save_dir = str(exp_dir)
        if hasattr(cfg, "debug") and cfg.debug is not None:
            cfg.debug.debug_dir = str(exp_dir / "debug")

        cfg_text = config_to_pretty_text(
            config=cfg,
            experiment_name=exp.name,
            variant=exp.variant,
            description=exp.description,
            dataset_info=dataset_info,
        )
        (exp_dir / "experiment_config.txt").write_text(cfg_text, encoding="utf-8")

        pipeline = DefectDetectionPipeline(cfg)

        summary_lines: list[str] = []
        defect_pcts: list[float] = []
        for j, sample in enumerate(samples, start=1):
            result = pipeline.run(sample)
            mask_bool = result.defect_mask.astype(bool)
            defect_pixels = int(mask_bool.sum())
            total_pixels = int(mask_bool.size)
            defect_pct = (defect_pixels / total_pixels * 100.0) if total_pixels else 0.0
            defect_pcts.append(defect_pct)

            comp = result.artifacts.decision_metadata.get("num_components")
            comp_text = str(comp) if isinstance(comp, int) else "NA"
            line = (
                f"[{j:03d}/{len(samples):03d}] {result.pair_id} | "
                f"defect_pixels={defect_pixels} | defect_pct={defect_pct:.4f}% | "
                f"components={comp_text}"
            )
            summary_lines.append(line)
            print("  " + line)

            case_dir = exp_dir / result.pair_id
            case_dir.mkdir(parents=True, exist_ok=True)

            if SAVE_PLOTS:
                save_detection_figure(result, case_dir / "detection_panels.png")
            if SAVE_MASKS:
                save_binary_mask(result.defect_mask, case_dir / "defect_mask.png")
                save_binary_mask(result.defect_mask, case_dir / "defect_mask.tiff")
            if SAVE_SCORE_MAPS and result.artifacts.anomaly_map is not None:
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

        _write_summary(exp_dir / "summary.txt", summary_lines, defect_pcts)
        index_lines.append(f"{exp.name} | {exp.description} | {exp_dir}")

    (out_root / "experiment_index.txt").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"\nAll experiments complete. Outputs: {out_root}")


if __name__ == "__main__":
    run_experiments()

