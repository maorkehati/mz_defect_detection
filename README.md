# Semiconductor Defect Detection — Classical CV Pipeline

A modular **reference-vs-inspected** defect detection system for semiconductor imagery. The implementation is a fully configurable **classical computer vision** stack: preprocessing, geometric alignment, photometric normalization, anomaly-map construction, and peak-based localization—designed to isolate **small, localized, structurally distinct** defects while suppressing structured background differences, edges, and alignment artifacts.

---

## 1. Project overview

This repository provides:

- A **pipeline** (`DefectDetectionPipeline`) that ingests paired TIFFs (reference and inspected), runs alignment and comparison, and outputs a **binary defect mask**.
- Multiple **comparator** and **postprocessing** backends (e.g., gradient-domain residual, artifact-residual maps, MAD thresholding, peak extraction with non-maximum suppression).
- **Diagnostics**: compact multi-panel figures per case, optional extended comparator debug images, exported anomaly maps and masks.
- **Scripts** for single-batch runs, experiment batches, and parameter sweeps.

---

## 2. Problem description

### Inputs

For each **case**, the system receives:

- A **reference** image: a nominally clean chip region (or template).
- An **inspected** image: the same region after processing; it may contain **defects**.

Because dies are nominally identical, any discrepancy between the two images may reflect:

- sensor noise
- illumination or gain differences
- slight **misalignment**
- or **true defects**

### Goal

Produce a **binary defect mask** whose foreground pixels correspond to real defects, with high precision on structured wafers where edges and repeated geometry dominate false-alarm risk.

### Core difficulty

The detector must separate **true localized defects** from:

- strong chip **edges** and boundaries
- repeated geometric **texture**
- **alignment-induced** structured residuals
- noise

---

## 3. Approach and methodology

The solution follows a **principled classical CV** workflow:

1. **Preprocessing** — denoise / stabilize inputs for stable alignment and comparison.
2. **Alignment** — rigid **Euclidean** alignment (rotation + translation) so residuals reflect real differences, not global shift.
3. **Photometric normalization** — reduce global intensity bias between channels.
4. **Anomaly map** — a continuous **discrepancy map** emphasizing structure (e.g., gradients, local contrast, artifact-aware residuals) rather than raw intensity difference alone.
5. **Peak-based defect extraction** — defects are treated as **localized phenomena**: extract **local maxima** on the anomaly map, score **local prominence (peakness)**, suppress edge-dominated responses, apply **adaptive acceptance** from **global candidate statistics**, then **greedy score NMS** to avoid duplicate detections.

This aligns the algorithm with the physics of inspection: defects are **compact and salient**; background mismatch is often **extended and structured**.

---

## 4. Detailed pipeline explanation

### 4.1 Strong alignment (search Euclidean)

Misalignment creates large structured residuals that can overwhelm true defects. The default alignment is a **search-based rigid transform**:

- Coarse-to-fine **angle** search over a bounded range.
- **Translation** estimated per angle (phase-correlation–style scoring in the implementation).
- Scoring uses **gradient-domain** agreement so the solution is physically plausible and repeatable across parts.

Metadata (rotation, translation, overlap) is recorded for auditing and appears on diagnostic figures.

### 4.2 Structure-domain comparison

The pipeline emphasizes **structural** discrepancy (gradient magnitude, local contrast, artifact-residual cues) rather than raw gray-level subtraction alone. That reduces sensitivity to global illumination shifts while preserving defect-like local contrast.

### 4.3 Thresholding and continuous maps

A **MAD-based** (or alternative) thresholding stage produces intermediate binary evidence and threshold maps used for masking and diagnostics. The **continuous anomaly map** is the primary signal for downstream peak extraction.

### 4.4 Edge handling

Strong edges are a major source of false positives. The comparator and postprocess can **detect strong edges** and **down-weight or reject** peaks on or near them, improving precision on patterned substrates.

### 4.5 Peakness and scoring

Each candidate peak is described by local statistics; **peakness** is defined as **center value minus local neighborhood mean**, capturing how **sharp and compact** the response is. The acceptance **score** combines the anomaly response with peakness (and optional edge-distance terms where enabled).

### 4.6 Non-maximum suppression (NMS)

After thresholding candidates by score, **greedy score NMS** enforces a minimum spatial separation between accepted peaks so each physical defect is represented once.

### 4.7 Adaptive acceptance

Final peak acceptance uses thresholds **derived from the statistics of the current candidate set** (e.g., pool size and score distribution), so behavior remains stable across images without relying on a single fixed global score cutoff.

### 4.8 Local mask refinement (optional)

After peaks are fixed, the final binary mask can **expand each accepted threshold component** inside a local ROI using the **continuous anomaly map**: a hysteresis-style rule keeps pixels at or above a relaxed cutoff (e.g., tied to the seed’s max anomaly) **only if** they belong to the same connected region as the seed when thresholded together with the seed (`growable = (A ≥ t) ∨ seed`). This recovers defect extent clipped by a strict MAD mask without introducing new peaks or distant blobs. Set `refine_mode` to `"none"` in `PeakNMSPostprocessConfig` to skip.

---

## 5. Visualization: multi-panel pipeline figures

The primary per-case diagnostic is the **compact seven-panel** figure written as:

`outs/detection_results/<pair_id>_pipeline.png`

Examples (when those cases exist in your dataset):

- `defective_examples__case1_pipeline.png`
- `defective_examples__case2_pipeline.png`
- `non_defective_examples__case3_pipeline.png`

Each row is a **left-to-right progression** through the pipeline. Typical panels are:

| Panel | Title (typical) | What it shows | How to read it |
|------|------------------|---------------|----------------|
| **1** | **Inspected Image** | Raw inspected input before alignment-driven comparison. | Baseline appearance; defects may be subtle. |
| **2** | **Preprocessed Image** | After preprocessing (with a **gain** statistic vs. raw in the title). | Smoother, more comparable intensity domain. |
| **3** | **Alignment Overlay** | **RG overlay**: reference in red channel, inspected in green—overlap appears yellow. Title lists **rotation (deg)** and **translation (px)** plus valid **overlap** fraction. | Good alignment: structures coincide (yellow); large red/green separation indicates misalignment. |
| **4** | **Anomaly** | Continuous **anomaly map** (colormap, e.g. magma). Title may summarize comparator options (e.g., artifact-residual, top-hat size, edge handling) and **norm gain**. **Cyan crosses**: ground-truth defect locations when available. | Defects should appear as **bright, localized hotspots**. Broad sheets of activation suggest alignment or structured residual issues. |
| **5** | **Thresholded Mask (Pre-Postprocess)** | Binary mask after **thresholding** the anomaly evidence (before peak extraction / NMS). Title includes the effective **threshold** value when available. GT markers repeated. | Indicates where the statistical test flags anomaly **before** peaks are selected and merged into the final mask. |
| **6** | **Final Binary Defect Mask** | **Final binary prediction** (same as `DetectionResult.defect_mask`): **1 = defect**, **0 = background**, `nearest` interpolation. Starts from threshold **connected components** per accepted peak (with disk fallback when needed), then optional **local hysteresis refinement** on the continuous anomaly map (ROI around each seed, relaxed cutoff, connectivity-only growth) to recover support that a strict MAD mask may clip—without adding detections elsewhere. | **Assignment-style deliverable**: fuller defect shapes while peak selection is unchanged. Non-defective cases stay **empty**. Matches the last panel. |
| **7** | **Final Detection + Ground Truth** | **Contours** of the **same final mask** on normalized inspected. Title: detection **count** and **total area**. **Cyan crosses**: GT when available. | Confirms panel 6 and the contour view are one consistent prediction. |

**Final Binary Defect Mask (panel 6):** Post-peak-selection mask. Each kept peak still maps to the same threshold component (or disk fallback); **refinement** only expands that seed within a local ROI using the anomaly map (`PeakNMSPostprocessConfig`: `refine_component_support`, `refine_mode`, `refine_roi_margin_px`, `refine_growth_component_max_fraction`, etc.). Disable with `refine_mode: "none"` to show the strict threshold-union mask only.

**Ground-truth overlays:** When annotation files are present for a pair, **cyan markers** denote known defect locations on panels that support them. They are for **evaluation and calibration** only—they do not drive the core detector in production configuration.

**Extended artifact-residual diagnostics:** For the `artifact_residual` comparator, a second file may be written:

`outs/detection_results/<pair_id>_artifact_residual_debug.png`

with additional intermediate maps (residuals, edge masks, combined signals) for deep inspection.

### Full detection figure (`detection_panels.png`)

When `run_pipeline.py` saves panels (default), each case folder also contains `detection_panels.png`: a **multi-panel grid** (e.g., inspected, reference, raw anomaly score, display-normalized score, binary threshold mask, final mask, threshold map, absolute residual) suitable for detailed review. This is generated via `visualization.save_detection_figure`.

---

## 6. How to run the code

### Environment

```bash
cd defect_detection
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux / macOS
pip install -r requirements.txt
```

For PNG/TIFF exports and figures, install optional packages:

```bash
pip install matplotlib tifffile
```

### Primary entry point: `scripts/run_pipeline.py`

Runs the configured pipeline on **all** reference/inspected pairs discovered under a dataset glob.

```bash
python scripts/run_pipeline.py
```

Defaults:

- **Config variant:** `search_euclidean_artifact_residual_mad` (override with `--config` / `--variant` or `PIPELINE_VARIANT`).
- **Dataset root glob:** parent folder of the repo (`--root-pattern`), matching subfolders that contain `case*_inspected_image.tif` and `case*_reference_image.tif`.

Useful flags:

| Flag | Role |
|------|------|
| `--config` / `--variant` | Select `config.build_*` preset (e.g. `search_euclidean_artifact_residual_mad`). |
| `--root-pattern` | Glob for dataset roots (e.g. `"D:/data/my_wafer/*"`). |
| `--inspected-pattern` / `--reference-pattern` | Filename globs per case folder. |
| `--recursive` | Recursive search for TIFFs. |
| `--gt-radius` | Radius (px) for optional **GT coverage** summary in the console. |
| `--no-save-panels` / `--no-save-masks` / `--no-save-score-maps` | Disable optional exports. |

There is **no separate `run_dataset.py`**: batch processing over the dataset is performed by **`run_pipeline.py`**, which uses `data.load_sample_pairs()` to enumerate pairs.

### Batch experiments: `scripts/run_experiments.py`

Runs multiple named experiments from `configs/experiment_matrix.py`, writing under `outs/<experiment_root>/`. Use this to compare variants systematically.

### Sweeps and diagnostics (optional)

- `scripts/run_artifact_residual_sweep.py`, `scripts/run_focused_sweep.py`, `scripts/run_peak_nms_sweep.py`, etc. — write under `outs/sweeps/<sweep_name>/`.
- `scripts/run_ranking_probe.py` and similar — diagnostics under `outs/diagnostics/...`.

---

## 7. Output structure

All paths are relative to the repository root unless noted.

| Location | Contents |
|----------|----------|
| **`outs/detection_results/`** | Primary run outputs from `run_pipeline.py`. |
| **`outs/detection_results/<pair_id>_pipeline.png`** | Seven-panel compact progression (+ optional `*_artifact_residual_debug.png`). |
| **`outs/detection_results/<pair_id>/`** | Per-case folder: `detection_panels.png`, `defect_mask.png`, `defect_mask.tiff`, `anomaly_score.tiff`, `anomaly_score_display.png` (when saving is enabled). |
| **`outs/debug/<pair_id>/`** | Optional stage debug PNGs when `debug.save_debug_images` is enabled in config. |
| **`outs/sweeps/<sweep_name>/`** | Sweep scripts: grids, CSV summaries, per-config figures. |
| **`outs/experiment_runs/`** | Default root for `run_experiments.py` (see `configs/experiment_matrix.py`). |
| **`outs/threshold_investigation/`** | Optional investigation artifacts from threshold-analysis scripts (if used). |

Console output includes **detection counts** and, when ground truth exists, **coverage** metrics (points matched within a radius of the final mask).

---

## 8. Repository structure (overview)

```
defect_detection/
├── config.py                 # Pipeline dataclasses and build_* presets
├── pipeline.py               # DefectDetectionPipeline orchestration
├── dd_types.py             # Result / artifact types
├── data/                   # TIFF loading and sample pairing
├── modules/                # Preprocessors, aligners, comparators, thresholding, postprocessing
├── visualization/          # Figures, compact pipeline plots, exports
├── utils/                  # Ground-truth helpers, metrics
├── configs/                # Experiment matrices
├── scripts/                # run_pipeline.py, sweeps, probes, experiments
├── requirements.txt
└── outs/                   # Generated outputs (gitignored or local)
```

---

## 9. Dependencies

Core (`requirements.txt`):

- **numpy**
- **opencv-python-headless**
- **scikit-image**
- **tqdm**

Recommended for visualization and full I/O:

- **matplotlib** — pipeline figures and `detection_panels.png`
- **tifffile** — float TIFF anomaly maps and binary masks

---

## Why this formulation works

The pipeline succeeds because it matches the **geometry and appearance** of real defects:

- Defects are **small, local, and structurally salient** in the chosen anomaly map.
- Dominant background differences are often **structured and extended** (edges, pattern, alignment residual).

By **aligning strongly**, **comparing in a structure-aware domain**, **suppressing edges**, **scoring local prominence**, and **consolidating peaks with NMS and data-driven acceptance**, the system isolates genuine defects while remaining interpretable and tunable for semiconductor inspection workflows.
