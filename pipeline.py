from __future__ import annotations

from pathlib import Path

import numpy as np

from config import PipelineConfig
from dd_types import DetectionResult, PipelineArtifacts, SamplePair
from factories import (
    build_aligner,
    build_comparator,
    build_normalizer,
    build_postprocessor,
    build_preprocessor,
    build_thresholding,
)


class DefectDetectionPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

        self.preprocessor = build_preprocessor(cfg.choices.preprocessing)
        self.aligner = build_aligner(cfg.choices.alignment)
        self.normalizer = build_normalizer(cfg.choices.normalization)
        self.comparator = build_comparator(cfg.choices.comparison)
        self.thresholding = build_thresholding(cfg.choices.thresholding)
        self.postprocessor = build_postprocessor(cfg.choices.postprocessing)

    def run(self, sample: SamplePair) -> DetectionResult:
        artifacts = PipelineArtifacts(
            reference_input=sample.reference_image,
            inspected_input=sample.inspected_image,
        )

        ref = sample.reference_image
        ins = sample.inspected_image

        self._validate_inputs(ref, ins)

        if self.cfg.preprocessing.enabled:
            ref, ins = self.preprocessor.run(ref, ins, self.cfg.preprocessing)
        artifacts.reference_preprocessed = ref
        artifacts.inspected_preprocessed = ins

        if self.cfg.alignment.enabled:
            ref, ins, alignment_metadata = self.aligner.run(ref, ins, self.cfg.alignment)
        else:
            alignment_metadata = {}
        artifacts.reference_aligned = ref
        artifacts.inspected_aligned = ins
        artifacts.alignment_metadata = alignment_metadata

        if self.cfg.normalization.enabled:
            ref, ins, normalization_metadata = self.normalizer.run(
                ref, ins, self.cfg.normalization
            )
        else:
            normalization_metadata = {}
        artifacts.reference_normalized = ref
        artifacts.inspected_normalized = ins
        artifacts.normalization_metadata = normalization_metadata

        anomaly_map = self.comparator.run(ref, ins, self.cfg.comparison)
        artifacts.anomaly_map = anomaly_map

        binary_mask_raw, threshold_map = self.thresholding.run(
            anomaly_map,
            self.cfg.thresholding,
        )
        artifacts.binary_mask_raw = binary_mask_raw
        artifacts.threshold_map = threshold_map

        binary_mask_final, decision_metadata = self.postprocessor.run(
            binary_mask_raw,
            anomaly_map,
            self.cfg.postprocessing,
        )
        artifacts.binary_mask_final = binary_mask_final
        artifacts.decision_metadata.update(decision_metadata)

        if self.cfg.output.save_intermediate:
            self._save_artifacts(sample.pair_id, artifacts)

        return DetectionResult(
            pair_id=sample.pair_id,
            defect_mask=binary_mask_final,
            artifacts=artifacts if self.cfg.output.return_artifacts else PipelineArtifacts(),
        )

    def _validate_inputs(self, ref: np.ndarray, ins: np.ndarray) -> None:
        if not isinstance(ref, np.ndarray) or not isinstance(ins, np.ndarray):
            raise TypeError("Both reference_image and inspected_image must be numpy arrays.")

        if self.cfg.fail_on_shape_mismatch and ref.shape != ins.shape:
            raise ValueError(
                f"Shape mismatch between reference and inspected images: "
                f"{ref.shape} vs {ins.shape}"
            )

    def _save_artifacts(self, pair_id: str, artifacts: PipelineArtifacts) -> None:
        save_dir = self.cfg.output.save_dir
        if save_dir is None:
            raise ValueError("output.save_dir must be provided if save_intermediate=True")
        _ = Path(save_dir) / pair_id
        raise NotImplementedError("_save_artifacts is not implemented yet.")