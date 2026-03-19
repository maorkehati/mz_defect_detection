from __future__ import annotations

import numpy as np

from modules.base import PostprocessorBase
from utils.morphology import (
    binary_closing,
    binary_opening,
    connected_components_with_stats,
    remove_small_components,
)


class BasicMorphologyPostprocessor(PostprocessorBase):
    def __init__(self) -> None:
        super().__init__(name="basic_morphology")

    def run(self, binary_mask_raw, anomaly_map, cfg):
        self.validate_config(cfg)
        self.validate_array(binary_mask_raw, "binary_mask_raw")
        self.validate_array(anomaly_map, "anomaly_map")
        self.validate_same_shape(
            binary_mask_raw, anomaly_map, "binary_mask_raw", "anomaly_map"
        )

        remove_small = bool(self.get_param(cfg, "remove_small_objects", False))
        min_area = int(self.get_param(cfg, "min_component_area", 1))
        morph_open_iterations = int(self.get_param(cfg, "morph_open_iterations", 0))
        morph_close_iterations = int(self.get_param(cfg, "morph_close_iterations", 0))

        mask = binary_mask_raw.astype(bool)

        if morph_open_iterations > 0:
            mask = binary_opening(mask, iterations=morph_open_iterations)

        if morph_close_iterations > 0:
            mask = binary_closing(mask, iterations=morph_close_iterations)

        if remove_small:
            mask = remove_small_components(mask, min_area=min_area)

        _, num_components, stats = connected_components_with_stats(mask)

        decision_metadata = {
            "num_components": int(num_components),
            "components": stats,
            "method": self.name,
        }

        return mask.astype(bool), decision_metadata

