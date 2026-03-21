from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Tuple, TypeVar

import numpy as np

from config import (
    AlignmentConfig,
    ComparisonConfig,
    NormalizationConfig,
    PostprocessingConfig,
    PreprocessingConfig,
    ThresholdingConfig,
)
from utils.validation import ensure_numpy_array, ensure_same_shape

Array = np.ndarray
TConfig = TypeVar("TConfig")


class BaseModule(ABC, Generic[TConfig]):
    def __init__(self, name: str):
        self.name = name

    def validate_array(self, x: Array, arg_name: str) -> None:
        ensure_numpy_array(x, arg_name)

    def validate_same_shape(self, a: Array, b: Array, a_name: str, b_name: str) -> None:
        ensure_same_shape(a, b, a_name, b_name)

    def validate_config(self, cfg: TConfig) -> None:
        if cfg is None:
            raise ValueError(f"{self.name}: cfg must not be None.")

    def get_param(self, cfg: TConfig, key: str, default: Any = None) -> Any:
        params = getattr(cfg, "params", None)
        if params is None:
            return default
        return params.get(key, default)


class PreprocessorBase(BaseModule[PreprocessingConfig], ABC):
    @abstractmethod
    def run(
        self,
        reference_image: Array,
        inspected_image: Array,
        cfg: PreprocessingConfig,
    ) -> Tuple[Array, Array]:
        raise NotImplementedError


class AlignerBase(BaseModule[AlignmentConfig], ABC):
    @abstractmethod
    def run(
        self,
        reference_image: Array,
        inspected_image: Array,
        cfg: AlignmentConfig,
    ) -> Tuple[Array, Array, Dict[str, Any]]:
        raise NotImplementedError


class NormalizerBase(BaseModule[NormalizationConfig], ABC):
    @abstractmethod
    def run(
        self,
        reference_image: Array,
        inspected_image: Array,
        cfg: NormalizationConfig,
    ) -> Tuple[Array, Array, Dict[str, Any]]:
        raise NotImplementedError


class ComparatorBase(BaseModule[ComparisonConfig], ABC):
    @abstractmethod
    def run(
        self,
        reference_image: Array,
        inspected_image: Array,
        cfg: ComparisonConfig,
    ) -> Any:
        raise NotImplementedError


class ThresholdingBase(BaseModule[ThresholdingConfig], ABC):
    @abstractmethod
    def run(
        self,
        anomaly_map: Array,
        cfg: ThresholdingConfig,
    ) -> Any:
        raise NotImplementedError


class PostprocessorBase(BaseModule[PostprocessingConfig], ABC):
    @abstractmethod
    def run(
        self,
        binary_mask_raw: Array,
        anomaly_map: Array,
        cfg: PostprocessingConfig,
    ) -> Tuple[Array, Dict[str, Any]]:
        raise NotImplementedError