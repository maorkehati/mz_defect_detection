from registries import (
    ALIGNER_REGISTRY,
    COMPARATOR_REGISTRY,
    NORMALIZER_REGISTRY,
    POSTPROCESSOR_REGISTRY,
    PREPROCESSOR_REGISTRY,
    THRESHOLDING_REGISTRY,
)


def _build_from_registry(name: str, registry: dict, family_name: str):
    if name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unknown {family_name} method: {name}. "
            f"Available options: [{available}]"
        )
    return registry[name]()


def build_preprocessor(name: str):
    return _build_from_registry(name, PREPROCESSOR_REGISTRY, "preprocessing")


def build_aligner(name: str):
    return _build_from_registry(name, ALIGNER_REGISTRY, "alignment")


def build_normalizer(name: str):
    return _build_from_registry(name, NORMALIZER_REGISTRY, "normalization")


def build_comparator(name: str):
    return _build_from_registry(name, COMPARATOR_REGISTRY, "comparison")


def build_thresholding(name: str):
    return _build_from_registry(name, THRESHOLDING_REGISTRY, "thresholding")


def build_postprocessor(name: str):
    return _build_from_registry(name, POSTPROCESSOR_REGISTRY, "postprocessing")