from __future__ import annotations

from dataclasses import asdict
from pprint import pformat

from config import (
    build_search_euclidean_gradient_difference_edge_suppressed_mad_config,
    build_search_euclidean_gradient_difference_mad_config,
    build_search_euclidean_gradient_difference_otsu_config,
    build_ecc_euclidean_ssim_config,
    build_ecc_euclidean_ssim_fixed_threshold_config,
    build_ecc_euclidean_ssim_otsu_config,
    build_ecc_affine_ssim_otsu_config,
    build_ecc_affine_projected_euclidean_ssim_otsu_config,
    build_ecc_translation_ssim_config,
    build_search_euclidean_edge_distance_ssim_otsu_config,
    build_search_euclidean_ssim_otsu_config,
    build_default_config,
    build_orb_ssim_config,
    build_orb_ssim_fixed_threshold_config,
    build_orb_ssim_otsu_config,
    build_ssim_config,
)


def build_pipeline_config_from_variant(variant: str):
    v = variant.strip().lower()
    if v == "default":
        return build_default_config()
    if v == "ssim":
        return build_ssim_config()
    if v == "orb_ssim":
        return build_orb_ssim_config()
    if v == "orb_ssim_otsu":
        return build_orb_ssim_otsu_config()
    if v == "orb_ssim_fixed":
        return build_orb_ssim_fixed_threshold_config()
    if v == "ecc_translation_ssim":
        return build_ecc_translation_ssim_config()
    if v == "ecc_euclidean_ssim":
        return build_ecc_euclidean_ssim_config()
    if v == "ecc_euclidean_ssim_otsu":
        return build_ecc_euclidean_ssim_otsu_config()
    if v == "ecc_euclidean_ssim_fixed":
        return build_ecc_euclidean_ssim_fixed_threshold_config()
    if v == "ecc_affine_ssim_otsu":
        return build_ecc_affine_ssim_otsu_config()
    if v == "ecc_affine_projected_euclidean_ssim_otsu":
        return build_ecc_affine_projected_euclidean_ssim_otsu_config()
    if v == "search_euclidean_ssim_otsu":
        return build_search_euclidean_ssim_otsu_config()
    if v == "search_euclidean_edge_distance_ssim_otsu":
        return build_search_euclidean_edge_distance_ssim_otsu_config()
    if v == "search_euclidean_gradient_difference_otsu":
        return build_search_euclidean_gradient_difference_otsu_config()
    if v == "search_euclidean_gradient_difference_mad":
        return build_search_euclidean_gradient_difference_mad_config()
    if v == "search_euclidean_gradient_difference_edge_suppressed_mad":
        return build_search_euclidean_gradient_difference_edge_suppressed_mad_config()
    raise ValueError(
        f"Unknown variant '{variant}'. "
        "Expected one of: default, ssim, orb_ssim, orb_ssim_otsu, orb_ssim_fixed, "
        "ecc_translation_ssim, ecc_euclidean_ssim, ecc_euclidean_ssim_otsu, "
        "ecc_euclidean_ssim_fixed, ecc_affine_ssim_otsu, "
        "ecc_affine_projected_euclidean_ssim_otsu, search_euclidean_ssim_otsu, "
        "search_euclidean_edge_distance_ssim_otsu, "
        "search_euclidean_gradient_difference_otsu, "
        "search_euclidean_gradient_difference_mad, "
        "search_euclidean_gradient_difference_edge_suppressed_mad."
    )


def apply_overrides(config, overrides: dict[str, object]) -> None:
    for path, value in overrides.items():
        parts = path.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid override path '{path}'. Expected dotted format.")

        target = config
        for attr in parts[:-1]:
            if not hasattr(target, attr):
                raise ValueError(f"Invalid override path '{path}': missing attribute '{attr}'.")
            target = getattr(target, attr)

        leaf = parts[-1]
        if hasattr(target, leaf):
            setattr(target, leaf, value)
            continue

        params = getattr(target, "params", None)
        if isinstance(params, dict):
            params[leaf] = value
            continue

        raise ValueError(
            f"Invalid override path '{path}': '{leaf}' is neither an attribute nor a params key."
        )


def config_to_pretty_text(
    config,
    experiment_name: str,
    variant: str,
    description: str,
    dataset_info: dict,
) -> str:
    lines = [
        f"experiment_name: {experiment_name}",
        f"variant: {variant}",
        f"description: {description}",
        "",
        "dataset_info:",
        pformat(dataset_info, sort_dicts=False),
        "",
        "module_choices:",
        pformat(asdict(config.choices), sort_dicts=False),
        "",
        "resolved_pipeline_config:",
        pformat(asdict(config), sort_dicts=False),
        "",
    ]
    return "\n".join(lines)

