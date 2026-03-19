from __future__ import annotations

import numpy as np


def ensure_numpy_array(x: np.ndarray, arg_name: str) -> None:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{arg_name} must be a numpy.ndarray, got {type(x)}.")

    # Modules expect numeric arrays, but some stages (e.g. postprocessing)
    # operate on boolean masks as well.
    if not (np.issubdtype(x.dtype, np.number) or np.issubdtype(x.dtype, np.bool_)):
        raise TypeError(
            f"{arg_name} must be a numeric or boolean numpy array, got dtype={x.dtype}."
        )


def ensure_same_shape(
    a: np.ndarray, b: np.ndarray, a_name: str, b_name: str
) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{a_name} and {b_name} must have the same shape: {a.shape} vs {b.shape}.")


__all__ = ["ensure_numpy_array", "ensure_same_shape"]