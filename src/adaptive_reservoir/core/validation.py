"""Shared validation helpers for public runtime inputs."""

from __future__ import annotations

import math

import numpy as np

from adaptive_reservoir.core.protocols import FloatArray


def validate_input_vector(
    x: object,
    *,
    input_dim: int,
    dtype: str,
    name: str = "x",
) -> FloatArray:
    """Validate and return a read-only 1D numeric input vector."""

    if isinstance(x, (str, bytes)):
        msg = f"{name} must be a 1D numeric vector"
        raise ValueError(msg)
    try:
        array = np.asarray(x, dtype=dtype)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must contain only numeric values"
        raise ValueError(msg) from exc
    if array.ndim != 1:
        msg = f"{name} must be a 1D numeric vector"
        raise ValueError(msg)
    if array.size != input_dim:
        msg = f"expected input_dim={input_dim}, got {array.size}"
        raise ValueError(msg)
    if not np.issubdtype(array.dtype, np.floating):
        msg = f"{name} must have a floating dtype"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)

    readonly = array.astype(array.dtype, copy=True)
    readonly.setflags(write=False)
    return readonly


def validate_optional_target(target: object | None, *, name: str = "target") -> float | None:
    """Validate and return an optional finite scalar target."""

    if target is None:
        return None
    if isinstance(target, (str, bytes, bool)):
        msg = f"{name} must be numeric"
        raise ValueError(msg)
    try:
        value = float(target)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be numeric"
        raise ValueError(msg) from exc
    if not math.isfinite(value):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return value


__all__ = ["validate_input_vector", "validate_optional_target"]
