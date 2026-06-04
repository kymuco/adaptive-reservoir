"""Mathematical state objects for adaptive-reservoir."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class ReservoirState:
    """Mathematical reservoir state.

    This object stores only numeric runtime state. It must not store user facts,
    messages, semantic memory, policy decisions, or host-system domain events.
    """

    activations: FloatArray
    fast_trace: FloatArray
    mid_trace: FloatArray
    slow_trace: FloatArray
    samples_seen: int = 0

    @classmethod
    def zeros(cls, n_cells: int, dtype: str = "float64") -> ReservoirState:
        """Create a zero-initialized reservoir state."""

        if n_cells <= 0:
            msg = "n_cells must be positive"
            raise ValueError(msg)
        np_dtype = _resolve_dtype(dtype)
        return cls(
            activations=np.zeros(n_cells, dtype=np_dtype),
            fast_trace=np.zeros(n_cells, dtype=np_dtype),
            mid_trace=np.zeros(n_cells, dtype=np_dtype),
            slow_trace=np.zeros(n_cells, dtype=np_dtype),
            samples_seen=0,
        )

    def __post_init__(self) -> None:
        _validate_samples_seen(self.samples_seen)
        activations = _readonly_float_vector("activations", self.activations)
        fast_trace = _readonly_float_vector("fast_trace", self.fast_trace)
        mid_trace = _readonly_float_vector("mid_trace", self.mid_trace)
        slow_trace = _readonly_float_vector("slow_trace", self.slow_trace)
        _validate_matching_shapes(activations, fast_trace, mid_trace, slow_trace)

        object.__setattr__(self, "activations", activations)
        object.__setattr__(self, "fast_trace", fast_trace)
        object.__setattr__(self, "mid_trace", mid_trace)
        object.__setattr__(self, "slow_trace", slow_trace)


def _resolve_dtype(dtype: str) -> np.dtype[np.floating]:
    if dtype not in {"float32", "float64"}:
        msg = "dtype must be one of: float32, float64"
        raise ValueError(msg)
    return np.dtype(dtype)


def _readonly_float_vector(name: str, value: FloatArray) -> FloatArray:
    array = np.asarray(value)
    if array.ndim != 1:
        msg = f"{name} must be a 1D array"
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


def _validate_matching_shapes(*arrays: FloatArray) -> None:
    expected_shape = arrays[0].shape
    if any(array.shape != expected_shape for array in arrays[1:]):
        msg = "activations, fast_trace, mid_trace, and slow_trace must have the same shape"
        raise ValueError(msg)


def _validate_samples_seen(samples_seen: int) -> None:
    if samples_seen < 0:
        msg = "samples_seen must be non-negative"
        raise ValueError(msg)
