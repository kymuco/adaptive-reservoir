"""State diagnostics for reservoir activations and traces."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from adaptive_reservoir.core.protocols import FloatArray
from adaptive_reservoir.core.state import ReservoirState


@dataclass(frozen=True, slots=True)
class TraceNorms:
    """RMS norms for multi-timescale trace vectors."""

    fast: float
    mid: float
    slow: float

    def __post_init__(self) -> None:
        _validate_non_negative_finite("fast", self.fast)
        _validate_non_negative_finite("mid", self.mid)
        _validate_non_negative_finite("slow", self.slow)


@dataclass(frozen=True, slots=True)
class StateDiagnostics:
    """Diagnostics derived from previous and current reservoir state."""

    state_norm: float
    state_delta: float
    saturation_rate: float
    trace_norms: TraceNorms

    def __post_init__(self) -> None:
        _validate_non_negative_finite("state_norm", self.state_norm)
        _validate_non_negative_finite("state_delta", self.state_delta)
        _validate_unit_interval("saturation_rate", self.saturation_rate)


def calculate_state_diagnostics(
    *,
    previous: ReservoirState,
    current: ReservoirState,
    saturation_threshold: float,
) -> StateDiagnostics:
    """Calculate deterministic diagnostics for a reservoir state transition."""

    _validate_saturation_threshold(saturation_threshold)
    _validate_matching_activation_shapes(previous, current)
    state_delta = current.activations - previous.activations
    return StateDiagnostics(
        state_norm=rms_norm(current.activations),
        state_delta=rms_norm(state_delta),
        saturation_rate=saturation_rate(
            current.activations,
            threshold=saturation_threshold,
        ),
        trace_norms=TraceNorms(
            fast=rms_norm(current.fast_trace),
            mid=rms_norm(current.mid_trace),
            slow=rms_norm(current.slow_trace),
        ),
    )


def rms_norm(values: FloatArray) -> float:
    """Return root-mean-square norm for a numeric vector."""

    array = np.asarray(values)
    if array.ndim != 1:
        msg = "values must be a 1D array"
        raise ValueError(msg)
    if not np.issubdtype(array.dtype, np.floating):
        msg = "values must have a floating dtype"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = "values must contain only finite values"
        raise ValueError(msg)
    if array.size == 0:
        msg = "values must not be empty"
        raise ValueError(msg)
    return float(math.sqrt(float(np.mean(np.square(array, dtype=np.float64)))))


def saturation_rate(values: FloatArray, *, threshold: float) -> float:
    """Return the fraction of values whose absolute magnitude reaches threshold."""

    _validate_saturation_threshold(threshold)
    array = np.asarray(values)
    if array.ndim != 1:
        msg = "values must be a 1D array"
        raise ValueError(msg)
    if not np.issubdtype(array.dtype, np.floating):
        msg = "values must have a floating dtype"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = "values must contain only finite values"
        raise ValueError(msg)
    if array.size == 0:
        msg = "values must not be empty"
        raise ValueError(msg)
    return float(np.mean(np.abs(array) >= threshold))


def _validate_matching_activation_shapes(
    previous: ReservoirState,
    current: ReservoirState,
) -> None:
    if previous.activations.shape != current.activations.shape:
        msg = "previous and current activations must have matching shapes"
        raise ValueError(msg)


def _validate_saturation_threshold(value: float) -> None:
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        msg = "saturation_threshold must be in the range (0.0, 1.0]"
        raise ValueError(msg)


def _validate_non_negative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        msg = f"{name} must be finite and non-negative"
        raise ValueError(msg)


def _validate_unit_interval(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        msg = f"{name} must be in the range [0.0, 1.0]"
        raise ValueError(msg)
