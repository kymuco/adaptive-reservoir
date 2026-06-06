"""State diagnostics for reservoir runtime telemetry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_reservoir.core.protocols import FloatArray
from adaptive_reservoir.core.state import ReservoirState


@dataclass(frozen=True, slots=True)
class TraceNorms:
    """L2 norms for reservoir trace vectors."""

    fast: float
    mid: float
    slow: float


@dataclass(frozen=True, slots=True)
class StateDiagnostics:
    """Numeric diagnostics derived from reservoir state and features."""

    state_norm: float
    state_delta: float
    feature_norm: float
    saturation_rate: float
    trace_norms: TraceNorms


def calculate_state_diagnostics(
    *,
    previous_state: ReservoirState,
    current_state: ReservoirState,
    features: FloatArray,
    saturation_threshold: float,
) -> StateDiagnostics:
    """Calculate deterministic diagnostics for a completed reservoir step."""

    activations = current_state.activations
    previous_activations = previous_state.activations
    features_array = np.asarray(features)
    return StateDiagnostics(
        state_norm=_l2_norm(activations),
        state_delta=_l2_norm(activations - previous_activations),
        feature_norm=_l2_norm(features_array),
        saturation_rate=_saturation_rate(activations, saturation_threshold),
        trace_norms=TraceNorms(
            fast=_l2_norm(current_state.fast_trace),
            mid=_l2_norm(current_state.mid_trace),
            slow=_l2_norm(current_state.slow_trace),
        ),
    )


def _l2_norm(values: FloatArray) -> float:
    return float(np.linalg.norm(values))


def _saturation_rate(activations: FloatArray, threshold: float) -> float:
    return float(np.mean(np.abs(activations) >= threshold))
