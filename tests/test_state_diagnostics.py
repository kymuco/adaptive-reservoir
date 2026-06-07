import math

import numpy as np
import pytest

from adaptive_reservoir import (
    StateDiagnostics,
    TraceNorms,
    calculate_state_diagnostics,
    rms_norm,
)
from adaptive_reservoir.diagnostics.state import saturation_rate
from adaptive_reservoir.core.state import ReservoirState


def test_rms_norm_uses_root_mean_square() -> None:
    values = np.array([3.0, 4.0])

    assert rms_norm(values) == math.sqrt(12.5)


def test_state_diagnostics_calculates_state_norm_and_delta() -> None:
    previous = _state(activations=[0.0, 0.5, -0.5])
    current = _state(activations=[1.0, 0.5, -1.0])

    diagnostics = calculate_state_diagnostics(
        previous=previous,
        current=current,
        saturation_threshold=0.95,
    )

    assert diagnostics.state_norm == pytest.approx(math.sqrt((1.0 + 0.25 + 1.0) / 3.0))
    assert diagnostics.state_delta == pytest.approx(math.sqrt((1.0 + 0.0 + 0.25) / 3.0))


def test_saturation_rate_counts_abs_values_reaching_threshold() -> None:
    values = np.array([-0.95, -0.94, 0.0, 0.99])

    assert saturation_rate(values, threshold=0.95) == 0.5


def test_saturation_rate_returns_zero_when_no_cells_are_saturated() -> None:
    values = np.array([-0.1, 0.0, 0.8])

    assert saturation_rate(values, threshold=0.95) == 0.0


def test_saturation_rate_returns_one_when_all_cells_are_saturated() -> None:
    values = np.array([-1.0, -0.95, 0.95, 1.0])

    assert saturation_rate(values, threshold=0.95) == 1.0


def test_state_diagnostics_reports_trace_norms() -> None:
    previous = _state(activations=[0.0, 0.0, 0.0])
    current = _state(
        activations=[0.1, 0.2, 0.3],
        fast_trace=[1.0, 0.0, 0.0],
        mid_trace=[0.0, 2.0, 0.0],
        slow_trace=[0.0, 0.0, 3.0],
    )

    diagnostics = calculate_state_diagnostics(
        previous=previous,
        current=current,
        saturation_threshold=0.95,
    )

    assert diagnostics.trace_norms.fast == pytest.approx(math.sqrt(1.0 / 3.0))
    assert diagnostics.trace_norms.mid == pytest.approx(math.sqrt(4.0 / 3.0))
    assert diagnostics.trace_norms.slow == pytest.approx(math.sqrt(9.0 / 3.0))


def test_state_diagnostics_rejects_invalid_threshold() -> None:
    previous = _state(activations=[0.0, 0.0])
    current = _state(activations=[0.0, 0.0])

    with pytest.raises(ValueError, match="saturation_threshold"):
        calculate_state_diagnostics(
            previous=previous,
            current=current,
            saturation_threshold=0.0,
        )


def test_state_diagnostics_rejects_mismatched_activation_shapes() -> None:
    previous = _state(activations=[0.0, 0.0])
    current = _state(activations=[0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="matching shapes"):
        calculate_state_diagnostics(
            previous=previous,
            current=current,
            saturation_threshold=0.95,
        )


def test_trace_norms_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="fast must be finite and non-negative"):
        TraceNorms(fast=-0.1, mid=0.0, slow=0.0)


def test_state_diagnostics_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="saturation_rate"):
        StateDiagnostics(
            state_norm=0.0,
            state_delta=0.0,
            saturation_rate=1.1,
            trace_norms=TraceNorms(fast=0.0, mid=0.0, slow=0.0),
        )


def _state(
    *,
    activations: list[float],
    fast_trace: list[float] | None = None,
    mid_trace: list[float] | None = None,
    slow_trace: list[float] | None = None,
) -> ReservoirState:
    n_cells = len(activations)
    return ReservoirState(
        activations=np.array(activations, dtype=np.float64),
        fast_trace=np.array(fast_trace or [0.0] * n_cells, dtype=np.float64),
        mid_trace=np.array(mid_trace or [0.0] * n_cells, dtype=np.float64),
        slow_trace=np.array(slow_trace or [0.0] * n_cells, dtype=np.float64),
    )
