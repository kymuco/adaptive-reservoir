import numpy as np

from adaptive_reservoir import ReservoirState, calculate_state_diagnostics


def test_state_diagnostics_compute_exact_norms_and_delta() -> None:
    previous = _state(activations=[1.0, 0.0, 0.0])
    current = _state(activations=[0.0, 2.0, 0.0])
    features = np.array([0.0, 2.0, 0.0, 0.5])

    diagnostics = calculate_state_diagnostics(
        previous_state=previous,
        current_state=current,
        features=features,
        saturation_threshold=0.95,
    )

    assert diagnostics.state_norm == 2.0
    assert diagnostics.state_delta == np.sqrt(5.0)
    assert diagnostics.feature_norm == np.sqrt(4.25)


def test_state_diagnostics_compute_saturation_rate() -> None:
    current = _state(activations=[-0.96, 0.0, 0.95, 0.5])

    diagnostics = calculate_state_diagnostics(
        previous_state=_state(activations=[0.0, 0.0, 0.0, 0.0]),
        current_state=current,
        features=current.activations,
        saturation_threshold=0.95,
    )

    assert diagnostics.saturation_rate == 0.5


def test_state_diagnostics_compute_trace_norms() -> None:
    current = ReservoirState(
        activations=np.array([0.0, 0.0]),
        fast_trace=np.array([3.0, 4.0]),
        mid_trace=np.array([5.0, 12.0]),
        slow_trace=np.array([8.0, 15.0]),
    )

    diagnostics = calculate_state_diagnostics(
        previous_state=ReservoirState.zeros(n_cells=2),
        current_state=current,
        features=current.activations,
        saturation_threshold=0.95,
    )

    assert diagnostics.trace_norms.fast == 5.0
    assert diagnostics.trace_norms.mid == 13.0
    assert diagnostics.trace_norms.slow == 17.0


def test_state_diagnostics_are_finite() -> None:
    previous = _state(activations=[0.0, 0.0, 0.0])
    current = _state(activations=[0.1, -0.2, 0.3])

    diagnostics = calculate_state_diagnostics(
        previous_state=previous,
        current_state=current,
        features=np.array([0.1, -0.2, 0.3]),
        saturation_threshold=0.95,
    )

    assert np.isfinite(diagnostics.state_norm)
    assert np.isfinite(diagnostics.state_delta)
    assert np.isfinite(diagnostics.feature_norm)
    assert np.isfinite(diagnostics.saturation_rate)
    assert np.isfinite(diagnostics.trace_norms.fast)
    assert np.isfinite(diagnostics.trace_norms.mid)
    assert np.isfinite(diagnostics.trace_norms.slow)


def _state(*, activations: list[float]) -> ReservoirState:
    values = np.array(activations)
    return ReservoirState(
        activations=values,
        fast_trace=np.zeros_like(values),
        mid_trace=np.zeros_like(values),
        slow_trace=np.zeros_like(values),
    )
