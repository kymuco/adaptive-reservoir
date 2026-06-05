import numpy as np

from adaptive_reservoir import (
    AdaptiveReservoir,
    AdaptiveStepResult,
    ReservoirConfig,
    ReservoirState,
)

StreamItem = tuple[tuple[float, ...], float | None]
DeterministicResultView = tuple[object, ...]
DeterministicStateView = tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...], int]


def test_zero_initial_state_is_deterministic_for_same_shape_and_dtype() -> None:
    left = ReservoirState.zeros(n_cells=4, dtype="float64")
    right = ReservoirState.zeros(n_cells=4, dtype="float64")

    np.testing.assert_array_equal(left.activations, right.activations)
    np.testing.assert_array_equal(left.fast_trace, right.fast_trace)
    np.testing.assert_array_equal(left.mid_trace, right.mid_trace)
    np.testing.assert_array_equal(left.slow_trace, right.slow_trace)
    assert left.samples_seen == right.samples_seen


def test_same_input_stream_produces_same_draft_output_stream_except_timing() -> None:
    config = ReservoirConfig(input_dim=2, seed=42)
    stream = _example_stream()

    left = _run_stream(AdaptiveReservoir(config), stream)
    right = _run_stream(AdaptiveReservoir(config), stream)

    assert [_deterministic_view(result) for result in left] == [
        _deterministic_view(result) for result in right
    ]
    assert all(result.metrics.us_per_sample is not None for result in left)
    assert all(result.metrics.us_per_sample is not None for result in right)


def test_reset_replays_same_draft_output_stream_except_timing() -> None:
    model = AdaptiveReservoir(ReservoirConfig(input_dim=2, seed=42))
    stream = _example_stream()

    first = _run_stream(model, stream)
    model.reset()
    second = _run_stream(model, stream)

    assert [_deterministic_view(result) for result in first] == [
        _deterministic_view(result) for result in second
    ]


def _example_stream() -> tuple[StreamItem, ...]:
    return (
        ((0.1, -0.2), 1.0),
        ((0.3, 0.4), None),
        ((-0.5, 0.6), -1.0),
    )


def _run_stream(
    model: AdaptiveReservoir,
    stream: tuple[StreamItem, ...],
) -> list[AdaptiveStepResult]:
    return [model.step(x, target=target) for x, target in stream]


def _deterministic_view(result: AdaptiveStepResult) -> DeterministicResultView:
    metrics = result.metrics
    return (
        result.prediction,
        result.features,
        result.channels,
        metrics.samples_seen,
        metrics.prediction_available,
        metrics.target_available,
        metrics.readout_updated,
        metrics.state_norm,
        metrics.feature_norm,
        metrics.prediction_error,
        _state_view(result.state),
    )


def _state_view(state: ReservoirState | None) -> DeterministicStateView | None:
    if state is None:
        return None
    return (
        tuple(float(value) for value in state.activations),
        tuple(float(value) for value in state.fast_trace),
        tuple(float(value) for value in state.mid_trace),
        tuple(float(value) for value in state.slow_trace),
        state.samples_seen,
    )
