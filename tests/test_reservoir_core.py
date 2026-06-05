import numpy as np
import pytest

from adaptive_reservoir import ReservoirConfig, ReservoirCore, ReservoirState
from adaptive_reservoir.topology import EdgeList


def test_reservoir_core_initializes_zero_state() -> None:
    core = ReservoirCore.from_config(_config())

    assert isinstance(core.state, ReservoirState)
    np.testing.assert_array_equal(core.state.activations, np.zeros(4))
    assert core.state.samples_seen == 0


def test_reservoir_core_builds_input_projection_shape_and_dtype() -> None:
    core = ReservoirCore.from_config(_config(dtype="float32"))

    assert core.input_weights.shape == (4, 2)
    assert core.input_weights.dtype == np.float32


def test_reservoir_core_input_projection_is_seeded() -> None:
    left = ReservoirCore.from_config(_config(seed=42))
    right = ReservoirCore.from_config(_config(seed=42))

    np.testing.assert_array_equal(left.input_weights, right.input_weights)


def test_reservoir_core_input_projection_changes_with_seed() -> None:
    left = ReservoirCore.from_config(_config(seed=42))
    right = ReservoirCore.from_config(_config(seed=43))

    assert not np.array_equal(left.input_weights, right.input_weights)


def test_reservoir_core_step_validates_input_dim() -> None:
    core = ReservoirCore.from_config(_config())

    with pytest.raises(ValueError, match="expected input_dim=2, got 1"):
        core.step([1.0])


def test_reservoir_core_step_rejects_non_finite_input() -> None:
    core = ReservoirCore.from_config(_config())

    with pytest.raises(ValueError, match="all input values must be finite"):
        core.step([1.0, float("nan")])


def test_reservoir_core_step_increments_samples_seen() -> None:
    core = ReservoirCore.from_config(_config())

    state = core.step([0.5, -0.25])

    assert state.samples_seen == 1
    assert core.state.samples_seen == 1


def test_reservoir_core_step_updates_activations_with_tanh() -> None:
    core = _manual_core(leak_rate=1.0)

    state = core.step([0.5, -1.0])

    expected_pre_activation = np.array([0.5, -1.0, 0.0])
    np.testing.assert_allclose(state.activations, np.tanh(expected_pre_activation))


def test_reservoir_core_step_applies_leak_rate() -> None:
    core = _manual_core(leak_rate=0.25)

    state = core.step([0.5, -1.0])

    expected_pre_activation = np.array([0.5, -1.0, 0.0])
    np.testing.assert_allclose(state.activations, 0.25 * np.tanh(expected_pre_activation))


def test_reservoir_core_step_uses_sparse_recurrent_edges() -> None:
    core = _manual_core(leak_rate=1.0)
    core.step([1.0, 0.0])

    state = core.step([0.0, 0.0])

    previous = np.tanh(np.array([1.0, 0.0, 0.0]))
    expected_pre_activation = np.array([0.0, previous[0] * 0.5, 0.0])
    np.testing.assert_allclose(state.activations, np.tanh(expected_pre_activation))


def test_reservoir_core_step_applies_fatigue_rate() -> None:
    core = _manual_core(leak_rate=1.0, fatigue_rate=0.5)
    core.step([1.0, 0.0])

    state = core.step([0.0, 0.0])

    previous = np.tanh(np.array([1.0, 0.0, 0.0]))
    expected_pre_activation = np.array([-0.5 * previous[0], previous[0] * 0.5, 0.0])
    np.testing.assert_allclose(state.activations, np.tanh(expected_pre_activation))


def test_reservoir_core_step_preserves_traces_for_pr31() -> None:
    core = _manual_core(leak_rate=1.0)
    fast_trace = core.state.fast_trace
    mid_trace = core.state.mid_trace
    slow_trace = core.state.slow_trace

    state = core.step([0.5, -1.0])

    np.testing.assert_array_equal(state.fast_trace, fast_trace)
    np.testing.assert_array_equal(state.mid_trace, mid_trace)
    np.testing.assert_array_equal(state.slow_trace, slow_trace)


def test_reservoir_core_step_keeps_state_finite() -> None:
    core = ReservoirCore.from_config(_config())

    for _ in range(8):
        state = core.step([100.0, -100.0])

    assert np.all(np.isfinite(state.activations))


def test_reservoir_core_validates_recurrent_edge_count() -> None:
    config = _config()
    with pytest.raises(ValueError, match="recurrent_edges.n_nodes must match"):
        ReservoirCore(
            config=config,
            recurrent_edges=EdgeList(
                n_nodes=3,
                sources=np.array([0]),
                targets=np.array([1]),
                weights=np.array([1.0]),
            ),
            input_weights=np.zeros((4, 2)),
            state=ReservoirState.zeros(n_cells=4),
        )


def test_reservoir_core_validates_input_weight_shape() -> None:
    config = _config()
    with pytest.raises(ValueError, match="input_weights must have shape"):
        ReservoirCore(
            config=config,
            recurrent_edges=_manual_edges(),
            input_weights=np.zeros((3, 2)),
            state=ReservoirState.zeros(n_cells=4),
        )


def _config(
    *,
    seed: int = 42,
    dtype: str = "float64",
) -> ReservoirConfig:
    return ReservoirConfig(
        input_dim=2,
        n_cells=4,
        topology="ring_shortcuts",
        seed=seed,
        dtype=dtype,  # type: ignore[arg-type]
    )


def _manual_core(*, leak_rate: float, fatigue_rate: float = 0.0) -> ReservoirCore:
    config = ReservoirConfig(
        input_dim=2,
        n_cells=3,
        topology="random_sparse",
        leak_rate=leak_rate,
        recurrent_scale=1.0,
        fatigue_rate=fatigue_rate,
    )
    return ReservoirCore(
        config=config,
        recurrent_edges=_manual_edges(),
        input_weights=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        ),
        state=ReservoirState.zeros(n_cells=3),
    )


def _manual_edges() -> EdgeList:
    return EdgeList(
        n_nodes=3,
        sources=np.array([0]),
        targets=np.array([1]),
        weights=np.array([0.5]),
    )
