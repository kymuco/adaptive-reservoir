from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from adaptive_reservoir import ReservoirConfig
from adaptive_reservoir.core.reservoir import (
    ReservoirCore,
    _ring_shortcuts_recurrent_drive_into,
    _sparse_recurrent_drive,
)
from adaptive_reservoir.core.state import ReservoirState
from adaptive_reservoir.core.validation import validate_input_vector
from adaptive_reservoir.topology import EdgeList

_INPUT_STREAM = (
    (0.1, -0.2),
    (0.3, 0.4),
    (-0.5, 0.2),
    (0.0, -0.7),
    (0.8, 0.1),
)


@pytest.mark.parametrize("n_cells", [2, 3, 16])
def test_ring_shortcuts_fast_path_installed_for_default_topology(n_cells: int) -> None:
    core = _ring_core(n_cells=n_cells, dtype="float64")

    assert core._ring_fast_path is not None


@pytest.mark.parametrize("topology", ["random_sparse", "modular_small_world"])
def test_ring_shortcuts_fast_path_not_installed_for_non_ring_topologies(
    topology: str,
) -> None:
    core = ReservoirCore.from_config(
        ReservoirConfig(
            input_dim=2,
            n_cells=16,
            topology=topology,
            feature_mode="state_slow_raw",
            seed=23,
            dtype="float64",
        )
    )

    assert core._ring_fast_path is None


def test_custom_ring_shortcuts_edge_list_falls_back_to_sparse_path() -> None:
    dtype = np.dtype("float64")
    config = ReservoirConfig(
        input_dim=2,
        n_cells=4,
        topology="ring_shortcuts",
        feature_mode="state_slow_raw",
        seed=29,
        dtype="float64",
    )
    edges = EdgeList(
        n_nodes=4,
        sources=np.array([0, 1], dtype=np.int64),
        targets=np.array([1, 2], dtype=np.int64),
        weights=np.array([0.5, -0.25], dtype=dtype),
    )
    core = ReservoirCore(
        config=config,
        recurrent_edges=edges,
        input_weights=np.ones((4, 2), dtype=dtype),
        state=ReservoirState.zeros(n_cells=4, dtype="float64"),
    )

    assert core._ring_fast_path is None
    assert core.step((0.2, -0.1)).samples_seen == 1


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("n_cells", [2, 3, 16])
def test_ring_shortcuts_fast_path_matches_sparse_drive(
    dtype: str,
    n_cells: int,
) -> None:
    core = _ring_core(n_cells=n_cells, dtype=dtype)
    plan = core._ring_fast_path
    assert plan is not None
    state = np.linspace(-0.7, 0.9, n_cells, dtype=np.dtype(dtype))
    expected = _sparse_recurrent_drive(
        edges=core.recurrent_edges,
        state=state,
        dtype=core.input_weights.dtype,
    )
    actual = np.empty(core.recurrent_edges.n_nodes, dtype=core.input_weights.dtype)
    scratch = np.empty_like(actual)
    source_values = np.empty(core.recurrent_edges.n_edges, dtype=core.input_weights.dtype)
    contributions = np.empty(core.recurrent_edges.n_edges, dtype=core.input_weights.dtype)

    returned = _ring_shortcuts_recurrent_drive_into(
        plan=plan,
        state=state,
        out=actual,
        scratch=scratch,
        source_values=source_values,
        contributions=contributions,
    )

    assert returned is actual
    assert np.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("n_cells", [2, 3, 16])
def test_ring_shortcuts_fast_path_step_matches_reference_stream(
    dtype: str,
    n_cells: int,
) -> None:
    core = _ring_core(n_cells=n_cells, dtype=dtype)
    assert core._ring_fast_path is not None
    reference_state = core.state

    for sample in _INPUT_STREAM:
        expected = _reference_step(core, reference_state, sample)
        actual = core.step(sample)

        _assert_state_close(actual, expected)
        reference_state = expected


def test_ring_shortcuts_fast_path_keeps_returned_state_read_only() -> None:
    state = _ring_core(n_cells=16, dtype="float64").step((0.1, -0.2))

    assert not state.activations.flags.writeable
    assert not state.fast_trace.flags.writeable
    assert not state.mid_trace.flags.writeable
    assert not state.slow_trace.flags.writeable


def _ring_core(*, n_cells: int, dtype: str) -> ReservoirCore:
    return ReservoirCore.from_config(
        ReservoirConfig(
            input_dim=2,
            n_cells=n_cells,
            topology="ring_shortcuts",
            feature_mode="state_slow_raw",
            seed=19,
            dtype=dtype,
        )
    )


def _reference_step(
    core: ReservoirCore,
    state: ReservoirState,
    x: Sequence[float],
) -> ReservoirState:
    input_vector = validate_input_vector(
        x,
        input_dim=core.config.input_dim,
        dtype=core.config.dtype,
    )
    input_vector = input_vector.astype(core.input_weights.dtype, copy=False)
    previous = state.activations
    input_drive = core.input_weights @ input_vector
    recurrent_drive = _sparse_recurrent_drive(
        edges=core.recurrent_edges,
        state=previous,
        dtype=core.input_weights.dtype,
    )
    pre_activation = (
        input_drive
        + core.config.recurrent_scale * recurrent_drive
        - core.config.fatigue_rate * previous
    )
    candidate = np.tanh(pre_activation)
    new_activations = (
        (1.0 - core.config.leak_rate) * previous
        + core.config.leak_rate * candidate
    ).astype(core.input_weights.dtype, copy=False)
    trace_config = core.config.trace
    return ReservoirState(
        activations=new_activations,
        fast_trace=_reference_trace(
            state.fast_trace,
            new_activations,
            trace_config.fast_decay,
        ),
        mid_trace=_reference_trace(
            state.mid_trace,
            new_activations,
            trace_config.mid_decay,
        ),
        slow_trace=_reference_trace(
            state.slow_trace,
            new_activations,
            trace_config.slow_decay,
        ),
        samples_seen=state.samples_seen + 1,
    )


def _reference_trace(
    old_trace: np.ndarray,
    state: np.ndarray,
    decay: float,
) -> np.ndarray:
    return (decay * old_trace + (1.0 - decay) * state).astype(
        state.dtype,
        copy=False,
    )


def _assert_state_close(actual: ReservoirState, expected: ReservoirState) -> None:
    assert np.allclose(actual.activations, expected.activations, atol=1e-6)
    assert np.allclose(actual.fast_trace, expected.fast_trace, atol=1e-6)
    assert np.allclose(actual.mid_trace, expected.mid_trace, atol=1e-6)
    assert np.allclose(actual.slow_trace, expected.slow_trace, atol=1e-6)
    assert actual.samples_seen == expected.samples_seen
