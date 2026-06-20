from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from adaptive_reservoir import AdaptiveReservoir, ReservoirConfig
from adaptive_reservoir.core.reservoir import (
    ReservoirCore,
    _sparse_recurrent_drive,
    _sparse_recurrent_drive_into,
)
from adaptive_reservoir.core.state import ReservoirState
from adaptive_reservoir.core.validation import validate_input_vector


_INPUT_STREAM = (
    (0.1, -0.2),
    (0.3, 0.4),
    (-0.5, 0.2),
    (0.0, -0.7),
    (0.8, 0.1),
)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "topology",
    ["random_sparse", "ring_shortcuts", "modular_small_world"],
)
def test_preallocated_step_matches_reference_stream(dtype: str, topology: str) -> None:
    core = _core(dtype=dtype, topology=topology)
    reference_state = core.state

    for sample in _INPUT_STREAM:
        expected = _reference_step(core, reference_state, sample)
        actual = core.step(sample)

        _assert_state_close(actual, expected)
        assert actual.activations.dtype == np.dtype(dtype)
        assert actual.fast_trace.dtype == np.dtype(dtype)
        assert actual.mid_trace.dtype == np.dtype(dtype)
        assert actual.slow_trace.dtype == np.dtype(dtype)

        reference_state = expected


def test_preallocated_step_keeps_returned_state_read_only() -> None:
    state = _core(dtype="float64", topology="ring_shortcuts").step((0.1, -0.2))

    assert not state.activations.flags.writeable
    assert not state.fast_trace.flags.writeable
    assert not state.mid_trace.flags.writeable
    assert not state.slow_trace.flags.writeable


def test_reservoir_core_reuses_work_buffers_between_steps() -> None:
    core = _core(dtype="float64", topology="ring_shortcuts")
    buffer_ids = _work_buffer_ids(core)

    core.step((0.1, -0.2))
    core.step((0.3, 0.4))

    assert _work_buffer_ids(core) == buffer_ids


def test_sparse_recurrent_drive_into_matches_allocating_reference() -> None:
    core = _core(dtype="float64", topology="modular_small_world")
    expected = _sparse_recurrent_drive(
        edges=core.recurrent_edges,
        state=core.state.activations,
        dtype=core.input_weights.dtype,
    )
    actual = np.empty(core.recurrent_edges.n_nodes, dtype=core.input_weights.dtype)
    source_values = np.empty(core.recurrent_edges.n_edges, dtype=core.input_weights.dtype)
    contributions = np.empty(core.recurrent_edges.n_edges, dtype=core.input_weights.dtype)

    returned = _sparse_recurrent_drive_into(
        edges=core.recurrent_edges,
        state=core.state.activations,
        dtype=core.input_weights.dtype,
        out=actual,
        source_values=source_values,
        contributions=contributions,
    )

    assert returned is actual
    assert np.allclose(actual, expected)


def test_adaptive_reservoir_predictions_remain_deterministic() -> None:
    config = ReservoirConfig(
        input_dim=2,
        n_cells=16,
        topology="ring_shortcuts",
        feature_mode="state_slow_raw",
        seed=17,
        dtype="float64",
    )
    first = AdaptiveReservoir(config)
    second = AdaptiveReservoir(config)
    targets = (0.2, -0.1, 0.4, -0.3, 0.0)

    first_predictions = tuple(
        first.step(sample, target=target).prediction
        for sample, target in zip(_INPUT_STREAM, targets, strict=True)
    )
    second_predictions = tuple(
        second.step(sample, target=target).prediction
        for sample, target in zip(_INPUT_STREAM, targets, strict=True)
    )

    assert first_predictions == second_predictions


def _core(*, dtype: str, topology: str) -> ReservoirCore:
    return ReservoirCore.from_config(
        ReservoirConfig(
            input_dim=2,
            n_cells=16,
            topology=topology,
            feature_mode="state_slow_raw",
            seed=11,
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
    recurrent_drive = _reference_sparse_recurrent_drive(core, previous)
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


def _reference_sparse_recurrent_drive(
    core: ReservoirCore,
    state: np.ndarray,
) -> np.ndarray:
    edges = core.recurrent_edges
    drive = np.zeros(edges.n_nodes, dtype=core.input_weights.dtype)
    contributions = edges.weights.astype(core.input_weights.dtype, copy=False) * state[
        edges.sources
    ]
    np.add.at(drive, edges.targets, contributions)
    return drive


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


def _work_buffer_ids(core: ReservoirCore) -> tuple[int, ...]:
    return (
        id(core._work_input_drive),
        id(core._work_recurrent_drive),
        id(core._work_pre_activation),
        id(core._work_candidate),
        id(core._work_new_activations),
        id(core._work_fast_trace),
        id(core._work_mid_trace),
        id(core._work_slow_trace),
        id(core._work_edge_sources),
        id(core._work_edge_contributions),
    )
