import numpy as np
import pytest

from adaptive_reservoir import ReservoirState


def test_zero_state_has_expected_shape_and_dtype() -> None:
    state = ReservoirState.zeros(n_cells=4, dtype="float32")

    assert state.activations.shape == (4,)
    assert state.fast_trace.shape == (4,)
    assert state.mid_trace.shape == (4,)
    assert state.slow_trace.shape == (4,)
    assert state.activations.dtype == np.float32
    assert state.samples_seen == 0


def test_zero_state_requires_positive_n_cells() -> None:
    with pytest.raises(ValueError, match="n_cells must be positive"):
        ReservoirState.zeros(n_cells=0)


def test_zero_state_validates_dtype() -> None:
    with pytest.raises(ValueError, match="dtype must be one of"):
        ReservoirState.zeros(n_cells=4, dtype="int32")


def test_state_arrays_are_defensively_copied_and_read_only() -> None:
    source = np.zeros(4, dtype=np.float64)
    state = ReservoirState(
        activations=source,
        fast_trace=source,
        mid_trace=source,
        slow_trace=source,
    )

    source[0] = 1.0

    assert state.activations[0] == 0.0
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        state.activations[0] = 1.0


def test_state_rejects_non_1d_arrays() -> None:
    values = np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="activations must be a 1D array"):
        ReservoirState(
            activations=values,
            fast_trace=np.zeros(4, dtype=np.float64),
            mid_trace=np.zeros(4, dtype=np.float64),
            slow_trace=np.zeros(4, dtype=np.float64),
        )


def test_state_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="must have the same shape"):
        ReservoirState(
            activations=np.zeros(4, dtype=np.float64),
            fast_trace=np.zeros(3, dtype=np.float64),
            mid_trace=np.zeros(4, dtype=np.float64),
            slow_trace=np.zeros(4, dtype=np.float64),
        )


def test_state_rejects_non_floating_arrays() -> None:
    with pytest.raises(ValueError, match="activations must have a floating dtype"):
        ReservoirState(
            activations=np.zeros(4, dtype=np.int64),
            fast_trace=np.zeros(4, dtype=np.float64),
            mid_trace=np.zeros(4, dtype=np.float64),
            slow_trace=np.zeros(4, dtype=np.float64),
        )


def test_state_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="activations must contain only finite values"):
        ReservoirState(
            activations=np.array([0.0, np.nan], dtype=np.float64),
            fast_trace=np.zeros(2, dtype=np.float64),
            mid_trace=np.zeros(2, dtype=np.float64),
            slow_trace=np.zeros(2, dtype=np.float64),
        )


def test_state_rejects_negative_samples_seen() -> None:
    with pytest.raises(ValueError, match="samples_seen must be non-negative"):
        ReservoirState(
            activations=np.zeros(4, dtype=np.float64),
            fast_trace=np.zeros(4, dtype=np.float64),
            mid_trace=np.zeros(4, dtype=np.float64),
            slow_trace=np.zeros(4, dtype=np.float64),
            samples_seen=-1,
        )
