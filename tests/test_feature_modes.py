import numpy as np
import pytest

from adaptive_reservoir import ReservoirState, extract_features


def test_state_raw_returns_activations() -> None:
    state = _state()

    features = extract_features(state, "state_raw")

    np.testing.assert_array_equal(features, state.activations)
    assert features.shape == (3,)


def test_state_slow_raw_concatenates_state_and_slow_trace() -> None:
    state = _state()

    features = extract_features(state, "state_slow_raw")

    expected = np.concatenate((state.activations, state.slow_trace))
    np.testing.assert_array_equal(features, expected)
    assert features.shape == (6,)


def test_multi_raw_concatenates_state_fast_mid_and_slow() -> None:
    state = _state()

    features = extract_features(state, "multi_raw")

    expected = np.concatenate(
        (
            state.activations,
            state.fast_trace,
            state.mid_trace,
            state.slow_trace,
        )
    )
    np.testing.assert_array_equal(features, expected)
    assert features.shape == (12,)


def test_feature_mode_preserves_dtype() -> None:
    state = ReservoirState(
        activations=np.array([1.0, 2.0], dtype=np.float32),
        fast_trace=np.array([0.1, 0.2], dtype=np.float32),
        mid_trace=np.array([0.3, 0.4], dtype=np.float32),
        slow_trace=np.array([0.5, 0.6], dtype=np.float32),
    )

    features = extract_features(state, "multi_raw")

    assert features.dtype == np.float32


def test_feature_mode_output_is_read_only() -> None:
    state = _state()

    features = extract_features(state, "state_raw")

    assert not features.flags.writeable
    with pytest.raises(ValueError):
        features[0] = 999.0


def test_feature_mode_output_does_not_mutate_state() -> None:
    state = _state()

    features = extract_features(state, "state_raw")

    assert not np.shares_memory(features, state.activations)


def test_invalid_feature_mode_raises_clear_error() -> None:
    state = _state()

    with pytest.raises(ValueError, match="unsupported feature mode"):
        extract_features(state, "bad")  # type: ignore[arg-type]


def _state() -> ReservoirState:
    return ReservoirState(
        activations=np.array([1.0, 2.0, 3.0]),
        fast_trace=np.array([0.1, 0.2, 0.3]),
        mid_trace=np.array([0.4, 0.5, 0.6]),
        slow_trace=np.array([0.7, 0.8, 0.9]),
        samples_seen=2,
    )
