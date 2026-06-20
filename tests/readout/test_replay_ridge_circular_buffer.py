from __future__ import annotations

import math

import numpy as np
import pytest

from adaptive_reservoir.readout.base import (
    READOUT_SNAPSHOT_SCHEMA_VERSION,
    ReadoutSnapshot,
)
from adaptive_reservoir.readout.replay_ridge import (
    REPLAY_RIDGE_READOUT_NAME,
    ReplayRidgeReadout,
)


def test_replay_ridge_uses_numpy_circular_buffers() -> None:
    readout = ReplayRidgeReadout(feature_dim=2, buffer_size=4, dtype="float64")

    assert isinstance(readout._features_buffer, np.ndarray)
    assert isinstance(readout._targets_buffer, np.ndarray)
    assert readout._features_buffer.shape == (4, 2)
    assert readout._targets_buffer.shape == (4,)
    assert readout.buffer_count == 0


def test_replay_ridge_reuses_buffer_arrays_between_updates() -> None:
    readout = ReplayRidgeReadout(feature_dim=2, buffer_size=3, dtype="float64")
    features_buffer_id = id(readout._features_buffer)
    targets_buffer_id = id(readout._targets_buffer)

    for index in range(8):
        readout.update(_features(index), _target(index))

    assert id(readout._features_buffer) == features_buffer_id
    assert id(readout._targets_buffer) == targets_buffer_id
    assert readout.buffer_count == 3


def test_replay_ridge_circular_snapshot_keeps_latest_samples_in_logical_order() -> None:
    readout = ReplayRidgeReadout(
        feature_dim=2,
        buffer_size=3,
        refit_interval=99,
        dtype="float64",
    )

    for index in range(5):
        readout.update(_features(index), _target(index))

    snapshot = readout.snapshot()

    assert snapshot.state["features_buffer"] == (
        _features(2),
        _features(3),
        _features(4),
    )
    assert snapshot.state["targets_buffer"] == (
        _target(2),
        _target(3),
        _target(4),
    )
    assert readout.buffer_count == 3


@pytest.mark.parametrize(("dtype", "atol"), [("float32", 1e-5), ("float64", 1e-10)])
def test_replay_ridge_circular_refit_matches_legacy_list_vstack_reference(
    dtype: str,
    atol: float,
) -> None:
    readout = ReplayRidgeReadout(
        feature_dim=2,
        buffer_size=4,
        refit_interval=1,
        alpha=1e-3,
        dtype=dtype,
    )
    reference_features: list[np.ndarray] = []
    reference_targets: list[float] = []

    for index in range(8):
        sample = _features(index)
        target = _target(index)
        _legacy_append(
            reference_features,
            reference_targets,
            sample,
            target,
            buffer_size=4,
            dtype=dtype,
        )
        expected_weights, expected_bias = _legacy_refit(
            reference_features,
            reference_targets,
            feature_dim=2,
            alpha=1e-3,
            dtype=dtype,
        )

        readout.update(sample, target)

        assert readout.solve_count == index + 1
        assert readout.buffer_count == min(index + 1, 4)
        assert np.allclose(readout.weights, expected_weights, atol=atol)
        assert readout.bias == pytest.approx(expected_bias, abs=atol)


def test_replay_ridge_snapshot_restore_roundtrip_after_wrap() -> None:
    readout = ReplayRidgeReadout(
        feature_dim=2,
        buffer_size=3,
        refit_interval=1,
        alpha=1e-3,
        dtype="float64",
    )
    for index in range(5):
        readout.update(_features(index), _target(index))
    snapshot = readout.snapshot()

    restored = ReplayRidgeReadout(
        feature_dim=2,
        buffer_size=3,
        refit_interval=1,
        alpha=1e-3,
        dtype="float64",
    )
    restored.restore(snapshot)

    assert restored.snapshot().to_dict() == snapshot.to_dict()
    assert restored.predict((0.25, -0.75)) == pytest.approx(readout.predict((0.25, -0.75)))

    readout.update(_features(9), _target(9))
    restored.update(_features(9), _target(9))

    assert restored.snapshot().to_dict() == readout.snapshot().to_dict()


def test_replay_ridge_float32_resume_refits_in_logical_order_after_wrap() -> None:
    readout = ReplayRidgeReadout(
        feature_dim=3,
        buffer_size=4,
        refit_interval=1,
        alpha=1e-2,
        dtype="float32",
    )
    for features, target in _ill_conditioned_stream():
        readout.update(features, target)
    restored = ReplayRidgeReadout(
        feature_dim=3,
        buffer_size=4,
        refit_interval=1,
        alpha=1e-2,
        dtype="float32",
    )
    restored.restore(readout.snapshot())

    next_features = (1.0e6, -1.0e-3, 3.0)
    next_target = -2.5e3
    readout.update(next_features, next_target)
    restored.update(next_features, next_target)

    assert restored.snapshot().to_dict() == readout.snapshot().to_dict()


def test_replay_ridge_float32_delayed_snapshot_keeps_large_finite_target() -> None:
    readout = ReplayRidgeReadout(
        feature_dim=1,
        buffer_size=2,
        refit_interval=99,
        dtype="float32",
    )

    readout.update((1.0,), 1.0e39)
    snapshot = readout.snapshot()

    target = snapshot.state["targets_buffer"][0]
    assert math.isfinite(target)
    assert target == pytest.approx(1.0e39)
    # ReadoutSnapshot.to_dict() returns a JSON-friendly list, not the raw tuple state.
    assert snapshot.to_dict()["state"]["targets_buffer"] == [1.0e39]


def test_replay_ridge_restores_old_logical_snapshot_format() -> None:
    snapshot = ReadoutSnapshot(
        schema_version=READOUT_SNAPSHOT_SCHEMA_VERSION,
        name=REPLAY_RIDGE_READOUT_NAME,
        state={
            "feature_dim": 2,
            "buffer_size": 3,
            "refit_interval": 1,
            "alpha": 1e-3,
            "dtype": "float64",
            "weights": (1.0, -0.5),
            "bias": 0.25,
            "samples_seen": 2,
            "solve_count": 1,
            "features_buffer": (_features(0), _features(1)),
            "targets_buffer": (_target(0), _target(1)),
        },
    )
    readout = ReplayRidgeReadout(
        feature_dim=2,
        buffer_size=3,
        refit_interval=1,
        alpha=1e-3,
        dtype="float64",
    )

    readout.restore(snapshot)

    assert readout.snapshot().to_dict() == snapshot.to_dict()
    assert readout.predict((2.0, 4.0)) == pytest.approx(0.25)


def test_replay_ridge_weights_remain_read_only_after_circular_refit() -> None:
    readout = ReplayRidgeReadout(feature_dim=2, buffer_size=3, dtype="float64")

    for index in range(4):
        readout.update(_features(index), _target(index))

    assert not readout.weights.flags.writeable


def _features(index: int) -> tuple[float, float]:
    return (float(index), float(index) + 0.5)


def _target(index: int) -> float:
    return float(index) - 0.25


def _ill_conditioned_stream() -> tuple[tuple[tuple[float, float, float], float], ...]:
    return (
        ((1.0e6, 1.0e-3, -1.0), 2.0e3),
        ((-1.0e6, 2.0e-3, 1.5), -2.0e3),
        ((5.0e5, -1.0e-3, 2.0), 1.0e3),
        ((-5.0e5, -2.0e-3, -2.5), -1.0e3),
        ((2.5e5, 3.0e-3, 4.0), 5.0e2),
        ((-2.5e5, -3.0e-3, -4.5), -5.0e2),
    )


def _legacy_append(
    features_buffer: list[np.ndarray],
    targets_buffer: list[float],
    features: tuple[float, float],
    target: float,
    *,
    buffer_size: int,
    dtype: str,
) -> None:
    if len(features_buffer) >= buffer_size:
        features_buffer.pop(0)
        targets_buffer.pop(0)
    features_buffer.append(np.asarray(features, dtype=dtype))
    targets_buffer.append(target)


def _legacy_refit(
    features_buffer: list[np.ndarray],
    targets_buffer: list[float],
    *,
    feature_dim: int,
    alpha: float,
    dtype: str,
) -> tuple[np.ndarray, float]:
    features = np.vstack(features_buffer).astype(dtype, copy=False)
    targets = np.asarray(targets_buffer, dtype=dtype)
    bias_column = np.ones((features.shape[0], 1), dtype=dtype)
    design = np.hstack((features, bias_column))
    penalty = np.eye(feature_dim + 1, dtype=dtype) * alpha
    penalty[-1, -1] = 0.0
    normal_matrix = design.T @ design + penalty
    rhs = design.T @ targets
    coefficients = np.linalg.solve(normal_matrix, rhs)
    return np.asarray(coefficients[:-1], dtype=dtype), float(coefficients[-1])
