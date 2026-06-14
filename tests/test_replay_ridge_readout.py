from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from adaptive_reservoir.readout import ReadoutSnapshot, ReplayRidgeReadout


def test_replay_ridge_predict_starts_from_zero() -> None:
    readout = ReplayRidgeReadout(feature_dim=2)

    assert readout.predict([1.0, -2.0]) == 0.0
    assert readout.samples_seen == 0
    assert readout.buffer_count == 0
    assert readout.bias == 0.0
    np.testing.assert_allclose(readout.weights, np.zeros(2))


def test_replay_ridge_refits_after_update_when_interval_is_one() -> None:
    readout = ReplayRidgeReadout(feature_dim=2, refit_interval=1, alpha=1e-6)
    features = [1.0, 0.0]
    target = 1.0

    before = readout.predict(features)
    readout.update(features, target)
    after = readout.predict(features)

    assert readout.samples_seen == 1
    assert readout.buffer_count == 1
    assert abs(target - after) < abs(target - before)


def test_replay_ridge_refit_interval_delays_weight_update() -> None:
    readout = ReplayRidgeReadout(feature_dim=1, refit_interval=2, alpha=1e-6)

    readout.update([1.0], 3.0)
    after_first = readout.predict([1.0])

    readout.update([2.0], 5.0)
    after_second = readout.predict([1.0])

    assert after_first == 0.0
    assert after_second != 0.0
    assert readout.samples_seen == 2
    assert readout.buffer_count == 2


def test_replay_ridge_buffer_respects_buffer_size() -> None:
    readout = ReplayRidgeReadout(feature_dim=1, buffer_size=2, alpha=1e-6)

    readout.update([0.0], 1.0)
    readout.update([1.0], 3.0)
    readout.update([2.0], 5.0)
    snapshot = readout.snapshot()

    assert readout.buffer_count == 2
    assert snapshot.state["features_buffer"] == ((1.0,), (2.0,))
    assert snapshot.state["targets_buffer"] == (3.0, 5.0)


def test_replay_ridge_learns_simple_linear_mapping() -> None:
    readout = ReplayRidgeReadout(
        feature_dim=1,
        buffer_size=8,
        refit_interval=1,
        alpha=1e-6,
    )

    for features, target in [
        ([0.0], 1.0),
        ([1.0], 3.0),
        ([2.0], 5.0),
        ([3.0], 7.0),
    ]:
        readout.update(features, target)

    assert readout.predict([4.0]) == pytest.approx(9.0, abs=1e-2)


def test_replay_ridge_rejects_invalid_constructor_values() -> None:
    with pytest.raises(ValueError, match="feature_dim must be a positive integer"):
        ReplayRidgeReadout(feature_dim=0)
    with pytest.raises(ValueError, match="buffer_size must be a positive integer"):
        ReplayRidgeReadout(feature_dim=1, buffer_size=0)
    with pytest.raises(ValueError, match="refit_interval must be a positive integer"):
        ReplayRidgeReadout(feature_dim=1, refit_interval=0)
    with pytest.raises(ValueError, match="alpha must be finite and positive"):
        ReplayRidgeReadout(feature_dim=1, alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be finite and positive"):
        ReplayRidgeReadout(feature_dim=1, alpha=float("nan"))


def test_replay_ridge_rejects_non_floating_dtype() -> None:
    with pytest.raises(ValueError, match="dtype must be a floating dtype"):
        ReplayRidgeReadout(feature_dim=1, dtype="int64")


def test_replay_ridge_rejects_wrong_feature_dim() -> None:
    readout = ReplayRidgeReadout(feature_dim=2)

    with pytest.raises(ValueError, match="expected feature_dim=2, got 1"):
        readout.predict([1.0])

    with pytest.raises(ValueError, match="expected feature_dim=2, got 1"):
        readout.update([1.0], 1.0)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_replay_ridge_rejects_non_finite_features(bad_value: float) -> None:
    readout = ReplayRidgeReadout(feature_dim=2)

    with pytest.raises(ValueError, match="features must contain only finite values"):
        readout.predict([bad_value, 1.0])


@pytest.mark.parametrize("bad_target", [float("nan"), float("inf"), -float("inf")])
def test_replay_ridge_rejects_non_finite_target(bad_target: float) -> None:
    readout = ReplayRidgeReadout(feature_dim=2)

    with pytest.raises(ValueError, match="target must be finite"):
        readout.update([1.0, 0.0], bad_target)


def test_replay_ridge_weights_property_is_read_only_copy() -> None:
    readout = ReplayRidgeReadout(feature_dim=2, alpha=1e-6)
    readout.update([1.0, 0.0], 1.0)

    weights = readout.weights

    assert weights.flags.writeable is False
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        weights[0] = 100.0
    assert readout.predict([1.0, 0.0]) != 100.0


def test_replay_ridge_snapshot_contains_numeric_state_only() -> None:
    readout = ReplayRidgeReadout(
        feature_dim=2,
        buffer_size=3,
        refit_interval=2,
        alpha=1e-6,
    )
    readout.update([1.0, -1.0], 0.5)

    snapshot = readout.snapshot()

    assert snapshot.name == "replay_ridge"
    assert snapshot.schema_version == 1
    assert set(snapshot.state) == {
        "alpha",
        "bias",
        "buffer_size",
        "dtype",
        "feature_dim",
        "features_buffer",
        "refit_interval",
        "samples_seen",
        "targets_buffer",
        "weights",
    }
    assert "raw_inputs" not in snapshot.state
    assert "target_history" not in snapshot.state
    assert "messages" not in snapshot.state
    assert "policy_decision" not in snapshot.state


def test_replay_ridge_snapshot_uses_tuple_buffers() -> None:
    readout = ReplayRidgeReadout(feature_dim=2)
    readout.update([1.0, -1.0], 0.5)
    snapshot = readout.snapshot()

    assert snapshot.state["features_buffer"] == ((1.0, -1.0),)
    assert snapshot.state["targets_buffer"] == (0.5,)
    with pytest.raises(TypeError):
        snapshot.state["features_buffer"] = ()  # type: ignore[index]


def test_replay_ridge_snapshot_is_independent_from_future_updates() -> None:
    readout = ReplayRidgeReadout(feature_dim=1, alpha=1e-6)
    readout.update([1.0], 3.0)
    snapshot = readout.snapshot()
    expected_features = snapshot.state["features_buffer"]
    expected_targets = snapshot.state["targets_buffer"]

    readout.update([2.0], 5.0)

    assert snapshot.state["features_buffer"] == expected_features
    assert snapshot.state["targets_buffer"] == expected_targets


def test_replay_ridge_restore_recovers_prediction() -> None:
    readout = ReplayRidgeReadout(
        feature_dim=1,
        buffer_size=4,
        refit_interval=1,
        alpha=1e-6,
    )
    for features, target in [([0.0], 1.0), ([1.0], 3.0), ([2.0], 5.0)]:
        readout.update(features, target)
    snapshot = readout.snapshot()
    expected = readout.predict([3.0])

    readout.update([10.0], -10.0)
    readout.restore(snapshot)
    actual = readout.predict([3.0])

    assert actual == pytest.approx(expected)
    assert readout.samples_seen == snapshot.state["samples_seen"]
    assert readout.buffer_count == len(snapshot.state["targets_buffer"])


def test_replay_ridge_restore_rejects_wrong_snapshot_type() -> None:
    readout = ReplayRidgeReadout(feature_dim=2)

    with pytest.raises(TypeError, match="snapshot must be a ReadoutSnapshot"):
        readout.restore({})  # type: ignore[arg-type]


def test_replay_ridge_restore_rejects_wrong_snapshot_name() -> None:
    readout = ReplayRidgeReadout(feature_dim=2)
    snapshot = replace(readout.snapshot(), name="other")

    with pytest.raises(ValueError, match="snapshot name must be 'replay_ridge'"):
        readout.restore(snapshot)


def test_replay_ridge_restore_rejects_incompatible_feature_dim() -> None:
    readout = ReplayRidgeReadout(feature_dim=2)
    snapshot = ReadoutSnapshot(
        schema_version=1,
        name="replay_ridge",
        state={
            "feature_dim": 3,
            "buffer_size": 2,
            "refit_interval": 1,
            "alpha": 1e-3,
            "dtype": "float64",
            "weights": (0.0, 0.0, 0.0),
            "bias": 0.0,
            "samples_seen": 0,
            "features_buffer": (),
            "targets_buffer": (),
        },
    )

    with pytest.raises(ValueError, match="snapshot feature_dim must match 2; got 3"):
        readout.restore(snapshot)


def test_replay_ridge_restore_rejects_incompatible_config() -> None:
    readout = ReplayRidgeReadout(
        feature_dim=2,
        buffer_size=2,
        refit_interval=1,
        alpha=1e-3,
    )
    snapshot = readout.snapshot()

    bad_buffer = replace(snapshot, state={**snapshot.state, "buffer_size": 3})
    with pytest.raises(ValueError, match="snapshot buffer_size must match 2; got 3"):
        readout.restore(bad_buffer)

    bad_interval = replace(snapshot, state={**snapshot.state, "refit_interval": 2})
    with pytest.raises(ValueError, match="snapshot refit_interval must match 1; got 2"):
        readout.restore(bad_interval)

    bad_alpha = replace(snapshot, state={**snapshot.state, "alpha": 1e-2})
    with pytest.raises(ValueError, match="snapshot alpha must match current readout"):
        readout.restore(bad_alpha)

    bad_dtype = replace(snapshot, state={**snapshot.state, "dtype": "float32"})
    with pytest.raises(ValueError, match="snapshot dtype must match 'float64'"):
        readout.restore(bad_dtype)


def test_replay_ridge_restore_rejects_bad_weights_shape() -> None:
    readout = ReplayRidgeReadout(feature_dim=2)
    snapshot = readout.snapshot()
    bad_snapshot = replace(snapshot, state={**snapshot.state, "weights": (0.0,)})

    with pytest.raises(ValueError, match="expected feature_dim=2, got 1"):
        readout.restore(bad_snapshot)


def test_replay_ridge_restore_rejects_bad_buffer_shape() -> None:
    readout = ReplayRidgeReadout(feature_dim=2, buffer_size=2)
    snapshot = readout.snapshot()
    bad_snapshot = replace(
        snapshot,
        state={**snapshot.state, "features_buffer": ((1.0,),), "targets_buffer": (1.0,)},
    )

    with pytest.raises(ValueError, match="expected feature_dim=2, got 1"):
        readout.restore(bad_snapshot)


def test_replay_ridge_restore_rejects_mismatched_buffer_lengths() -> None:
    readout = ReplayRidgeReadout(feature_dim=2)
    snapshot = readout.snapshot()
    bad_snapshot = replace(
        snapshot,
        state={**snapshot.state, "features_buffer": ((1.0, 0.0),), "targets_buffer": ()},
    )

    with pytest.raises(
        ValueError,
        match="snapshot state.targets_buffer length must match features_buffer length",
    ):
        readout.restore(bad_snapshot)


def test_replay_ridge_restore_rejects_oversized_buffer() -> None:
    readout = ReplayRidgeReadout(feature_dim=1, buffer_size=1)
    snapshot = readout.snapshot()
    bad_snapshot = replace(
        snapshot,
        state={
            **snapshot.state,
            "features_buffer": ((1.0,), (2.0,)),
            "targets_buffer": (1.0, 2.0),
        },
    )

    with pytest.raises(
        ValueError,
        match="snapshot state.features_buffer must not exceed buffer_size",
    ):
        readout.restore(bad_snapshot)


def test_replay_ridge_restore_rejects_non_finite_buffer_values() -> None:
    readout = ReplayRidgeReadout(feature_dim=1)
    snapshot = readout.snapshot()
    bad_features = replace(
        snapshot,
        state={**snapshot.state, "features_buffer": ((float("nan"),),), "targets_buffer": (1.0,)},
    )
    with pytest.raises(ValueError, match="features must contain only finite values"):
        readout.restore(bad_features)

    bad_targets = replace(
        snapshot,
        state={**snapshot.state, "features_buffer": ((1.0,),), "targets_buffer": (float("inf"),)},
    )
    with pytest.raises(ValueError, match="snapshot state.targets_buffer values must be finite"):
        readout.restore(bad_targets)


def test_replay_ridge_restore_rejects_samples_seen_less_than_buffer_length() -> None:
    readout = ReplayRidgeReadout(feature_dim=1)
    snapshot = readout.snapshot()
    bad_snapshot = replace(
        snapshot,
        state={
            **snapshot.state,
            "samples_seen": 0,
            "features_buffer": ((1.0,),),
            "targets_buffer": (1.0,),
        },
    )

    with pytest.raises(
        ValueError,
        match="snapshot samples_seen must be at least the replay buffer length",
    ):
        readout.restore(bad_snapshot)
