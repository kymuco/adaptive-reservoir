from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from adaptive_reservoir.readout import ReadoutSnapshot, SlidingWindowRidgeReadout


def _state(snapshot: ReadoutSnapshot, **updates: object) -> dict[str, object]:
    return {**snapshot.state, **updates}


def test_sliding_ridge_predict_starts_from_zero() -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=2)

    assert readout.predict([1.0, -2.0]) == 0.0
    assert readout.samples_seen == 0
    assert readout.window_count == 0
    assert readout.bias == 0.0
    np.testing.assert_allclose(readout.weights, np.zeros(2))


def test_sliding_ridge_updates_after_interval_one() -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=2, update_interval=1, alpha=1e-6)
    features = [1.0, 0.0]
    target = 1.0

    before = readout.predict(features)
    readout.update(features, target)
    after = readout.predict(features)

    assert readout.samples_seen == 1
    assert readout.window_count == 1
    assert abs(target - after) < abs(target - before)


def test_sliding_ridge_update_interval_delays_refit() -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=1, update_interval=2, alpha=1e-6)

    readout.update([1.0], 3.0)
    after_first = readout.predict([1.0])

    readout.update([2.0], 5.0)
    after_second = readout.predict([1.0])

    assert after_first == 0.0
    assert after_second != 0.0
    assert readout.samples_seen == 2
    assert readout.window_count == 2


def test_sliding_ridge_window_respects_window_size() -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=1, window_size=2, alpha=1e-6)

    readout.update([0.0], 1.0)
    readout.update([1.0], 3.0)
    readout.update([2.0], 5.0)
    snapshot = readout.snapshot()

    assert readout.window_count == 2
    assert snapshot.state["features_window"] == ((1.0,), (2.0,))
    assert snapshot.state["targets_window"] == (3.0, 5.0)


def test_sliding_ridge_learns_recent_linear_mapping() -> None:
    readout = SlidingWindowRidgeReadout(
        feature_dim=1,
        window_size=8,
        update_interval=1,
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


def test_sliding_ridge_forgets_old_samples() -> None:
    readout = SlidingWindowRidgeReadout(
        feature_dim=1,
        window_size=3,
        update_interval=1,
        alpha=1e-6,
    )

    for features, target in [([0.0], 1.0), ([1.0], 3.0), ([2.0], 5.0)]:
        readout.update(features, target)
    old_prediction = readout.predict([3.0])

    for features, target in [([0.0], 0.0), ([1.0], -1.0), ([2.0], -2.0)]:
        readout.update(features, target)
    new_prediction = readout.predict([3.0])
    snapshot = readout.snapshot()

    assert old_prediction > 0.0
    assert new_prediction == pytest.approx(-3.0, abs=1e-2)
    assert snapshot.state["features_window"] == ((0.0,), (1.0,), (2.0,))
    assert snapshot.state["targets_window"] == (0.0, -1.0, -2.0)


def test_sliding_ridge_rejects_invalid_constructor_values() -> None:
    with pytest.raises(ValueError, match="feature_dim must be a positive integer"):
        SlidingWindowRidgeReadout(feature_dim=0)
    with pytest.raises(ValueError, match="window_size must be a positive integer"):
        SlidingWindowRidgeReadout(feature_dim=1, window_size=0)
    with pytest.raises(ValueError, match="update_interval must be a positive integer"):
        SlidingWindowRidgeReadout(feature_dim=1, update_interval=0)
    with pytest.raises(ValueError, match="alpha must be finite and positive"):
        SlidingWindowRidgeReadout(feature_dim=1, alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be finite and positive"):
        SlidingWindowRidgeReadout(feature_dim=1, alpha=float("nan"))


def test_sliding_ridge_rejects_non_floating_dtype() -> None:
    with pytest.raises(ValueError, match="dtype must be a floating dtype"):
        SlidingWindowRidgeReadout(feature_dim=1, dtype="int64")


def test_sliding_ridge_rejects_linalg_unsupported_dtype() -> None:
    with pytest.raises(ValueError, match="dtype must be one of: float32, float64"):
        SlidingWindowRidgeReadout(feature_dim=1, dtype="float16")


def test_sliding_ridge_rejects_wrong_feature_dim() -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=2)

    with pytest.raises(ValueError, match="expected feature_dim=2, got 1"):
        readout.predict([1.0])

    with pytest.raises(ValueError, match="expected feature_dim=2, got 1"):
        readout.update([1.0], 1.0)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_sliding_ridge_rejects_non_finite_features(bad_value: float) -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=2)

    with pytest.raises(ValueError, match="features must contain only finite values"):
        readout.predict([bad_value, 1.0])


@pytest.mark.parametrize("bad_target", [float("nan"), float("inf"), -float("inf")])
def test_sliding_ridge_rejects_non_finite_target(bad_target: float) -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=2)

    with pytest.raises(ValueError, match="target must be finite"):
        readout.update([1.0, 0.0], bad_target)


def test_sliding_ridge_weights_property_is_read_only_copy() -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=2, alpha=1e-6)
    readout.update([1.0, 0.0], 1.0)

    weights = readout.weights

    assert weights.flags.writeable is False
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        weights[0] = 100.0
    assert readout.predict([1.0, 0.0]) != 100.0


def test_sliding_ridge_snapshot_contains_numeric_state_only() -> None:
    readout = SlidingWindowRidgeReadout(
        feature_dim=2,
        window_size=3,
        update_interval=2,
        alpha=1e-6,
    )
    readout.update([1.0, -1.0], 0.5)

    snapshot = readout.snapshot()

    assert snapshot.name == "sliding_ridge"
    assert snapshot.schema_version == 1
    assert set(snapshot.state) == {
        "alpha",
        "bias",
        "dtype",
        "feature_dim",
        "features_window",
        "samples_seen",
        "targets_window",
        "update_interval",
        "weights",
        "window_size",
    }
    assert "raw_inputs" not in snapshot.state
    assert "target_history" not in snapshot.state
    assert "messages" not in snapshot.state
    assert "policy_decision" not in snapshot.state


def test_sliding_ridge_snapshot_uses_tuple_windows() -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=2)
    readout.update([1.0, -1.0], 0.5)
    snapshot = readout.snapshot()

    assert snapshot.state["features_window"] == ((1.0, -1.0),)
    assert snapshot.state["targets_window"] == (0.5,)
    with pytest.raises(TypeError):
        snapshot.state["features_window"] = ()  # type: ignore[index]


def test_sliding_ridge_snapshot_is_independent_from_future_updates() -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=1, alpha=1e-6)
    readout.update([1.0], 3.0)
    snapshot = readout.snapshot()
    expected_features = snapshot.state["features_window"]
    expected_targets = snapshot.state["targets_window"]

    readout.update([2.0], 5.0)

    assert snapshot.state["features_window"] == expected_features
    assert snapshot.state["targets_window"] == expected_targets


def test_sliding_ridge_restore_recovers_prediction() -> None:
    readout = SlidingWindowRidgeReadout(
        feature_dim=1,
        window_size=4,
        update_interval=1,
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
    assert readout.window_count == len(snapshot.state["targets_window"])


def test_sliding_ridge_restore_rejects_wrong_snapshot_type() -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=2)

    with pytest.raises(TypeError, match="snapshot must be a ReadoutSnapshot"):
        readout.restore({})  # type: ignore[arg-type]


def test_sliding_ridge_restore_rejects_wrong_snapshot_name() -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=2)
    snapshot = replace(readout.snapshot(), name="other")

    with pytest.raises(ValueError, match="snapshot name must be 'sliding_ridge'"):
        readout.restore(snapshot)


def test_sliding_ridge_restore_rejects_incompatible_feature_dim() -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=2)
    snapshot = ReadoutSnapshot(
        schema_version=1,
        name="sliding_ridge",
        state={
            "feature_dim": 3,
            "window_size": 2,
            "alpha": 1e-3,
            "update_interval": 1,
            "dtype": "float64",
            "weights": (0.0, 0.0, 0.0),
            "bias": 0.0,
            "samples_seen": 0,
            "features_window": (),
            "targets_window": (),
        },
    )

    with pytest.raises(ValueError, match="snapshot feature_dim must match 2; got 3"):
        readout.restore(snapshot)


def test_sliding_ridge_restore_rejects_incompatible_config() -> None:
    readout = SlidingWindowRidgeReadout(
        feature_dim=2,
        window_size=2,
        update_interval=1,
        alpha=1e-3,
    )
    snapshot = readout.snapshot()

    bad_window = replace(snapshot, state=_state(snapshot, window_size=3))
    with pytest.raises(ValueError, match="snapshot window_size must match 2; got 3"):
        readout.restore(bad_window)

    bad_interval = replace(snapshot, state=_state(snapshot, update_interval=2))
    with pytest.raises(ValueError, match="snapshot update_interval must match 1; got 2"):
        readout.restore(bad_interval)

    bad_alpha = replace(snapshot, state=_state(snapshot, alpha=1e-2))
    with pytest.raises(ValueError, match="snapshot alpha must match current readout"):
        readout.restore(bad_alpha)

    bad_dtype = replace(snapshot, state=_state(snapshot, dtype="float32"))
    with pytest.raises(ValueError, match="snapshot dtype must match 'float64'"):
        readout.restore(bad_dtype)


def test_sliding_ridge_restore_rejects_bad_weights_shape() -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=2)
    snapshot = readout.snapshot()
    bad_snapshot = replace(snapshot, state=_state(snapshot, weights=(0.0,)))

    with pytest.raises(ValueError, match="expected feature_dim=2, got 1"):
        readout.restore(bad_snapshot)


def test_sliding_ridge_restore_rejects_bad_window_shape() -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=2, window_size=2)
    snapshot = readout.snapshot()
    bad_snapshot = replace(
        snapshot,
        state=_state(
            snapshot,
            features_window=((1.0,),),
            targets_window=(1.0,),
        ),
    )

    with pytest.raises(ValueError, match="expected feature_dim=2, got 1"):
        readout.restore(bad_snapshot)


def test_sliding_ridge_restore_rejects_mismatched_window_lengths() -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=2)
    snapshot = readout.snapshot()
    bad_snapshot = replace(
        snapshot,
        state=_state(snapshot, features_window=((1.0, 0.0),), targets_window=()),
    )

    with pytest.raises(
        ValueError,
        match="snapshot state.targets_window length must match features_window length",
    ):
        readout.restore(bad_snapshot)


def test_sliding_ridge_restore_rejects_oversized_window() -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=1, window_size=1)
    snapshot = readout.snapshot()
    bad_snapshot = replace(
        snapshot,
        state=_state(
            snapshot,
            features_window=((1.0,), (2.0,)),
            targets_window=(1.0, 2.0),
        ),
    )

    with pytest.raises(
        ValueError,
        match="snapshot state.features_window must not exceed window_size",
    ):
        readout.restore(bad_snapshot)


def test_sliding_ridge_restore_rejects_non_finite_window_values() -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=1)
    snapshot = readout.snapshot()
    bad_features = replace(
        snapshot,
        state=_state(
            snapshot,
            features_window=((float("nan"),),),
            targets_window=(1.0,),
        ),
    )
    with pytest.raises(ValueError, match="features must contain only finite values"):
        readout.restore(bad_features)

    bad_targets = replace(
        snapshot,
        state=_state(
            snapshot,
            features_window=((1.0,),),
            targets_window=(float("inf"),),
        ),
    )
    with pytest.raises(
        ValueError,
        match="snapshot state.targets_window values must be finite",
    ):
        readout.restore(bad_targets)


def test_sliding_ridge_restore_rejects_samples_seen_less_than_window_length() -> None:
    readout = SlidingWindowRidgeReadout(feature_dim=1)
    snapshot = readout.snapshot()
    bad_snapshot = replace(
        snapshot,
        state=_state(
            snapshot,
            samples_seen=0,
            features_window=((1.0,),),
            targets_window=(1.0,),
        ),
    )

    with pytest.raises(
        ValueError,
        match="snapshot samples_seen must be at least the sliding window length",
    ):
        readout.restore(bad_snapshot)
