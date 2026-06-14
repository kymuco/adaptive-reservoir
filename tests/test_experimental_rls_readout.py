from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from adaptive_reservoir.experimental import RLSReadout
from adaptive_reservoir.readout import ReadoutSnapshot


def _state(snapshot: ReadoutSnapshot, **updates: object) -> dict[str, object]:
    return {**snapshot.state, **updates}


def test_rls_is_experimental_only() -> None:
    import adaptive_reservoir.readout as readout

    assert not hasattr(readout, "RLSReadout")


def test_rls_predict_starts_from_zero() -> None:
    readout = RLSReadout(feature_dim=2)

    assert readout.predict([1.0, -2.0]) == 0.0
    assert readout.samples_seen == 0
    assert readout.bias == 0.0
    np.testing.assert_allclose(readout.weights, np.zeros(2))
    np.testing.assert_allclose(readout.covariance, np.eye(3) * 1_000.0)


def test_rls_update_changes_prediction_toward_target() -> None:
    readout = RLSReadout(feature_dim=2)
    features = [1.0, 0.0]
    target = 1.0

    before = readout.predict(features)
    readout.update(features, target)
    after = readout.predict(features)

    assert readout.samples_seen == 1
    assert abs(target - after) < abs(target - before)


def test_rls_learns_simple_linear_mapping() -> None:
    readout = RLSReadout(feature_dim=1, forgetting_factor=1.0)

    for features, target in [
        ([0.0], 1.0),
        ([1.0], 3.0),
        ([2.0], 5.0),
        ([3.0], 7.0),
    ]:
        readout.update(features, target)

    assert readout.predict([4.0]) == pytest.approx(9.0, abs=1e-2)


def test_rls_forgetting_factor_allows_recent_regime_adaptation() -> None:
    readout = RLSReadout(
        feature_dim=1,
        forgetting_factor=0.9,
        covariance_scale=1_000.0,
    )

    for features, target in [([0.0], 1.0), ([1.0], 3.0), ([2.0], 5.0)]:
        readout.update(features, target)
    old_prediction = readout.predict([3.0])

    for features, target in [
        ([0.0], 0.0),
        ([1.0], -1.0),
        ([2.0], -2.0),
        ([3.0], -3.0),
        ([4.0], -4.0),
    ]:
        readout.update(features, target)
    new_prediction = readout.predict([5.0])

    assert old_prediction > 0.0
    assert new_prediction < old_prediction
    assert new_prediction < 0.0


def test_rls_rejects_invalid_constructor_values() -> None:
    with pytest.raises(ValueError, match="feature_dim must be a positive integer"):
        RLSReadout(feature_dim=0)
    with pytest.raises(ValueError, match="forgetting_factor must be finite"):
        RLSReadout(feature_dim=1, forgetting_factor=0.0)
    with pytest.raises(ValueError, match="forgetting_factor must be finite"):
        RLSReadout(feature_dim=1, forgetting_factor=1.1)
    with pytest.raises(ValueError, match="covariance_scale must be finite and positive"):
        RLSReadout(feature_dim=1, covariance_scale=0.0)
    with pytest.raises(ValueError, match="jitter must be finite and positive"):
        RLSReadout(feature_dim=1, jitter=float("nan"))


def test_rls_rejects_non_float64_dtype() -> None:
    with pytest.raises(ValueError, match="dtype must be float64 for experimental RLS"):
        RLSReadout(feature_dim=1, dtype="float32")

    with pytest.raises(ValueError, match="dtype must be float64 for experimental RLS"):
        RLSReadout(feature_dim=1, dtype="int64")


def test_rls_rejects_wrong_feature_dim() -> None:
    readout = RLSReadout(feature_dim=2)

    with pytest.raises(ValueError, match="expected feature_dim=2, got 1"):
        readout.predict([1.0])

    with pytest.raises(ValueError, match="expected feature_dim=2, got 1"):
        readout.update([1.0], 1.0)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_rls_rejects_non_finite_features(bad_value: float) -> None:
    readout = RLSReadout(feature_dim=2)

    with pytest.raises(ValueError, match="features must contain only finite values"):
        readout.predict([bad_value, 1.0])


@pytest.mark.parametrize("bad_target", [float("nan"), float("inf"), -float("inf")])
def test_rls_rejects_non_finite_target(bad_target: float) -> None:
    readout = RLSReadout(feature_dim=2)

    with pytest.raises(ValueError, match="target must be finite"):
        readout.update([1.0, 0.0], bad_target)


def test_rls_update_is_atomic_when_target_is_invalid() -> None:
    readout = RLSReadout(feature_dim=2)
    weights_before = readout.weights
    covariance_before = readout.covariance

    with pytest.raises(ValueError, match="target must be finite"):
        readout.update([1.0, 0.0], float("nan"))

    assert readout.samples_seen == 0
    np.testing.assert_allclose(readout.weights, weights_before)
    np.testing.assert_allclose(readout.covariance, covariance_before)


def test_rls_weights_property_is_read_only_copy() -> None:
    readout = RLSReadout(feature_dim=2)
    readout.update([1.0, 0.0], 1.0)

    weights = readout.weights

    assert weights.flags.writeable is False
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        weights[0] = 100.0
    assert readout.predict([1.0, 0.0]) != 100.0


def test_rls_covariance_property_is_read_only_copy() -> None:
    readout = RLSReadout(feature_dim=2)
    readout.update([1.0, 0.0], 1.0)

    covariance = readout.covariance

    assert covariance.flags.writeable is False
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        covariance[0, 0] = 0.0
    assert readout.covariance[0, 0] != 0.0


def test_rls_snapshot_contains_numeric_state_only() -> None:
    readout = RLSReadout(feature_dim=2, forgetting_factor=0.95, jitter=1e-7)
    readout.update([1.0, -1.0], 0.5)

    snapshot = readout.snapshot()

    assert snapshot.name == "experimental_rls"
    assert snapshot.schema_version == 1
    assert set(snapshot.state) == {
        "bias",
        "covariance",
        "covariance_scale",
        "dtype",
        "feature_dim",
        "forgetting_factor",
        "jitter",
        "samples_seen",
        "weights",
    }
    assert "raw_inputs" not in snapshot.state
    assert "target_history" not in snapshot.state
    assert "messages" not in snapshot.state
    assert "policy_decision" not in snapshot.state


def test_rls_snapshot_uses_tuple_covariance() -> None:
    readout = RLSReadout(feature_dim=2)
    readout.update([1.0, -1.0], 0.5)
    snapshot = readout.snapshot()

    covariance = snapshot.state["covariance"]

    assert isinstance(covariance, tuple)
    assert isinstance(covariance[0], tuple)
    with pytest.raises(TypeError):
        snapshot.state["covariance"] = ()  # type: ignore[index]


def test_rls_snapshot_is_independent_from_future_updates() -> None:
    readout = RLSReadout(feature_dim=1)
    readout.update([1.0], 3.0)
    snapshot = readout.snapshot()
    expected_weights = snapshot.state["weights"]
    expected_covariance = snapshot.state["covariance"]

    readout.update([2.0], 5.0)

    assert snapshot.state["weights"] == expected_weights
    assert snapshot.state["covariance"] == expected_covariance


def test_rls_restore_recovers_prediction() -> None:
    readout = RLSReadout(feature_dim=1, forgetting_factor=1.0)
    for features, target in [([0.0], 1.0), ([1.0], 3.0), ([2.0], 5.0)]:
        readout.update(features, target)
    snapshot = readout.snapshot()
    expected = readout.predict([3.0])

    readout.update([10.0], -10.0)
    readout.restore(snapshot)
    actual = readout.predict([3.0])

    assert actual == pytest.approx(expected)
    assert readout.samples_seen == snapshot.state["samples_seen"]


def test_rls_restore_rejects_wrong_snapshot_type() -> None:
    readout = RLSReadout(feature_dim=2)

    with pytest.raises(TypeError, match="snapshot must be a ReadoutSnapshot"):
        readout.restore({})  # type: ignore[arg-type]


def test_rls_restore_rejects_wrong_snapshot_name() -> None:
    readout = RLSReadout(feature_dim=2)
    snapshot = replace(readout.snapshot(), name="other")

    with pytest.raises(ValueError, match="snapshot name must be 'experimental_rls'"):
        readout.restore(snapshot)


def test_rls_restore_rejects_incompatible_feature_dim() -> None:
    readout = RLSReadout(feature_dim=2)
    snapshot = ReadoutSnapshot(
        schema_version=1,
        name="experimental_rls",
        state={
            "feature_dim": 3,
            "forgetting_factor": 0.99,
            "covariance_scale": 1_000.0,
            "jitter": 1e-8,
            "dtype": "float64",
            "weights": (0.0, 0.0, 0.0),
            "bias": 0.0,
            "covariance": ((1.0, 0.0, 0.0, 0.0),) * 4,
            "samples_seen": 0,
        },
    )

    with pytest.raises(ValueError, match="snapshot feature_dim must match 2; got 3"):
        readout.restore(snapshot)


def test_rls_restore_rejects_incompatible_config() -> None:
    readout = RLSReadout(feature_dim=2, forgetting_factor=0.95, jitter=1e-7)
    snapshot = readout.snapshot()

    bad_forgetting = replace(snapshot, state=_state(snapshot, forgetting_factor=0.99))
    with pytest.raises(ValueError, match="snapshot forgetting_factor must match"):
        readout.restore(bad_forgetting)

    bad_scale = replace(snapshot, state=_state(snapshot, covariance_scale=2_000.0))
    with pytest.raises(ValueError, match="snapshot covariance_scale must match"):
        readout.restore(bad_scale)

    bad_jitter = replace(snapshot, state=_state(snapshot, jitter=1e-8))
    with pytest.raises(ValueError, match="snapshot jitter must match current readout"):
        readout.restore(bad_jitter)

    bad_dtype = replace(snapshot, state=_state(snapshot, dtype="float32"))
    with pytest.raises(ValueError, match="snapshot dtype must match 'float64'"):
        readout.restore(bad_dtype)


def test_rls_restore_rejects_bad_weights_shape() -> None:
    readout = RLSReadout(feature_dim=2)
    snapshot = readout.snapshot()
    bad_snapshot = replace(snapshot, state=_state(snapshot, weights=(0.0,)))

    with pytest.raises(ValueError, match="expected feature_dim=2, got 1"):
        readout.restore(bad_snapshot)


def test_rls_restore_rejects_bad_covariance_shape() -> None:
    readout = RLSReadout(feature_dim=2)
    snapshot = readout.snapshot()
    bad_snapshot = replace(snapshot, state=_state(snapshot, covariance=((1.0,),)))

    with pytest.raises(ValueError, match="covariance must have shape"):
        readout.restore(bad_snapshot)


def test_rls_restore_rejects_non_finite_covariance() -> None:
    readout = RLSReadout(feature_dim=1)
    snapshot = readout.snapshot()
    bad_snapshot = replace(
        snapshot,
        state=_state(
            snapshot,
            covariance=((1.0, 0.0), (0.0, float("nan"))),
        ),
    )

    with pytest.raises(ValueError, match="covariance must contain only finite values"):
        readout.restore(bad_snapshot)


def test_rls_restore_rejects_negative_samples_seen() -> None:
    readout = RLSReadout(feature_dim=1)
    snapshot = readout.snapshot()
    bad_snapshot = replace(snapshot, state=_state(snapshot, samples_seen=-1))

    with pytest.raises(ValueError, match="snapshot samples_seen must be non-negative"):
        readout.restore(bad_snapshot)
