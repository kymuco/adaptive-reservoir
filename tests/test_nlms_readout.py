from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from adaptive_reservoir.readout import NLMSReadout, ReadoutSnapshot


def test_nlms_predict_starts_from_zero() -> None:
    readout = NLMSReadout(feature_dim=2)

    assert readout.predict([1.0, -2.0]) == 0.0
    assert readout.samples_seen == 0
    assert readout.bias == 0.0
    np.testing.assert_allclose(readout.weights, np.zeros(2))


def test_nlms_update_changes_prediction_toward_target() -> None:
    readout = NLMSReadout(feature_dim=2, learning_rate=0.5)
    features = [1.0, 0.0]
    target = 1.0

    before = readout.predict(features)
    readout.update(features, target)
    after = readout.predict(features)

    assert abs(target - after) < abs(target - before)


def test_nlms_update_increments_samples_seen() -> None:
    readout = NLMSReadout(feature_dim=2)

    readout.update([1.0, 0.0], 1.0)
    readout.update([0.0, 1.0], -1.0)

    assert readout.samples_seen == 2


def test_nlms_update_uses_normalized_lms_rule() -> None:
    readout = NLMSReadout(feature_dim=2, learning_rate=0.5, epsilon=1e-8)

    readout.update([2.0, 0.0], 1.0)

    expected_weight = 0.5 * 1.0 * 2.0 / (1e-8 + 4.0)
    np.testing.assert_allclose(readout.weights, np.array([expected_weight, 0.0]))
    assert readout.bias == pytest.approx(0.5)


def test_nlms_rejects_invalid_constructor_values() -> None:
    with pytest.raises(ValueError, match="feature_dim must be a positive integer"):
        NLMSReadout(feature_dim=0)
    with pytest.raises(ValueError, match="learning_rate must be finite and positive"):
        NLMSReadout(feature_dim=1, learning_rate=0.0)
    with pytest.raises(ValueError, match="epsilon must be finite and positive"):
        NLMSReadout(feature_dim=1, epsilon=float("nan"))


def test_nlms_rejects_non_floating_dtype() -> None:
    with pytest.raises(ValueError, match="dtype must be a floating dtype"):
        NLMSReadout(feature_dim=1, dtype="int64")


def test_nlms_rejects_wrong_feature_dim() -> None:
    readout = NLMSReadout(feature_dim=2)

    with pytest.raises(ValueError, match="expected feature_dim=2, got 1"):
        readout.predict([1.0])

    with pytest.raises(ValueError, match="expected feature_dim=2, got 1"):
        readout.update([1.0], 1.0)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nlms_rejects_non_finite_features(bad_value: float) -> None:
    readout = NLMSReadout(feature_dim=2)

    with pytest.raises(ValueError, match="features must contain only finite values"):
        readout.predict([bad_value, 1.0])


@pytest.mark.parametrize("bad_target", [float("nan"), float("inf"), -float("inf")])
def test_nlms_rejects_non_finite_target(bad_target: float) -> None:
    readout = NLMSReadout(feature_dim=2)

    with pytest.raises(ValueError, match="target must be finite"):
        readout.update([1.0, 0.0], bad_target)


def test_nlms_weights_property_is_read_only_copy() -> None:
    readout = NLMSReadout(feature_dim=2)
    readout.update([1.0, 0.0], 1.0)

    weights = readout.weights

    assert weights.flags.writeable is False
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        weights[0] = 100.0
    assert readout.predict([1.0, 0.0]) != 100.0


def test_nlms_snapshot_contains_numeric_state_only() -> None:
    readout = NLMSReadout(feature_dim=2, learning_rate=0.25, epsilon=1e-6)
    readout.update([1.0, -1.0], 0.5)

    snapshot = readout.snapshot()

    assert snapshot.name == "nlms"
    assert snapshot.schema_version == 1
    assert set(snapshot.state) == {
        "bias",
        "dtype",
        "epsilon",
        "feature_dim",
        "learning_rate",
        "samples_seen",
        "weights",
    }
    assert "raw_inputs" not in snapshot.state
    assert "target_history" not in snapshot.state
    assert "messages" not in snapshot.state
    assert "policy_decision" not in snapshot.state


def test_nlms_snapshot_state_is_read_only() -> None:
    readout = NLMSReadout(feature_dim=2)
    snapshot = readout.snapshot()

    with pytest.raises(TypeError):
        snapshot.state["bias"] = 1.0  # type: ignore[index]


def test_nlms_snapshot_is_independent_from_future_updates() -> None:
    readout = NLMSReadout(feature_dim=2, learning_rate=0.5)
    readout.update([1.0, 0.0], 1.0)
    snapshot = readout.snapshot()
    expected_weights = snapshot.state["weights"]
    expected_bias = snapshot.state["bias"]

    readout.update([0.0, 1.0], -1.0)

    assert snapshot.state["weights"] == expected_weights
    assert snapshot.state["bias"] == expected_bias


def test_nlms_restore_recovers_prediction() -> None:
    readout = NLMSReadout(feature_dim=2, learning_rate=0.5)
    readout.update([1.0, 0.0], 1.0)
    snapshot = readout.snapshot()
    expected = readout.predict([1.0, 0.0])

    readout.update([-1.0, 2.0], -0.5)
    readout.restore(snapshot)
    actual = readout.predict([1.0, 0.0])

    assert actual == pytest.approx(expected)
    assert readout.samples_seen == snapshot.state["samples_seen"]


def test_nlms_restore_rejects_wrong_snapshot_type() -> None:
    readout = NLMSReadout(feature_dim=2)

    with pytest.raises(TypeError, match="snapshot must be a ReadoutSnapshot"):
        readout.restore({})  # type: ignore[arg-type]


def test_nlms_restore_rejects_wrong_snapshot_name() -> None:
    readout = NLMSReadout(feature_dim=2)
    snapshot = replace(readout.snapshot(), name="other")

    with pytest.raises(ValueError, match="snapshot name must be 'nlms'"):
        readout.restore(snapshot)


def test_nlms_restore_rejects_incompatible_feature_dim() -> None:
    readout = NLMSReadout(feature_dim=2)
    snapshot = ReadoutSnapshot(
        schema_version=1,
        name="nlms",
        state={
            "feature_dim": 3,
            "dtype": "float64",
            "learning_rate": 0.1,
            "epsilon": 1e-8,
            "weights": (0.0, 0.0, 0.0),
            "bias": 0.0,
            "samples_seen": 0,
        },
    )

    with pytest.raises(ValueError, match="snapshot feature_dim must match 2; got 3"):
        readout.restore(snapshot)


def test_nlms_restore_rejects_incompatible_dtype() -> None:
    readout = NLMSReadout(feature_dim=2, dtype="float64")
    snapshot = readout.snapshot()
    bad_snapshot = replace(
        snapshot,
        state={**snapshot.state, "dtype": "float32"},
    )

    with pytest.raises(ValueError, match="snapshot dtype must match 'float64'"):
        readout.restore(bad_snapshot)


def test_nlms_restore_rejects_bad_weights_shape() -> None:
    readout = NLMSReadout(feature_dim=2)
    snapshot = readout.snapshot()
    bad_snapshot = replace(
        snapshot,
        state={**snapshot.state, "weights": (0.0,)},
    )

    with pytest.raises(ValueError, match="expected feature_dim=2, got 1"):
        readout.restore(bad_snapshot)


def test_nlms_restore_rejects_negative_samples_seen() -> None:
    readout = NLMSReadout(feature_dim=2)
    snapshot = readout.snapshot()
    bad_snapshot = replace(
        snapshot,
        state={**snapshot.state, "samples_seen": -1},
    )

    with pytest.raises(ValueError, match="snapshot samples_seen must be non-negative"):
        readout.restore(bad_snapshot)
