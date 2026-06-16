from __future__ import annotations

import numpy as np
import pytest

from adaptive_reservoir import (
    AdaptiveReservoir,
    ReadoutConfig,
    ReservoirConfig,
    ReservoirSnapshot,
)


def test_predict_without_input_returns_current_state_prediction() -> None:
    model = AdaptiveReservoir(_config())

    prediction = model.predict()

    assert prediction == pytest.approx(0.0)
    assert model.samples_seen == 0


def test_predict_with_input_does_not_increment_samples_seen() -> None:
    model = AdaptiveReservoir(_config())

    prediction = model.predict([0.5, -0.25])

    assert prediction == pytest.approx(0.0)
    assert model.samples_seen == 0


def test_predict_without_input_does_not_mutate_snapshot() -> None:
    model = AdaptiveReservoir(_config())
    model.step([0.1, -0.1], target=1.0)

    before = model.snapshot()
    prediction = model.predict()
    after = model.snapshot()

    assert isinstance(prediction, float)
    _assert_snapshots_equal(after, before)


def test_predict_with_input_does_not_mutate_snapshot() -> None:
    model = AdaptiveReservoir(_config())
    model.step([0.1, -0.1], target=1.0)

    before = model.snapshot()
    prediction = model.predict([0.5, -0.25])
    after = model.snapshot()

    assert isinstance(prediction, float)
    _assert_snapshots_equal(after, before)


def test_predict_with_input_matches_step_prediction_after_restore() -> None:
    config = _config()
    model = AdaptiveReservoir(config)
    model.step([0.1, -0.1], target=1.0)
    snapshot = model.snapshot()

    predicted = model.predict([0.5, -0.25])

    copy = AdaptiveReservoir(config)
    copy.restore(snapshot)
    stepped = copy.step([0.5, -0.25])

    assert stepped.prediction == pytest.approx(predicted)


def test_predict_uses_current_readout_state() -> None:
    model = AdaptiveReservoir(
        _config(readout=ReadoutConfig(name="nlms", learning_rate=0.5))
    )

    before = model.predict([0.5, -0.25])
    model.step([0.5, -0.25], target=1.0)
    after = model.predict([0.5, -0.25])

    assert before == pytest.approx(0.0)
    assert after != pytest.approx(before)


def test_predict_rejects_bad_input_shape() -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises(ValueError, match="expected input_dim=2, got 1"):
        model.predict([1.0])


def test_predict_rejects_non_finite_input() -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises(ValueError, match="x must contain only finite values"):
        model.predict([1.0, float("nan")])


def _config(readout: ReadoutConfig | None = None) -> ReservoirConfig:
    return ReservoirConfig(
        input_dim=2,
        n_cells=4,
        topology="ring_shortcuts",
        seed=42,
        feature_mode="state_slow_raw",
        readout=readout or ReadoutConfig(name="sliding_ridge"),
    )


def _assert_snapshots_equal(actual: ReservoirSnapshot, expected: ReservoirSnapshot) -> None:
    assert actual.schema_version == expected.schema_version
    assert actual.state.samples_seen == expected.state.samples_seen
    np.testing.assert_allclose(actual.state.activations, expected.state.activations)
    np.testing.assert_allclose(actual.state.fast_trace, expected.state.fast_trace)
    np.testing.assert_allclose(actual.state.mid_trace, expected.state.mid_trace)
    np.testing.assert_allclose(actual.state.slow_trace, expected.state.slow_trace)
    assert actual.readout == expected.readout
    assert actual.channels == expected.channels
