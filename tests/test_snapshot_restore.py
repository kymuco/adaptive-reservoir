from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from adaptive_reservoir import (
    AdaptiveChannels,
    AdaptiveReservoir,
    AdaptiveReservoirMetricsSnapshot,
    ChannelCalculatorSnapshot,
    ReadoutConfig,
    ReservoirConfig,
    ReservoirSnapshot,
    ReservoirState,
)
from adaptive_reservoir.readout import ReadoutSnapshot


def test_snapshot_captures_numeric_runtime_state() -> None:
    model = AdaptiveReservoir(_config())

    result = model.step([0.5, -0.25])
    snapshot = model.snapshot()

    assert snapshot.schema_version == 4
    assert snapshot.state.samples_seen == result.metrics.samples_seen
    assert isinstance(snapshot.readout, ReadoutSnapshot)
    assert isinstance(snapshot.channels, ChannelCalculatorSnapshot)
    assert isinstance(snapshot.metrics, AdaptiveReservoirMetricsSnapshot)
    assert snapshot.channels.samples_seen == result.metrics.samples_seen
    assert snapshot.metrics.samples_seen == result.metrics.samples_seen
    np.testing.assert_allclose(snapshot.state.activations, result.state.activations)
    np.testing.assert_allclose(snapshot.state.fast_trace, result.state.fast_trace)
    np.testing.assert_allclose(snapshot.state.mid_trace, result.state.mid_trace)
    np.testing.assert_allclose(snapshot.state.slow_trace, result.state.slow_trace)


def test_snapshot_is_independent_from_future_model_steps() -> None:
    model = AdaptiveReservoir(_config())

    model.step([0.5, -0.25], target=1.0)
    snapshot = model.snapshot()
    captured_state = snapshot.state
    captured_readout = snapshot.readout
    captured_channels = snapshot.channels
    captured_metrics = snapshot.metrics

    model.step([0.25, 0.75], target=-1.0)
    model.step([-1.0, 0.5], target=0.25)

    np.testing.assert_allclose(snapshot.state.activations, captured_state.activations)
    np.testing.assert_allclose(snapshot.state.fast_trace, captured_state.fast_trace)
    np.testing.assert_allclose(snapshot.state.mid_trace, captured_state.mid_trace)
    np.testing.assert_allclose(snapshot.state.slow_trace, captured_state.slow_trace)
    assert snapshot.state.samples_seen == captured_state.samples_seen
    assert snapshot.readout == captured_readout
    assert snapshot.channels == captured_channels
    assert snapshot.metrics == captured_metrics


def test_restore_rewinds_model_state_and_continuation_is_deterministic() -> None:
    model = AdaptiveReservoir(_config())

    model.step([0.5, -0.25])
    snapshot = model.snapshot()

    expected = model.step([0.25, 0.75])

    model.step([-1.0, 0.5])
    model.step([0.0, 1.0])

    model.restore(snapshot)
    actual = model.step([0.25, 0.75])

    assert actual.metrics.samples_seen == expected.metrics.samples_seen
    assert actual.prediction == pytest.approx(expected.prediction)
    _assert_channels_close(actual.channels, expected.channels)
    np.testing.assert_allclose(actual.state.activations, expected.state.activations)
    np.testing.assert_allclose(actual.state.fast_trace, expected.state.fast_trace)
    np.testing.assert_allclose(actual.state.mid_trace, expected.state.mid_trace)
    np.testing.assert_allclose(actual.state.slow_trace, expected.state.slow_trace)


def test_restore_recovers_readout_prediction() -> None:
    model = AdaptiveReservoir(_nlms_config())

    model.step([0.5, -0.25], target=1.0)
    snapshot = model.snapshot()
    expected = model.step([0.5, -0.25]).prediction

    model.step([0.25, 0.75], target=-2.0)
    model.step([-1.0, 0.5], target=3.0)

    model.restore(snapshot)
    actual = model.step([0.5, -0.25]).prediction

    assert actual == pytest.approx(expected)


def test_reset_matches_fresh_model_behavior() -> None:
    config = _config()
    model = AdaptiveReservoir(config)
    fresh = AdaptiveReservoir(config)

    model.step([0.5, -0.25], target=1.0)
    model.step([0.25, 0.75], target=-1.0)
    model.reset()

    reset_result = model.step([0.5, -0.25])
    fresh_result = fresh.step([0.5, -0.25])

    assert reset_result.metrics.samples_seen == fresh_result.metrics.samples_seen
    assert reset_result.prediction == pytest.approx(fresh_result.prediction)
    _assert_channels_close(reset_result.channels, fresh_result.channels)
    np.testing.assert_allclose(reset_result.state.activations, fresh_result.state.activations)
    np.testing.assert_allclose(reset_result.state.fast_trace, fresh_result.state.fast_trace)
    np.testing.assert_allclose(reset_result.state.mid_trace, fresh_result.state.mid_trace)
    np.testing.assert_allclose(reset_result.state.slow_trace, fresh_result.state.slow_trace)


def test_restore_rejects_wrong_snapshot_type() -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises(TypeError, match="ReservoirSnapshot"):
        model.restore({"schema_version": 4, "state": None})  # type: ignore[arg-type]


def test_restore_rejects_bad_state_shape() -> None:
    model = AdaptiveReservoir(_config())
    snapshot = ReservoirSnapshot(
        state=ReservoirState(
            activations=np.zeros(3, dtype=np.float64),
            fast_trace=np.zeros(3, dtype=np.float64),
            mid_trace=np.zeros(3, dtype=np.float64),
            slow_trace=np.zeros(3, dtype=np.float64),
        ),
        readout=_readout_snapshot(),
        channels=_channel_snapshot(),
        metrics=_metrics_snapshot(),
    )

    with pytest.raises(ValueError, match="shape"):
        model.restore(snapshot)


def test_restore_rejects_bad_state_dtype() -> None:
    model = AdaptiveReservoir(_config())
    snapshot = ReservoirSnapshot(
        state=ReservoirState(
            activations=np.zeros(4, dtype=np.float32),
            fast_trace=np.zeros(4, dtype=np.float32),
            mid_trace=np.zeros(4, dtype=np.float32),
            slow_trace=np.zeros(4, dtype=np.float32),
        ),
        readout=_readout_snapshot(),
        channels=_channel_snapshot(),
        metrics=_metrics_snapshot(),
    )

    with pytest.raises(ValueError, match="dtype"):
        model.restore(snapshot)


def test_restore_rejects_bad_schema_version() -> None:
    model = AdaptiveReservoir(_config())
    snapshot = ReservoirSnapshot(
        state=ReservoirState(
            activations=np.zeros(4, dtype=np.float64),
            fast_trace=np.zeros(4, dtype=np.float64),
            mid_trace=np.zeros(4, dtype=np.float64),
            slow_trace=np.zeros(4, dtype=np.float64),
        ),
        readout=_readout_snapshot(),
        channels=_channel_snapshot(),
        metrics=_metrics_snapshot(),
        schema_version=999,
    )

    with pytest.raises(ValueError, match="schema_version"):
        model.restore(snapshot)


def test_restore_rejects_bad_readout_snapshot() -> None:
    model = AdaptiveReservoir(_config())
    snapshot = dataclasses.replace(_valid_snapshot(), readout=_bad_readout_snapshot())

    with pytest.raises(ValueError, match="snapshot name must be"):
        model.restore(snapshot)


def test_restore_rejects_bad_channel_snapshot() -> None:
    model = AdaptiveReservoir(_config())
    snapshot = dataclasses.replace(
        _valid_snapshot(),
        channels=dataclasses.replace(_channel_snapshot(), schema_version=999),
    )

    with pytest.raises(ValueError, match="schema_version"):
        model.restore(snapshot)


def test_restore_rejects_bad_metrics_snapshot() -> None:
    model = AdaptiveReservoir(_config())
    snapshot = dataclasses.replace(
        _valid_snapshot(),
        metrics="bad",  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="snapshot.metrics"):
        model.restore(snapshot)


def test_restore_is_atomic_when_readout_restore_fails() -> None:
    model = AdaptiveReservoir(_nlms_config())
    model.step([0.5, -0.25], target=1.0)
    current = model.snapshot()
    bad_snapshot = dataclasses.replace(
        current,
        readout=dataclasses.replace(current.readout, name="other"),
    )

    with pytest.raises(ValueError, match="snapshot name must be"):
        model.restore(bad_snapshot)

    after = model.snapshot()
    _assert_snapshots_equal(after, current)


def test_restore_is_atomic_when_channel_restore_fails() -> None:
    model = AdaptiveReservoir(_config())
    model.step([0.5, -0.25], target=1.0)
    current = model.snapshot()
    bad_snapshot = dataclasses.replace(
        current,
        channels=dataclasses.replace(current.channels, schema_version=999),
    )

    with pytest.raises(ValueError, match="schema_version"):
        model.restore(bad_snapshot)

    after = model.snapshot()
    _assert_snapshots_equal(after, current)


def test_restore_is_atomic_when_metrics_restore_fails() -> None:
    model = AdaptiveReservoir(_config())
    model.step([0.5, -0.25], target=1.0)
    current = model.snapshot()
    bad_snapshot = dataclasses.replace(
        current,
        metrics=dataclasses.replace(current.metrics, samples_seen=0),
    )

    with pytest.raises(ValueError, match="metrics samples_seen"):
        model.restore(bad_snapshot)

    after = model.snapshot()
    _assert_snapshots_equal(after, current)


def test_snapshot_contains_no_semantic_or_domain_fields() -> None:
    model = AdaptiveReservoir(_config())
    snapshot = model.snapshot()

    assert {field.name for field in dataclasses.fields(snapshot)} == {
        "channels",
        "metrics",
        "readout",
        "schema_version",
        "state",
    }
    forbidden_names = {
        "raw_inputs",
        "targets",
        "input_history",
        "target_history",
        "user_data",
        "semantic_labels",
        "domain_events",
        "conversation_data",
        "policy_decisions",
        "host_system_metadata",
        "hde_data",
        "config",
    }
    for name in forbidden_names:
        assert not hasattr(snapshot, name)
        assert not hasattr(snapshot.channels, name)
        assert not hasattr(snapshot.metrics, name)


def _config() -> ReservoirConfig:
    return ReservoirConfig(
        input_dim=2,
        n_cells=4,
        topology="ring_shortcuts",
        seed=42,
        feature_mode="state_slow_raw",
    )


def _nlms_config() -> ReservoirConfig:
    return ReservoirConfig(
        input_dim=2,
        n_cells=4,
        topology="ring_shortcuts",
        seed=42,
        feature_mode="state_slow_raw",
        readout=ReadoutConfig(name="nlms", learning_rate=0.5),
    )


def _assert_channels_close(actual: AdaptiveChannels, expected: AdaptiveChannels) -> None:
    assert actual.novelty == pytest.approx(expected.novelty)
    assert actual.stability == pytest.approx(expected.stability)
    assert actual.drift_pressure == pytest.approx(expected.drift_pressure)
    assert actual.confidence == pytest.approx(expected.confidence)
    assert actual.saturation == pytest.approx(expected.saturation)


def _assert_snapshots_equal(actual: ReservoirSnapshot, expected: ReservoirSnapshot) -> None:
    np.testing.assert_allclose(actual.state.activations, expected.state.activations)
    np.testing.assert_allclose(actual.state.fast_trace, expected.state.fast_trace)
    np.testing.assert_allclose(actual.state.mid_trace, expected.state.mid_trace)
    np.testing.assert_allclose(actual.state.slow_trace, expected.state.slow_trace)
    assert actual.state.samples_seen == expected.state.samples_seen
    assert actual.readout == expected.readout
    assert actual.channels == expected.channels
    assert actual.metrics == expected.metrics


def _valid_snapshot() -> ReservoirSnapshot:
    return AdaptiveReservoir(_config()).snapshot()


def _readout_snapshot() -> ReadoutSnapshot:
    return _valid_snapshot().readout


def _channel_snapshot() -> ChannelCalculatorSnapshot:
    return _valid_snapshot().channels


def _metrics_snapshot() -> AdaptiveReservoirMetricsSnapshot:
    return _valid_snapshot().metrics


def _bad_readout_snapshot() -> ReadoutSnapshot:
    return dataclasses.replace(_readout_snapshot(), name="other")
