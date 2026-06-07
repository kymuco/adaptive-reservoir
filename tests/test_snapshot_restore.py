import copy

import numpy as np
import pytest

from adaptive_reservoir import (
    AdaptiveReservoir,
    ReservoirConfig,
    ReservoirState,
    SNAPSHOT_API_STAGE,
    SNAPSHOT_SCHEMA_VERSION,
    restore_state,
    snapshot_state,
    validate_runtime_snapshot,
)


def test_snapshot_contains_schema_version_and_math_state_only() -> None:
    model = AdaptiveReservoir(_config())
    model.step([0.5, -0.25])

    snapshot = model.snapshot()

    assert snapshot["schema_version"] == SNAPSHOT_SCHEMA_VERSION
    assert snapshot["api_stage"] == SNAPSHOT_API_STAGE
    assert set(snapshot) == {
        "schema_version",
        "api_stage",
        "state",
        "readout_state",
        "metrics_buffers",
    }
    assert "config" not in snapshot
    assert "semantic_memory" not in snapshot
    assert "user_data" not in snapshot


def test_snapshot_serializes_state_as_plain_lists() -> None:
    model = AdaptiveReservoir(_config())
    result = model.step([0.5, -0.25])

    snapshot = model.snapshot()
    state = snapshot["state"]

    assert isinstance(state, dict)
    assert state["n_cells"] == model.config.n_cells
    assert state["dtype"] == model.config.dtype
    assert state["samples_seen"] == result.metrics.samples_seen
    assert isinstance(state["activations"], list)
    assert isinstance(state["fast_trace"], list)
    assert isinstance(state["mid_trace"], list)
    assert isinstance(state["slow_trace"], list)
    assert len(state["activations"]) == model.config.n_cells


def test_snapshot_state_roundtrip_restores_reservoir_state() -> None:
    model = AdaptiveReservoir(_config(dtype="float32"))
    result = model.step([0.5, -0.25])
    assert result.state is not None

    restored = restore_state(
        snapshot_state(result.state),
        expected_n_cells=model.config.n_cells,
        dtype=model.config.dtype,
    )

    assert restored.samples_seen == result.state.samples_seen
    assert restored.activations.dtype == np.float32
    np.testing.assert_allclose(restored.activations, result.state.activations)
    np.testing.assert_allclose(restored.fast_trace, result.state.fast_trace)
    np.testing.assert_allclose(restored.mid_trace, result.state.mid_trace)
    np.testing.assert_allclose(restored.slow_trace, result.state.slow_trace)


def test_restore_restores_samples_seen_and_traces() -> None:
    model = AdaptiveReservoir(_config())
    before = model.step([0.5, -0.25])
    assert before.state is not None
    snapshot = model.snapshot()

    model.step([0.25, 0.75])
    model.restore(snapshot)

    assert model.samples_seen == before.state.samples_seen
    after = model.step([0.0, 0.0])
    assert after.state is not None
    assert after.state.samples_seen == before.state.samples_seen + 1


def test_restore_allows_deterministic_continuation_after_restore() -> None:
    left = AdaptiveReservoir(_config())
    right = AdaptiveReservoir(_config())

    for sample in ([0.5, -0.25], [0.25, 0.75]):
        left.step(sample)
        right.step(sample)

    snapshot = left.snapshot()
    expected = left.step([0.1, -0.4])
    right.step([0.9, 0.9])
    right.restore(snapshot)
    actual = right.step([0.1, -0.4])

    assert expected.features == pytest.approx(actual.features)
    assert expected.metrics.samples_seen == actual.metrics.samples_seen
    assert expected.metrics.state_norm == pytest.approx(actual.metrics.state_norm)
    assert expected.metrics.state_delta == pytest.approx(actual.metrics.state_delta)
    assert expected.channels.saturation == pytest.approx(actual.channels.saturation)
    assert expected.state is not None
    assert actual.state is not None
    np.testing.assert_allclose(actual.state.activations, expected.state.activations)
    np.testing.assert_allclose(actual.state.fast_trace, expected.state.fast_trace)
    np.testing.assert_allclose(actual.state.mid_trace, expected.state.mid_trace)
    np.testing.assert_allclose(actual.state.slow_trace, expected.state.slow_trace)


def test_restore_rejects_missing_state() -> None:
    model = AdaptiveReservoir(_config())
    snapshot = model.snapshot()
    del snapshot["state"]

    with pytest.raises(ValueError, match="snapshot state is required"):
        model.restore(snapshot)


def test_restore_rejects_unsupported_schema_version() -> None:
    model = AdaptiveReservoir(_config())
    snapshot = model.snapshot()
    snapshot["schema_version"] = 999

    with pytest.raises(ValueError, match="unsupported snapshot schema_version"):
        model.restore(snapshot)


def test_restore_rejects_wrong_n_cells() -> None:
    model = AdaptiveReservoir(_config())
    snapshot = model.snapshot()
    assert isinstance(snapshot["state"], dict)
    snapshot["state"]["n_cells"] = 999

    with pytest.raises(ValueError, match="n_cells"):
        model.restore(snapshot)


def test_restore_rejects_wrong_dtype() -> None:
    model = AdaptiveReservoir(_config(dtype="float64"))
    snapshot = model.snapshot()
    assert isinstance(snapshot["state"], dict)
    snapshot["state"]["dtype"] = "float32"

    with pytest.raises(ValueError, match="dtype"):
        model.restore(snapshot)


def test_restore_rejects_non_finite_values() -> None:
    model = AdaptiveReservoir(_config())
    snapshot = model.snapshot()
    assert isinstance(snapshot["state"], dict)
    snapshot["state"]["activations"] = [0.0, float("nan"), 0.0, 0.0]

    with pytest.raises(ValueError, match="finite values"):
        model.restore(snapshot)


def test_restore_rejects_negative_samples_seen() -> None:
    model = AdaptiveReservoir(_config())
    snapshot = model.snapshot()
    assert isinstance(snapshot["state"], dict)
    snapshot["state"]["samples_seen"] = -1

    with pytest.raises(ValueError, match="samples_seen"):
        model.restore(snapshot)


def test_restore_rejects_non_none_readout_state_for_now() -> None:
    model = AdaptiveReservoir(_config())
    snapshot = model.snapshot()
    snapshot["readout_state"] = {"weights": [1.0]}

    with pytest.raises(ValueError, match="readout_state restore is not supported"):
        model.restore(snapshot)


def test_restore_rejects_non_empty_metrics_buffers_for_now() -> None:
    model = AdaptiveReservoir(_config())
    snapshot = model.snapshot()
    snapshot["metrics_buffers"] = {"state_norm": [0.1]}

    with pytest.raises(ValueError, match="metrics_buffers restore is not supported"):
        model.restore(snapshot)


def test_validate_runtime_snapshot_requires_metrics_buffers_mapping() -> None:
    model = AdaptiveReservoir(_config())
    snapshot = model.snapshot()
    snapshot["metrics_buffers"] = None

    with pytest.raises(ValueError, match="metrics_buffers must be a mapping"):
        validate_runtime_snapshot(snapshot)


def test_restore_does_not_alias_snapshot_lists() -> None:
    model = AdaptiveReservoir(_config())
    model.step([0.5, -0.25])
    snapshot = model.snapshot()
    copied = copy.deepcopy(snapshot)

    model.restore(snapshot)
    assert isinstance(snapshot["state"], dict)
    snapshot["state"]["activations"][0] = 999.0

    current = model.snapshot()
    assert current == copied


def test_reset_after_restore_returns_to_zero_state() -> None:
    model = AdaptiveReservoir(_config())
    model.step([0.5, -0.25])
    snapshot = model.snapshot()
    model.step([0.25, 0.75])
    model.restore(snapshot)

    model.reset()
    result = model.step([0.0, 0.0])

    assert result.state is not None
    assert result.metrics.samples_seen == 1
    np.testing.assert_array_equal(result.state.activations, np.zeros(model.config.n_cells))
    np.testing.assert_array_equal(result.state.fast_trace, np.zeros(model.config.n_cells))
    np.testing.assert_array_equal(result.state.mid_trace, np.zeros(model.config.n_cells))
    np.testing.assert_array_equal(result.state.slow_trace, np.zeros(model.config.n_cells))


def _config(*, dtype: str = "float64") -> ReservoirConfig:
    return ReservoirConfig(
        input_dim=2,
        n_cells=4,
        topology="ring_shortcuts",
        seed=42,
        dtype=dtype,  # type: ignore[arg-type]
    )
