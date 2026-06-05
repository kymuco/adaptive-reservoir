import numpy as np
import pytest

from adaptive_reservoir import AdaptiveChannels, AdaptiveReservoir, ReservoirConfig, ReservoirState


def test_adaptive_reservoir_step_returns_stateful_result() -> None:
    model = AdaptiveReservoir(_config())

    result = model.step([0.5, -0.25])

    assert result.prediction is None
    assert isinstance(result.channels, AdaptiveChannels)
    assert isinstance(result.state, ReservoirState)
    assert result.metrics.samples_seen == 1
    assert model.samples_seen == 1


def test_adaptive_reservoir_default_features_are_state_slow_raw() -> None:
    model = AdaptiveReservoir(_config())

    result = model.step([0.5, -0.25])

    assert result.state is not None
    expected = np.concatenate((result.state.activations, result.state.slow_trace))
    assert len(result.features) == 8
    np.testing.assert_allclose(result.features, expected)


@pytest.mark.parametrize(
    ("feature_mode", "expected_length"),
    [
        ("state_raw", 4),
        ("state_slow_raw", 8),
        ("multi_raw", 16),
    ],
)
def test_adaptive_reservoir_step_uses_configured_feature_mode(
    feature_mode: str,
    expected_length: int,
) -> None:
    model = AdaptiveReservoir(_config(feature_mode=feature_mode))

    result = model.step([0.5, -0.25])

    assert len(result.features) == expected_length


def test_adaptive_reservoir_step_increments_samples_seen() -> None:
    model = AdaptiveReservoir(_config())

    first = model.step([0.5, -0.25])
    second = model.step([0.25, 0.75])

    assert first.metrics.samples_seen == 1
    assert second.metrics.samples_seen == 2
    assert model.samples_seen == 2


def test_adaptive_reservoir_step_validates_input_dim() -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises(ValueError, match="expected input_dim=2, got 1"):
        model.step([1.0])


def test_adaptive_reservoir_step_rejects_non_finite_input() -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises(ValueError, match="all input values must be finite"):
        model.step([1.0, float("nan")])


def test_adaptive_reservoir_step_rejects_non_finite_target() -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises(ValueError, match="target must be finite"):
        model.step([1.0, 0.0], target=float("nan"))


def test_adaptive_reservoir_reset_reinitializes_core_state() -> None:
    model = AdaptiveReservoir(_config())
    model.step([0.5, -0.25])

    model.reset()

    assert model.samples_seen == 0
    result = model.step([0.0, 0.0])
    assert result.metrics.samples_seen == 1


def test_adaptive_reservoir_snapshot_remains_lightweight_in_pr33() -> None:
    model = AdaptiveReservoir(_config())
    model.step([0.5, -0.25])

    snapshot = model.snapshot()

    assert snapshot["samples_seen"] == 1
    assert snapshot["config"] == model.config
    assert snapshot["api_stage"] == "feature_modes_v1"


def _config(*, feature_mode: str = "state_slow_raw") -> ReservoirConfig:
    return ReservoirConfig(
        input_dim=2,
        n_cells=4,
        topology="ring_shortcuts",
        seed=42,
        feature_mode=feature_mode,  # type: ignore[arg-type]
    )
