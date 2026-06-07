import pytest

import adaptive_reservoir
from adaptive_reservoir import (
    AdaptiveReservoir,
    ReservoirConfig,
    ReservoirState,
    StateDiagnostics,
    TraceNorms,
    calculate_state_diagnostics,
    extract_features,
)


def test_package_imports() -> None:
    assert adaptive_reservoir.__version__ == "0.0.0"
    assert ReservoirState.zeros(n_cells=1).samples_seen == 0
    diagnostics = calculate_state_diagnostics(
        previous=ReservoirState.zeros(n_cells=1),
        current=ReservoirState.zeros(n_cells=1),
        saturation_threshold=0.95,
    )
    assert isinstance(diagnostics, StateDiagnostics)
    assert isinstance(diagnostics.trace_norms, TraceNorms)


def test_public_api_processes_one_reservoir_step() -> None:
    model = AdaptiveReservoir(ReservoirConfig(input_dim=2, seed=42))

    result = model.step([0.1, -0.2], target=1.0)

    assert result.prediction is None
    assert len(result.features) == 2 * model.config.n_cells
    assert result.channels.novelty == 0.0
    assert result.channels.stability == 1.0
    assert result.channels.drift_pressure == 0.0
    assert result.channels.confidence == 0.0
    assert result.channels.saturation == result.metrics.saturation_rate
    assert result.metrics.samples_seen == 1
    assert result.metrics.target_available is True
    assert result.metrics.state_norm is not None
    assert result.metrics.state_delta is not None
    assert result.metrics.feature_norm is not None
    assert result.metrics.saturation_rate is not None
    assert isinstance(result.metrics.trace_norms, TraceNorms)
    assert isinstance(result.state, ReservoirState)
    assert model.samples_seen == 1


def test_public_api_validates_input_dim() -> None:
    model = AdaptiveReservoir(ReservoirConfig(input_dim=2))

    with pytest.raises(ValueError, match="expected input_dim=2"):
        model.step([0.1])


def test_public_api_feature_norm_tracks_selected_feature_mode() -> None:
    model = AdaptiveReservoir(
        ReservoirConfig(input_dim=2, n_cells=4, feature_mode="state_raw", seed=42)
    )

    result = model.step([0.1, -0.2])

    assert result.state is not None
    expected_features = extract_features(result.state, "state_raw")
    assert result.features == tuple(float(value) for value in expected_features)
    assert result.metrics.feature_norm == pytest.approx(
        float((sum(value * value for value in result.features) / len(result.features)) ** 0.5)
    )


def test_public_api_saturation_channel_uses_saturation_rate() -> None:
    model = AdaptiveReservoir(
        ReservoirConfig(input_dim=1, n_cells=4, input_scale=100.0, leak_rate=1.0, seed=1)
    )

    result = model.step([1.0])

    assert result.channels.saturation == result.metrics.saturation_rate
    assert result.metrics.saturation_rate is not None
    assert 0.0 <= result.metrics.saturation_rate <= 1.0


def test_public_api_reset_makes_next_delta_from_zero_state() -> None:
    model = AdaptiveReservoir(ReservoirConfig(input_dim=2, seed=42))

    first = model.step([0.1, -0.2])
    model.step([0.3, 0.4])
    model.reset()
    after_reset = model.step([0.1, -0.2])

    assert after_reset.metrics.state_delta == pytest.approx(first.metrics.state_delta)
