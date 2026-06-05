import pytest

import adaptive_reservoir
from adaptive_reservoir import (
    AdaptiveReservoir,
    ReservoirConfig,
    ReservoirCore,
    ReservoirState,
    extract_features,
)


def test_package_imports() -> None:
    assert adaptive_reservoir.__version__ == "0.0.0"
    assert ReservoirState.zeros(n_cells=1).samples_seen == 0
    core = ReservoirCore.from_config(ReservoirConfig(input_dim=1, n_cells=4))
    assert core.state.samples_seen == 0
    assert len(extract_features(core.state, "state_raw")) == 4


def test_public_api_processes_one_reservoir_step() -> None:
    model = AdaptiveReservoir(ReservoirConfig(input_dim=2, seed=42))

    result = model.step([0.1, -0.2], target=1.0)

    assert result.prediction is None
    assert len(result.features) == 2 * model.config.n_cells
    assert result.channels.novelty == 0.0
    assert result.channels.stability == 1.0
    assert result.channels.drift_pressure == 0.0
    assert result.channels.confidence == 0.0
    assert result.channels.saturation == 0.0
    assert result.metrics.samples_seen == 1
    assert result.metrics.target_available is True
    assert isinstance(result.state, ReservoirState)
    assert model.samples_seen == 1


def test_public_api_validates_input_dim() -> None:
    model = AdaptiveReservoir(ReservoirConfig(input_dim=2))

    with pytest.raises(ValueError, match="expected input_dim=2"):
        model.step([0.1])
