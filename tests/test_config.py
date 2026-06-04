from dataclasses import FrozenInstanceError, replace

import pytest

from adaptive_reservoir import ChannelConfig, ReadoutConfig, ReservoirConfig, TraceConfig


def test_default_config_is_valid() -> None:
    config = ReservoirConfig(input_dim=8)

    assert config.input_dim == 8
    assert config.n_cells == 64
    assert config.topology == "modular_small_world"
    assert config.feature_mode == "state_slow_raw"
    assert config.seed == 0
    assert config.dtype == "float64"
    assert config.trace_decays == (0.5, 0.9, 0.99)
    assert config.readout.name == "sliding_ridge"
    assert config.channels.saturation_threshold == 0.95


def test_config_is_frozen() -> None:
    config = ReservoirConfig(input_dim=8)

    with pytest.raises(FrozenInstanceError):
        config.input_dim = 16  # type: ignore[misc]


def test_config_can_be_safely_copied() -> None:
    config = ReservoirConfig(input_dim=8)
    updated = replace(config, n_cells=128)

    assert config.n_cells == 64
    assert updated.n_cells == 128


@pytest.mark.parametrize("input_dim", [0, -1])
def test_invalid_input_dim_raises_clear_error(input_dim: int) -> None:
    with pytest.raises(ValueError, match="input_dim must be positive"):
        ReservoirConfig(input_dim=input_dim)


@pytest.mark.parametrize("n_cells", [0, -1])
def test_invalid_n_cells_raises_clear_error(n_cells: int) -> None:
    with pytest.raises(ValueError, match="n_cells must be positive"):
        ReservoirConfig(input_dim=8, n_cells=n_cells)


@pytest.mark.parametrize(
    ("field_name", "kwargs", "message"),
    [
        ("topology", {"topology": "bad"}, "topology must be one of"),
        ("feature_mode", {"feature_mode": "bad"}, "feature_mode must be one of"),
        ("dtype", {"dtype": "bad"}, "dtype must be one of"),
    ],
)
def test_invalid_reservoir_choices_raise_clear_errors(
    field_name: str,
    kwargs: dict[str, str],
    message: str,
) -> None:
    del field_name

    with pytest.raises(ValueError, match=message):
        ReservoirConfig(input_dim=8, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"fast_decay": -0.1}, "fast_decay must be in the range"),
        ({"mid_decay": 1.0}, "mid_decay must be in the range"),
        ({"slow_decay": 1.5}, "slow_decay must be in the range"),
    ],
)
def test_invalid_trace_config_raises_clear_errors(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TraceConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": "bad"}, "readout.name must be one of"),
        ({"learning_rate": 0.0}, "learning_rate must be positive"),
        ({"ridge_alpha": -0.1}, "ridge_alpha must be non-negative"),
        ({"buffer_size": 0}, "buffer_size must be positive"),
        ({"window_size": 0}, "window_size must be positive"),
        ({"update_interval": 0}, "update_interval must be positive"),
    ],
)
def test_invalid_readout_config_raises_clear_errors(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ReadoutConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"novelty_window": 0}, "novelty_window must be positive"),
        ({"stability_window": 0}, "stability_window must be positive"),
        ({"drift_window": 0}, "drift_window must be positive"),
        ({"saturation_threshold": 0.0}, "saturation_threshold must be in the range"),
        ({"saturation_threshold": 1.1}, "saturation_threshold must be in the range"),
        ({"epsilon": 0.0}, "epsilon must be positive"),
    ],
)
def test_invalid_channel_config_raises_clear_errors(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ChannelConfig(**kwargs)  # type: ignore[arg-type]
