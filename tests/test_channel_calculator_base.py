from __future__ import annotations

import math

import numpy as np
import pytest

from adaptive_reservoir import ChannelConfig, ReservoirState
from adaptive_reservoir.channels import AdaptiveChannelCalculator
from adaptive_reservoir.core.result import AdaptiveChannels


def test_channel_calculator_returns_default_finite_channels() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[1.0, 2.0],
        state=ReservoirState.zeros(n_cells=2),
        features=[0.1, 0.2],
        prediction=0.0,
        target=None,
    )

    assert isinstance(channels, AdaptiveChannels)
    assert channels.novelty == 0.0
    assert channels.stability == 1.0
    assert channels.drift_pressure == 0.0
    assert channels.confidence == 0.0
    assert channels.saturation == 0.0
    assert _channels_are_finite_and_bounded(channels)


def test_channel_calculator_target_defaults_to_none() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[1.0, 2.0],
        state=ReservoirState.zeros(n_cells=2),
        features=[0.1, 0.2],
        prediction=0.0,
    )

    assert _channels_are_finite_and_bounded(channels)
    assert calculator.samples_seen == 1
    assert calculator.prediction_count == 1
    assert calculator.prediction_error_count == 0


def test_channel_calculator_allows_missing_prediction() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[1.0, 2.0],
        state=ReservoirState.zeros(n_cells=2),
        features=[0.1, 0.2],
        prediction=None,
        target=1.0,
    )

    assert _channels_are_finite_and_bounded(channels)
    assert calculator.samples_seen == 1
    assert calculator.feature_count == 1
    assert calculator.state_delta_count == 1
    assert calculator.prediction_count == 0
    assert calculator.prediction_error_count == 0


def test_channel_calculator_prediction_defaults_to_none() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[1.0, 2.0],
        state=ReservoirState.zeros(n_cells=2),
        features=[0.1, 0.2],
    )

    assert _channels_are_finite_and_bounded(channels)
    assert calculator.samples_seen == 1
    assert calculator.feature_count == 1
    assert calculator.prediction_count == 0
    assert calculator.prediction_error_count == 0


def test_channel_calculator_increments_samples_seen() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())
    state = ReservoirState.zeros(n_cells=2)

    first = calculator.update(
        input=[1.0, 2.0],
        state=state,
        features=[0.1, 0.2],
        prediction=0.0,
        target=None,
    )
    second = calculator.update(
        input=[2.0, 3.0],
        state=_state_with_activations([0.1, -0.1]),
        features=[0.2, 0.3],
        prediction=0.5,
        target=1.0,
    )

    assert _channels_are_finite_and_bounded(first)
    assert _channels_are_finite_and_bounded(second)
    assert calculator.samples_seen == 2
    assert calculator.feature_dim == 2


def test_channel_calculator_tracks_bounded_feature_history() -> None:
    calculator = AdaptiveChannelCalculator(
        config=ChannelConfig(novelty_window=2, stability_window=4, drift_window=4)
    )

    for index in range(5):
        calculator.update(
            input=[float(index)],
            state=_state_with_activations([float(index), 0.0]),
            features=[float(index), float(index + 1)],
            prediction=float(index),
            target=None,
        )

    assert calculator.samples_seen == 5
    assert calculator.feature_count == 2


def test_channel_calculator_tracks_bounded_state_delta_history() -> None:
    calculator = AdaptiveChannelCalculator(
        config=ChannelConfig(novelty_window=4, stability_window=3, drift_window=4)
    )

    for index in range(6):
        calculator.update(
            input=[float(index)],
            state=_state_with_activations([float(index), float(index)]),
            features=[float(index)],
            prediction=float(index),
            target=None,
        )

    assert calculator.state_delta_count == 3


def test_channel_calculator_tracks_bounded_prediction_history() -> None:
    calculator = AdaptiveChannelCalculator(
        config=ChannelConfig(novelty_window=4, stability_window=3, drift_window=4)
    )

    for index in range(6):
        calculator.update(
            input=[float(index)],
            state=_state_with_activations([0.0, float(index)]),
            features=[float(index)],
            prediction=float(index),
            target=None,
        )

    assert calculator.prediction_count == 3


def test_channel_calculator_tracks_prediction_error_only_when_target_available() -> None:
    calculator = AdaptiveChannelCalculator(
        config=ChannelConfig(novelty_window=4, stability_window=4, drift_window=2)
    )

    calculator.update(
        input=[0.0],
        state=ReservoirState.zeros(n_cells=2),
        features=[0.0],
        prediction=0.0,
        target=None,
    )
    for index in range(4):
        calculator.update(
            input=[float(index)],
            state=_state_with_activations([float(index), 0.0]),
            features=[float(index)],
            prediction=float(index),
            target=float(index + 1),
        )

    assert calculator.prediction_error_count == 2


def test_channel_calculator_rejects_wrong_config_type() -> None:
    with pytest.raises(TypeError, match="config must be a ChannelConfig"):
        AdaptiveChannelCalculator(config=object())  # type: ignore[arg-type]


def test_channel_calculator_rejects_wrong_state_type() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    with pytest.raises(TypeError, match="state must be a ReservoirState"):
        calculator.update(
            input=[1.0],
            state=object(),  # type: ignore[arg-type]
            features=[0.1],
            prediction=0.0,
            target=None,
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_channel_calculator_rejects_non_finite_features(bad_value: float) -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    with pytest.raises(ValueError, match="features must contain only finite values"):
        calculator.update(
            input=[1.0],
            state=ReservoirState.zeros(n_cells=2),
            features=[bad_value, 0.0],
            prediction=0.0,
            target=None,
        )


def test_channel_calculator_rejects_feature_dim_change() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())
    calculator.update(
        input=[1.0],
        state=ReservoirState.zeros(n_cells=2),
        features=[0.1, 0.2],
        prediction=0.0,
        target=None,
    )

    with pytest.raises(ValueError, match="expected feature_dim=2, got 1"):
        calculator.update(
            input=[1.0],
            state=ReservoirState.zeros(n_cells=2),
            features=[0.1],
            prediction=0.0,
            target=None,
        )


@pytest.mark.parametrize("bad_prediction", [float("nan"), float("inf"), -float("inf")])
def test_channel_calculator_rejects_non_finite_prediction(bad_prediction: float) -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    with pytest.raises(ValueError, match="prediction must be finite"):
        calculator.update(
            input=[1.0],
            state=ReservoirState.zeros(n_cells=2),
            features=[0.1, 0.2],
            prediction=bad_prediction,
            target=None,
        )


@pytest.mark.parametrize("bad_prediction", ["bad", True])
def test_channel_calculator_rejects_non_numeric_prediction(bad_prediction: object) -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    with pytest.raises(ValueError, match="prediction must be numeric"):
        calculator.update(
            input=[1.0],
            state=ReservoirState.zeros(n_cells=2),
            features=[0.1, 0.2],
            prediction=bad_prediction,
            target=None,
        )


@pytest.mark.parametrize("bad_target", [float("nan"), float("inf"), -float("inf")])
def test_channel_calculator_rejects_non_finite_target(bad_target: float) -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    with pytest.raises(ValueError, match="target must be finite"):
        calculator.update(
            input=[1.0],
            state=ReservoirState.zeros(n_cells=2),
            features=[0.1, 0.2],
            prediction=0.0,
            target=bad_target,
        )


def test_channel_calculator_rejects_invalid_dtype() -> None:
    with pytest.raises(ValueError, match="dtype must be one of: float32, float64"):
        AdaptiveChannelCalculator(config=ChannelConfig(), dtype="int64")


def test_channel_calculator_reset_clears_runtime_history() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())
    calculator.update(
        input=[1.0],
        state=ReservoirState.zeros(n_cells=2),
        features=[0.1, 0.2],
        prediction=0.0,
        target=1.0,
    )

    calculator.reset()

    assert calculator.samples_seen == 0
    assert calculator.feature_dim is None
    assert calculator.feature_count == 0
    assert calculator.state_delta_count == 0
    assert calculator.prediction_count == 0
    assert calculator.prediction_error_count == 0
    channels = calculator.update(
        input=[1.0],
        state=ReservoirState.zeros(n_cells=2),
        features=[0.3],
        prediction=0.0,
        target=None,
    )
    assert channels.stability == 1.0
    assert calculator.feature_dim == 1


def test_channel_calculator_does_not_store_raw_input() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())
    raw_input = [123.0, 456.0]

    calculator.update(
        input=raw_input,
        state=ReservoirState.zeros(n_cells=2),
        features=[0.1, 0.2],
        prediction=0.0,
        target=None,
    )

    assert not hasattr(calculator, "_input_window")
    assert not hasattr(calculator, "_raw_inputs")
    assert not hasattr(calculator, "_input_history")


def _state_with_activations(values: list[float]) -> ReservoirState:
    activations = np.asarray(values, dtype=np.float64)
    return ReservoirState(
        activations=activations,
        fast_trace=np.zeros_like(activations),
        mid_trace=np.zeros_like(activations),
        slow_trace=np.zeros_like(activations),
    )


def _channels_are_finite_and_bounded(channels: AdaptiveChannels) -> bool:
    values = (
        channels.novelty,
        channels.stability,
        channels.drift_pressure,
        channels.confidence,
        channels.saturation,
    )
    return all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
