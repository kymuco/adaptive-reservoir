from __future__ import annotations

import math

import numpy as np
import pytest

from adaptive_reservoir import ChannelConfig, ReservoirState
from adaptive_reservoir.channels import AdaptiveChannelCalculator
from adaptive_reservoir.core.result import AdaptiveChannels


def test_saturation_is_zero_when_no_activation_exceeds_threshold() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.5, -0.95]),
        features=[0.0, 0.0, 0.0],
        prediction=None,
    )

    assert channels.saturation == pytest.approx(0.0)
    assert _channels_are_finite_and_bounded(channels)


def test_saturation_counts_fraction_above_threshold() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.96, -0.99, 0.5]),
        features=[0.0, 0.0, 0.0, 0.0],
        prediction=None,
    )

    assert channels.saturation == pytest.approx(0.5)
    assert _channels_are_finite_and_bounded(channels)


def test_saturation_is_one_when_all_activations_exceed_threshold() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[0.0],
        state=_state([0.96, -0.99, 1.0]),
        features=[0.0, 0.0, 0.0],
        prediction=None,
    )

    assert channels.saturation == pytest.approx(1.0)
    assert _channels_are_finite_and_bounded(channels)


def test_saturation_uses_strictly_greater_than_threshold() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[0.0],
        state=_state([0.95, -0.95, 0.9501]),
        features=[0.0, 0.0, 0.0],
        prediction=None,
    )

    assert channels.saturation == pytest.approx(1.0 / 3.0)
    assert _channels_are_finite_and_bounded(channels)


def test_saturation_uses_configured_threshold() -> None:
    calculator = AdaptiveChannelCalculator(
        config=ChannelConfig(saturation_threshold=0.5),
    )

    channels = calculator.update(
        input=[0.0],
        state=_state([0.4, 0.6, -0.7, 0.5]),
        features=[0.0, 0.0, 0.0, 0.0],
        prediction=None,
    )

    assert channels.saturation == pytest.approx(0.5)
    assert _channels_are_finite_and_bounded(channels)


def test_saturation_does_not_depend_on_prediction_or_target() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    without_prediction = calculator.update(
        input=[0.0],
        state=_state([0.96, 0.0]),
        features=[0.0, 0.0],
        prediction=None,
        target=None,
    )
    calculator.reset()
    with_prediction = calculator.update(
        input=[0.0],
        state=_state([0.96, 0.0]),
        features=[0.0, 0.0],
        prediction=100.0,
        target=-100.0,
    )

    assert with_prediction.saturation == pytest.approx(without_prediction.saturation)
    assert _channels_are_finite_and_bounded(with_prediction)


def test_saturation_uses_current_activations_not_history() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    first = calculator.update(
        input=[0.0],
        state=_state([0.96, -0.99]),
        features=[0.0, 0.0],
        prediction=None,
    )
    second = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=None,
    )

    assert first.saturation == pytest.approx(1.0)
    assert second.saturation == pytest.approx(0.0)
    assert _channels_are_finite_and_bounded(second)


def test_saturation_handles_large_finite_activations() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[0.0],
        state=_state([1e308, -1e308, 0.0]),
        features=[0.0, 0.0, 0.0],
        prediction=None,
    )

    assert channels.saturation == pytest.approx(2.0 / 3.0)
    assert _channels_are_finite_and_bounded(channels)


def _state(values: list[float], *, dtype: np.dtype | type = np.float64) -> ReservoirState:
    activations = np.asarray(values, dtype=dtype)
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
