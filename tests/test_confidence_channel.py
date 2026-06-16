from __future__ import annotations

import math

import numpy as np
import pytest

from adaptive_reservoir import ChannelConfig, ReservoirState
from adaptive_reservoir.channels import AdaptiveChannelCalculator
from adaptive_reservoir.core.result import AdaptiveChannels


def test_confidence_is_zero_when_prediction_missing() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=None,
    )

    assert channels.confidence == pytest.approx(0.0)
    assert _channels_are_finite_and_bounded(channels)


def test_confidence_is_zero_for_zero_prediction() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=0.0,
    )

    assert channels.confidence == pytest.approx(0.0)
    assert _channels_are_finite_and_bounded(channels)


def test_confidence_uses_positive_prediction_magnitude() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=0.25,
    )

    assert channels.confidence == pytest.approx(0.25)
    assert _channels_are_finite_and_bounded(channels)


def test_confidence_uses_negative_prediction_magnitude() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=-0.75,
    )

    assert channels.confidence == pytest.approx(0.75)
    assert _channels_are_finite_and_bounded(channels)


def test_confidence_is_clipped_to_one() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=10.0,
    )

    assert channels.confidence == pytest.approx(1.0)
    assert _channels_are_finite_and_bounded(channels)


def test_confidence_does_not_depend_on_target() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    without_target = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=0.4,
        target=None,
    )
    calculator.reset()
    with_target = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=0.4,
        target=100.0,
    )

    assert with_target.confidence == pytest.approx(without_target.confidence)
    assert _channels_are_finite_and_bounded(with_target)


def test_confidence_handles_large_prediction() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=1e308,
    )

    assert channels.confidence == pytest.approx(1.0)
    assert _channels_are_finite_and_bounded(channels)


def test_confidence_uses_current_prediction_not_history() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    first = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=1.0,
    )
    second = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=None,
    )

    assert first.confidence == pytest.approx(1.0)
    assert second.confidence == pytest.approx(0.0)
    assert calculator.prediction_count == 1
    assert _channels_are_finite_and_bounded(second)


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
