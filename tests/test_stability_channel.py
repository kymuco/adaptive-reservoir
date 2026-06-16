from __future__ import annotations

import math

import numpy as np
import pytest

from adaptive_reservoir import ChannelConfig, ReservoirState
from adaptive_reservoir.channels import AdaptiveChannelCalculator
from adaptive_reservoir.core.result import AdaptiveChannels


def test_stability_starts_at_one_without_history() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=0.0,
    )

    assert channels.stability == pytest.approx(1.0)
    assert _channels_are_finite_and_bounded(channels)


def test_stability_stays_high_for_repeated_stream() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(stability_window=8))

    for _ in range(10):
        channels = calculator.update(
            input=[0.0],
            state=_state([0.1, -0.1]),
            features=[0.1, -0.1],
            prediction=0.5,
        )

    assert channels.stability > 0.95
    assert _channels_are_finite_and_bounded(channels)


def test_stability_drops_for_noisy_state_deltas() -> None:
    stable = AdaptiveChannelCalculator(config=ChannelConfig(stability_window=8))
    noisy = AdaptiveChannelCalculator(config=ChannelConfig(stability_window=8))

    for _ in range(10):
        stable_channels = stable.update(
            input=[0.0],
            state=_state([0.1, -0.1]),
            features=[0.0, 0.0],
            prediction=0.0,
        )

    noisy_states = [
        [0.0, 0.0],
        [10.0, -10.0],
        [0.1, 0.1],
        [-8.0, 8.0],
        [0.2, -0.2],
        [9.0, -9.0],
    ]
    for state_values in noisy_states:
        noisy_channels = noisy.update(
            input=[0.0],
            state=_state(state_values),
            features=[0.0, 0.0],
            prediction=0.0,
        )

    assert noisy_channels.stability < stable_channels.stability
    assert _channels_are_finite_and_bounded(noisy_channels)


def test_stability_drops_for_prediction_volatility() -> None:
    stable = AdaptiveChannelCalculator(config=ChannelConfig(stability_window=8))
    volatile = AdaptiveChannelCalculator(config=ChannelConfig(stability_window=8))

    for _ in range(10):
        stable_channels = stable.update(
            input=[0.0],
            state=_state([0.0, 0.0]),
            features=[0.0, 0.0],
            prediction=0.1,
        )

    for prediction in [10.0, -10.0, 10.0, -10.0, 10.0, -10.0]:
        volatile_channels = volatile.update(
            input=[0.0],
            state=_state([0.0, 0.0]),
            features=[0.0, 0.0],
            prediction=prediction,
        )

    assert volatile_channels.stability < stable_channels.stability
    assert _channels_are_finite_and_bounded(volatile_channels)


def test_stability_works_without_prediction() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(stability_window=4))

    for index in range(6):
        channels = calculator.update(
            input=[0.0],
            state=_state([float(index % 2), 0.0]),
            features=[0.0, 0.0],
            prediction=None,
        )

    assert _channels_are_finite_and_bounded(channels)
    assert calculator.prediction_count == 0


def test_stability_uses_bounded_state_and_prediction_histories() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(stability_window=3))

    for index in range(10):
        calculator.update(
            input=[0.0],
            state=_state([float(index), 0.0]),
            features=[0.0, 0.0],
            prediction=float(index),
        )

    assert calculator.state_delta_count == 3
    assert calculator.prediction_count == 3


def test_stability_handles_large_values_without_overflow() -> None:
    calculator = AdaptiveChannelCalculator(
        config=ChannelConfig(stability_window=4),
        dtype="float32",
    )
    large = 1e20

    for value in [large, -large, large, -large]:
        channels = calculator.update(
            input=[0.0],
            state=_state([value, -value], dtype=np.float32),
            features=np.asarray([0.0, 0.0], dtype=np.float32),
            prediction=value,
        )

    assert _channels_are_finite_and_bounded(channels)


def test_stability_keeps_novelty_behavior_intact() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(novelty_window=4))

    first = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=0.0,
    )
    second = calculator.update(
        input=[0.0],
        state=_state([10.0, 10.0]),
        features=[10.0, 10.0],
        prediction=0.0,
    )

    assert first.novelty == 0.0
    assert second.novelty > 0.8
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
