from __future__ import annotations

import math

import numpy as np
import pytest

from adaptive_reservoir import ChannelConfig, ReservoirState
from adaptive_reservoir.channels import AdaptiveChannelCalculator
from adaptive_reservoir.core.result import AdaptiveChannels


def test_novelty_starts_at_zero_without_history() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=None,
    )

    assert channels.novelty == 0.0
    assert _channels_are_finite_and_bounded(channels)


def test_novelty_stays_low_for_repeated_stream() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(novelty_window=8))

    values = []
    for _ in range(10):
        channels = calculator.update(
            input=[0.0],
            state=_state([0.1, -0.1]),
            features=[0.1, -0.1],
            prediction=None,
        )
        values.append(channels.novelty)

    assert values[-1] == pytest.approx(0.0)


def test_novelty_increases_on_feature_shift() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(novelty_window=8))

    for _ in range(5):
        calculator.update(
            input=[0.0],
            state=_state([0.0, 0.0]),
            features=[0.0, 0.0],
            prediction=None,
        )

    shifted = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[10.0, 10.0],
        prediction=None,
    )

    assert shifted.novelty > 0.8


def test_novelty_increases_on_state_shift() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(novelty_window=8))

    for _ in range(5):
        calculator.update(
            input=[0.0],
            state=_state([0.0, 0.0]),
            features=[0.0, 0.0],
            prediction=None,
        )

    shifted = calculator.update(
        input=[0.0],
        state=_state([10.0, 10.0]),
        features=[0.0, 0.0],
        prediction=None,
    )

    assert shifted.novelty > 0.8


def test_novelty_decreases_after_new_regime_becomes_normal() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(novelty_window=4))

    for _ in range(4):
        calculator.update(
            input=[0.0],
            state=_state([0.0, 0.0]),
            features=[0.0, 0.0],
            prediction=None,
        )

    first_shift = calculator.update(
        input=[0.0],
        state=_state([10.0, 10.0]),
        features=[10.0, 10.0],
        prediction=None,
    )

    later_values = []
    for _ in range(6):
        channels = calculator.update(
            input=[0.0],
            state=_state([10.0, 10.0]),
            features=[10.0, 10.0],
            prediction=None,
        )
        later_values.append(channels.novelty)

    assert first_shift.novelty > 0.8
    assert later_values[-1] < first_shift.novelty
    assert later_values[-1] == pytest.approx(0.0)


def test_novelty_feature_and_state_histories_are_bounded() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(novelty_window=3))

    for index in range(10):
        calculator.update(
            input=[float(index)],
            state=_state([float(index), 0.0]),
            features=[float(index), 0.0],
            prediction=None,
        )

    assert calculator.feature_count == 3
    assert calculator.state_count == 3


def test_novelty_channel_is_always_finite_and_bounded() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(novelty_window=3))

    for index in range(20):
        channels = calculator.update(
            input=[float(index)],
            state=_state([float(index % 5), float(index)]),
            features=[float(index), float(index % 3)],
            prediction=None,
        )

        assert _channels_are_finite_and_bounded(channels)


def test_novelty_does_not_require_prediction_or_target() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(novelty_window=4))

    first = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
    )
    second = calculator.update(
        input=[0.0],
        state=_state([5.0, 5.0]),
        features=[5.0, 5.0],
    )

    assert first.novelty == 0.0
    assert second.novelty > first.novelty
    assert calculator.prediction_count == 0
    assert calculator.prediction_error_count == 0


def test_novelty_uses_previous_history_before_appending_current_sample() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(novelty_window=4))

    first = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=None,
    )
    second = calculator.update(
        input=[0.0],
        state=_state([10.0, 10.0]),
        features=[10.0, 10.0],
        prediction=None,
    )

    assert first.novelty == 0.0
    assert second.novelty > 0.8


def _state(values: list[float]) -> ReservoirState:
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
