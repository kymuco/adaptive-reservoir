from __future__ import annotations

import math

import numpy as np
import pytest

from adaptive_reservoir import ChannelConfig, ReservoirState
from adaptive_reservoir.channels import AdaptiveChannelCalculator
from adaptive_reservoir.core.result import AdaptiveChannels


def test_drift_pressure_starts_low_with_single_supervised_error() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig())

    channels = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=0.0,
        target=1.0,
    )

    assert channels.drift_pressure == pytest.approx(0.0)
    assert _channels_are_finite_and_bounded(channels)


def test_drift_pressure_rises_for_increasing_prediction_errors() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(drift_window=6))

    for error in [0.1, 0.2, 0.4, 0.8, 1.2, 1.6]:
        channels = calculator.update(
            input=[0.0],
            state=_state([0.0, 0.0]),
            features=[0.0, 0.0],
            prediction=0.0,
            target=error,
        )

    assert channels.drift_pressure > 0.25
    assert _channels_are_finite_and_bounded(channels)


def test_drift_pressure_stays_low_for_flat_prediction_errors() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(drift_window=6))

    for _ in range(6):
        channels = calculator.update(
            input=[0.0],
            state=_state([0.0, 0.0]),
            features=[0.0, 0.0],
            prediction=0.0,
            target=0.5,
        )

    assert channels.drift_pressure == pytest.approx(0.0)
    assert _channels_are_finite_and_bounded(channels)


def test_drift_pressure_stays_low_for_decreasing_prediction_errors() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(drift_window=6))

    for error in [1.6, 1.2, 0.8, 0.4, 0.2, 0.1]:
        channels = calculator.update(
            input=[0.0],
            state=_state([0.0, 0.0]),
            features=[0.0, 0.0],
            prediction=0.0,
            target=error,
        )

    assert channels.drift_pressure == pytest.approx(0.0)
    assert _channels_are_finite_and_bounded(channels)


def test_drift_pressure_uses_unsupervised_proxy_without_target() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(novelty_window=4))

    calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=None,
        target=None,
    )
    channels = calculator.update(
        input=[0.0],
        state=_state([10.0, 10.0]),
        features=[10.0, 10.0],
        prediction=None,
        target=None,
    )

    assert channels.novelty > 0.8
    assert channels.drift_pressure > 0.4
    assert _channels_are_finite_and_bounded(channels)


def test_drift_pressure_uses_unsupervised_proxy_when_prediction_missing() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(novelty_window=4))

    calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=None,
        target=None,
    )
    channels = calculator.update(
        input=[0.0],
        state=_state([10.0, 10.0]),
        features=[10.0, 10.0],
        prediction=None,
        target=1.0,
    )

    assert channels.drift_pressure > 0.4
    assert calculator.prediction_error_count == 0
    assert _channels_are_finite_and_bounded(channels)


def test_drift_pressure_ignores_stale_supervised_errors_without_target() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(drift_window=4))

    for error in [0.1, 0.2, 1.0, 2.0]:
        calculator.update(
            input=[0.0],
            state=_state([0.0, 0.0]),
            features=[0.0, 0.0],
            prediction=0.0,
            target=error,
        )

    channels = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=None,
        target=None,
    )

    assert channels.novelty == pytest.approx(0.0)
    assert channels.stability > 0.95
    assert channels.drift_pressure == pytest.approx(0.0)
    assert calculator.prediction_error_count == 4
    assert _channels_are_finite_and_bounded(channels)


def test_drift_pressure_error_history_is_bounded() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(drift_window=3))

    for index in range(10):
        calculator.update(
            input=[0.0],
            state=_state([0.0, 0.0]),
            features=[0.0, 0.0],
            prediction=0.0,
            target=float(index),
        )

    assert calculator.prediction_error_count == 3


def test_drift_pressure_handles_large_float64_errors_without_overflow() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(drift_window=4))
    large = 1e308

    for target in [large, large, large, large]:
        channels = calculator.update(
            input=[0.0],
            state=_state([0.0, 0.0]),
            features=[0.0, 0.0],
            prediction=0.0,
            target=target,
        )

    assert channels.drift_pressure == pytest.approx(0.0)
    assert _channels_are_finite_and_bounded(channels)


def test_drift_pressure_handles_increasing_large_float64_errors() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(drift_window=4))

    for target in [1e307, 1e307, 1e308, 1e308]:
        channels = calculator.update(
            input=[0.0],
            state=_state([0.0, 0.0]),
            features=[0.0, 0.0],
            prediction=0.0,
            target=target,
        )

    assert channels.drift_pressure > 0.3
    assert _channels_are_finite_and_bounded(channels)


def test_drift_pressure_handles_large_opposite_prediction_and_target() -> None:
    calculator = AdaptiveChannelCalculator(config=ChannelConfig(drift_window=2))
    large = 1e308

    calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=large,
        target=-large,
    )
    channels = calculator.update(
        input=[0.0],
        state=_state([0.0, 0.0]),
        features=[0.0, 0.0],
        prediction=-large,
        target=large,
    )

    assert _channels_are_finite_and_bounded(channels)
    assert calculator.prediction_error_count == 2


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
