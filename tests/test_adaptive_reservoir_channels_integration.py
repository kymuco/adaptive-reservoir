from __future__ import annotations

import math

import pytest

from adaptive_reservoir import (
    AdaptiveChannels,
    AdaptiveReservoir,
    ChannelConfig,
    ReadoutConfig,
    ReservoirConfig,
)


def test_step_returns_calculator_backed_channels() -> None:
    model = AdaptiveReservoir(_config())

    first = model.step([0.1, -0.1], target=0.0)
    second = model.step([5.0, -5.0], target=1.0)

    assert isinstance(second.channels, AdaptiveChannels)
    assert _channels_are_finite_and_bounded(first.channels)
    assert _channels_are_finite_and_bounded(second.channels)
    assert second.channels.novelty > first.channels.novelty


def test_stable_stream_channels_are_finite_and_bounded() -> None:
    model = AdaptiveReservoir(_config())

    results = [model.step([0.1, -0.1], target=0.0) for _ in range(40)]

    assert all(_channels_are_finite_and_bounded(result.channels) for result in results)
    assert results[-1].channels.stability > 0.0
    assert results[-1].channels.novelty < 0.5
    assert results[-1].metrics.saturation_rate == pytest.approx(
        results[-1].channels.saturation,
    )


def test_noisy_stream_channels_are_finite_bounded_and_dynamic() -> None:
    model = AdaptiveReservoir(_config())
    noisy_inputs = [
        [10.0, -10.0],
        [0.1, -0.1],
        [-7.5, 7.5],
        [0.2, 0.0],
        [5.0, -2.5],
        [-0.1, 0.1],
        [-10.0, 10.0],
        [0.0, 0.0],
    ]

    results = [
        model.step(noisy_inputs[index % len(noisy_inputs)])
        for index in range(40)
    ]
    stabilities = [result.channels.stability for result in results]
    novelties = [result.channels.novelty for result in results]

    assert all(_channels_are_finite_and_bounded(result.channels) for result in results)
    assert min(stabilities) < 1.0
    assert max(novelties) > 0.0


def test_drift_stream_raises_drift_pressure() -> None:
    model = AdaptiveReservoir(_config(readout=ReadoutConfig(name="nlms", learning_rate=0.05)))

    for _ in range(12):
        baseline = model.step([0.2, -0.1], target=0.0)

    for target in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
        drift = model.step([0.2, -0.1], target=target)

    assert _channels_are_finite_and_bounded(baseline.channels)
    assert _channels_are_finite_and_bounded(drift.channels)
    assert drift.channels.drift_pressure > baseline.channels.drift_pressure


def test_step_metrics_saturation_matches_channel_saturation() -> None:
    model = AdaptiveReservoir(_config(channels=ChannelConfig(saturation_threshold=0.5)))

    result = model.step([10.0, -10.0])

    assert _channels_are_finite_and_bounded(result.channels)
    assert result.metrics.saturation_rate == pytest.approx(result.channels.saturation)


def test_reset_resets_channel_runtime_state() -> None:
    config = _config()
    model = AdaptiveReservoir(config)
    fresh = AdaptiveReservoir(config)

    for target in [0.0, 1.0, -1.0, 2.0, -2.0]:
        model.step([5.0, -5.0], target=target)
    model.reset()

    reset_result = model.step([0.1, -0.1])
    fresh_result = fresh.step([0.1, -0.1])

    _assert_channels_close(reset_result.channels, fresh_result.channels)


def _config(
    *,
    channels: ChannelConfig | None = None,
    readout: ReadoutConfig | None = None,
) -> ReservoirConfig:
    return ReservoirConfig(
        input_dim=2,
        n_cells=8,
        topology="ring_shortcuts",
        seed=123,
        feature_mode="state_raw",
        channels=ChannelConfig() if channels is None else channels,
        readout=ReadoutConfig() if readout is None else readout,
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


def _assert_channels_close(actual: AdaptiveChannels, expected: AdaptiveChannels) -> None:
    assert actual.novelty == pytest.approx(expected.novelty)
    assert actual.stability == pytest.approx(expected.stability)
    assert actual.drift_pressure == pytest.approx(expected.drift_pressure)
    assert actual.confidence == pytest.approx(expected.confidence)
    assert actual.saturation == pytest.approx(expected.saturation)
