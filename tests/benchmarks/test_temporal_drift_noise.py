"""Behavioral contract tests for the TDNR-01 Stage 1 generator."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from adaptive_reservoir.benchmarks.temporal_drift_noise import (
    TemporalDriftNoiseConfig,
    TemporalDriftNoiseStream,
    generate_temporal_drift_noise_stream,
)

ARRAY_FIELDS = (
    "latent_signal",
    "observed_signal",
    "inputs",
    "clean_targets",
    "training_targets",
    "drift_weight",
)


def _config(**changes: object) -> TemporalDriftNoiseConfig:
    config = TemporalDriftNoiseConfig(
        seed=17,
        n_samples=256,
        drift_at=128,
        delay_before=3,
        delay_after=12,
        gradual_steps=32,
    )
    return replace(config, **changes)


def _target_at(stream: TemporalDriftNoiseStream, index: int, delay: int) -> float:
    signal_index = stream.history_size + index
    latent = stream.latent_signal
    return float(
        np.tanh(
            0.75 * latent[signal_index - delay]
            + 0.25 * latent[signal_index - delay - 1]
        )
    )


def test_public_generator_defaults() -> None:
    stream = generate_temporal_drift_noise_stream()

    assert isinstance(stream, TemporalDriftNoiseStream)
    assert stream.config == TemporalDriftNoiseConfig()
    assert stream.n_samples == 4000
    assert stream.config.drift_at == 2000
    assert stream.inputs.shape == (4000, 2)
    assert stream.drift_weight[1999] == 0.0
    assert stream.drift_weight[2000] == 1.0


@pytest.mark.parametrize("signal_kind", ["sinusoidal", "ar1"])
@pytest.mark.parametrize("scenario", ["stationary", "abrupt", "gradual", "recurring"])
def test_identical_config_produces_identical_arrays(signal_kind: str, scenario: str) -> None:
    config = _config(
        signal_kind=signal_kind,
        scenario=scenario,
        input_noise=0.2,
        target_noise=0.1,
    )
    first = generate_temporal_drift_noise_stream(config)
    second = generate_temporal_drift_noise_stream(config)

    assert first.reference_scale == second.reference_scale
    for name in ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


@pytest.mark.parametrize("signal_kind", ["sinusoidal", "ar1"])
def test_changing_seed_changes_latent_signal(signal_kind: str) -> None:
    first = generate_temporal_drift_noise_stream(_config(seed=10, signal_kind=signal_kind))
    second = generate_temporal_drift_noise_stream(_config(seed=11, signal_kind=signal_kind))

    assert not np.array_equal(first.latent_signal, second.latent_signal)
    assert not np.array_equal(first.clean_targets, second.clean_targets)
    np.testing.assert_array_equal(first.drift_weight, second.drift_weight)


def test_shapes_dtype_finiteness_and_read_only_contract() -> None:
    config = _config(input_noise=0.8, target_noise=0.4)
    stream = generate_temporal_drift_noise_stream(config)

    assert stream.history_size == 13
    assert stream.latent_signal.shape == (config.n_samples + stream.history_size,)
    assert stream.observed_signal.shape == stream.latent_signal.shape
    assert stream.inputs.shape == (config.n_samples, 2)
    assert stream.clean_targets.shape == (config.n_samples,)
    assert stream.training_targets.shape == (config.n_samples,)
    assert stream.drift_weight.shape == (config.n_samples,)
    assert np.isfinite(stream.reference_scale)
    assert stream.reference_scale > 0.0

    for name in ARRAY_FIELDS:
        values = getattr(stream, name)
        assert values.dtype == np.dtype(np.float64)
        assert np.all(np.isfinite(values))
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.flat[0] = 123.0


def test_zero_noise_exposes_exact_clean_observations_and_targets() -> None:
    stream = generate_temporal_drift_noise_stream(_config())
    np.testing.assert_array_equal(stream.observed_signal, stream.latent_signal)
    np.testing.assert_array_equal(stream.training_targets, stream.clean_targets)


def test_observation_noise_does_not_modify_latent_or_target_data() -> None:
    clean = generate_temporal_drift_noise_stream(_config(target_noise=0.1))
    noisy = generate_temporal_drift_noise_stream(
        _config(input_noise=0.4, target_noise=0.1),
    )

    np.testing.assert_array_equal(clean.latent_signal, noisy.latent_signal)
    np.testing.assert_array_equal(clean.clean_targets, noisy.clean_targets)
    np.testing.assert_array_equal(clean.training_targets, noisy.training_targets)
    np.testing.assert_array_equal(clean.drift_weight, noisy.drift_weight)
    assert clean.reference_scale == noisy.reference_scale
    assert not np.array_equal(clean.observed_signal, noisy.observed_signal)
    assert not np.array_equal(clean.inputs, noisy.inputs)


def test_target_noise_does_not_modify_inputs_or_clean_targets() -> None:
    clean = generate_temporal_drift_noise_stream(_config(input_noise=0.2))
    noisy = generate_temporal_drift_noise_stream(
        _config(input_noise=0.2, target_noise=0.4),
    )

    np.testing.assert_array_equal(clean.latent_signal, noisy.latent_signal)
    np.testing.assert_array_equal(clean.observed_signal, noisy.observed_signal)
    np.testing.assert_array_equal(clean.inputs, noisy.inputs)
    np.testing.assert_array_equal(clean.clean_targets, noisy.clean_targets)
    assert not np.array_equal(clean.training_targets, noisy.training_targets)


def test_noise_amplitudes_share_the_same_random_realizations() -> None:
    low = generate_temporal_drift_noise_stream(
        _config(input_noise=0.1, target_noise=0.05),
    )
    high = generate_temporal_drift_noise_stream(
        _config(input_noise=0.4, target_noise=0.2),
    )

    np.testing.assert_allclose(
        high.observed_signal - high.latent_signal,
        4.0 * (low.observed_signal - low.latent_signal),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        high.training_targets - high.clean_targets,
        4.0 * (low.training_targets - low.clean_targets),
        rtol=1e-10,
        atol=1e-12,
    )


def test_noise_scaling_uses_only_aligned_pre_drift_latent_samples() -> None:
    config = _config(input_noise=0.2, target_noise=0.1)
    stream = generate_temporal_drift_noise_stream(config)
    start = stream.history_size
    stop = start + config.drift_at
    expected_scale = float(np.std(stream.latent_signal[start:stop]))

    assert stream.reference_scale == expected_scale

    # Verify explicit RNG component separation and noise scaling.
    input_rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([config.seed, 2])),
    )
    target_rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([config.seed, 3])),
    )
    input_draws = input_rng.standard_normal(stream.latent_signal.size)
    target_draws = target_rng.standard_normal(config.n_samples)

    np.testing.assert_array_equal(
        stream.observed_signal,
        stream.latent_signal + config.input_noise * expected_scale * input_draws,
    )
    np.testing.assert_array_equal(
        stream.training_targets,
        stream.clean_targets + config.target_noise * expected_scale * target_draws,
    )
    assert not np.array_equal(input_draws[: config.n_samples], target_draws)


def test_inputs_are_causal_features_of_noisy_observations() -> None:
    stream = generate_temporal_drift_noise_stream(_config(input_noise=0.4))
    history = stream.history_size
    current = stream.observed_signal[history:]
    previous = stream.observed_signal[history - 1 : -1]

    np.testing.assert_array_equal(stream.inputs[:, 0], current)
    np.testing.assert_array_equal(stream.inputs[:, 1], current - previous)
    assert stream.inputs[0, 1] == (
        stream.observed_signal[history] - stream.observed_signal[history - 1]
    )


def test_abrupt_drift_targets_switch_at_exact_boundary() -> None:
    config = _config(scenario="abrupt", input_noise=0.8)
    stream = generate_temporal_drift_noise_stream(config)

    np.testing.assert_array_equal(stream.drift_weight[: config.drift_at], 0.0)
    np.testing.assert_array_equal(stream.drift_weight[config.drift_at :], 1.0)

    expected = np.array(
        [
            _target_at(
                stream,
                index,
                config.delay_before if index < config.drift_at else config.delay_after,
            )
            for index in range(config.n_samples)
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(stream.clean_targets, expected, rtol=0.0, atol=1e-15)


def test_stationary_control_never_switches_target_function() -> None:
    config = _config(scenario="stationary", input_noise=0.2)
    stream = generate_temporal_drift_noise_stream(config)
    expected = np.array(
        [_target_at(stream, index, config.delay_before) for index in range(config.n_samples)],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(stream.drift_weight, 0.0)
    np.testing.assert_allclose(stream.clean_targets, expected, rtol=0.0, atol=1e-15)


def test_gradual_drift_interpolates_target_functions() -> None:
    config = _config(scenario="gradual", gradual_steps=32)
    stream = generate_temporal_drift_noise_stream(config)
    start = config.drift_at
    stop = start + config.gradual_steps

    np.testing.assert_array_equal(stream.drift_weight[:start], 0.0)
    np.testing.assert_array_equal(
        stream.drift_weight[start:stop],
        np.arange(config.gradual_steps, dtype=np.float64) / config.gradual_steps,
    )
    np.testing.assert_array_equal(stream.drift_weight[stop:], 1.0)
    assert stream.drift_weight[start + 16] == 0.5
    assert np.all(np.diff(stream.drift_weight) >= 0.0)

    expected = []
    for index, weight in enumerate(stream.drift_weight):
        before = _target_at(stream, index, config.delay_before)
        after = _target_at(stream, index, config.delay_after)
        expected.append(float((1.0 - weight) * before + weight * after))

    np.testing.assert_allclose(stream.clean_targets, expected, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("return_at", [None, 200])
def test_recurring_drift_restores_original_target_function(return_at: int | None) -> None:
    config = _config(scenario="recurring", return_at=return_at)
    stream = generate_temporal_drift_noise_stream(config)
    boundary = 192 if return_at is None else return_at

    assert config.resolved_return_at == boundary
    np.testing.assert_array_equal(stream.drift_weight[: config.drift_at], 0.0)
    np.testing.assert_array_equal(stream.drift_weight[config.drift_at : boundary], 1.0)
    np.testing.assert_array_equal(stream.drift_weight[boundary:], 0.0)

    expected = []
    for index in range(config.n_samples):
        delay = config.delay_before
        if config.drift_at <= index < boundary:
            delay = config.delay_after
        expected.append(_target_at(stream, index, delay))

    np.testing.assert_allclose(stream.clean_targets, expected, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("scenario", ["stationary", "gradual", "recurring"])
def test_scenario_changes_do_not_change_observation_realization(scenario: str) -> None:
    config = _config(input_noise=0.2)
    reference = generate_temporal_drift_noise_stream(config)
    variant = generate_temporal_drift_noise_stream(replace(config, scenario=scenario))

    np.testing.assert_array_equal(reference.latent_signal, variant.latent_signal)
    np.testing.assert_array_equal(reference.observed_signal, variant.observed_signal)
    np.testing.assert_array_equal(reference.inputs, variant.inputs)
    np.testing.assert_array_equal(
        reference.clean_targets[: config.drift_at],
        variant.clean_targets[: config.drift_at],
    )
    assert reference.reference_scale == variant.reference_scale


@pytest.mark.parametrize("signal_kind", ["sinusoidal", "ar1"])
def test_extending_abrupt_stream_preserves_existing_prefix(signal_kind: str) -> None:
    config = _config(signal_kind=signal_kind, input_noise=0.2, target_noise=0.1)
    short = generate_temporal_drift_noise_stream(config)
    long = generate_temporal_drift_noise_stream(replace(config, n_samples=512))

    assert short.reference_scale == long.reference_scale
    for name in ARRAY_FIELDS:
        short_array = getattr(short, name)
        long_array = getattr(long, name)
        np.testing.assert_array_equal(short_array, long_array[: len(short_array)])


def test_training_iterator_returns_aligned_observations_and_training_targets() -> None:
    stream = generate_temporal_drift_noise_stream(
        _config(input_noise=0.2, target_noise=0.3),
    )
    rows = list(stream.iter_training())
    assert len(rows) == stream.n_samples

    for index, (sample, target) in enumerate(rows):
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert all(isinstance(value, float) for value in sample)
        assert isinstance(target, float)
        np.testing.assert_array_equal(sample, stream.inputs[index])
        assert target == stream.training_targets[index]

    assert any(
        target != clean
        for (_, target), clean in zip(rows, stream.clean_targets, strict=True)
    )
    assert rows == list(stream.iter_training())


def test_generator_does_not_modify_global_numpy_rng_state() -> None:
    before = np.random.get_state()
    generate_temporal_drift_noise_stream(_config(input_noise=0.4, target_noise=0.2))
    after = np.random.get_state()

    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"n_samples": 0}, "n_samples"),
        ({"n_samples": True}, "n_samples"),
        ({"drift_at": 0}, "drift_at"),
        ({"drift_at": 1}, "drift_at"),
        ({"drift_at": 256}, "drift_at"),
        ({"delay_before": 0}, "delay_before"),
        ({"delay_after": -1}, "delay_after"),
        ({"delay_after": 3}, "differ"),
        ({"input_noise": -0.1}, "input_noise"),
        ({"input_noise": float("nan")}, "input_noise"),
        ({"input_noise": float("inf")}, "input_noise"),
        ({"input_noise": True}, "input_noise"),
        ({"input_noise": "0.2"}, "input_noise"),
        ({"target_noise": -0.1}, "target_noise"),
        ({"target_noise": float("nan")}, "target_noise"),
        ({"target_noise": float("inf")}, "target_noise"),
        ({"scenario": "unknown"}, "scenario"),
        ({"signal_kind": "unknown"}, "signal_kind"),
        ({"gradual_steps": 0}, "gradual_steps"),
        ({"scenario": "gradual", "gradual_steps": 128}, "finish"),
        ({"scenario": "recurring", "return_at": 128}, "return_at"),
        ({"scenario": "recurring", "return_at": 256}, "return_at"),
        ({"scenario": "recurring", "return_at": 150.5}, "return_at"),
        ({"scenario": "abrupt", "return_at": 200}, "return_at"),
        ({"scenario": "recurring", "drift_at": 255}, "return_at"),
    ],
)
def test_invalid_configuration_is_rejected(changes: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _config(**changes)


def test_generator_rejects_non_config_argument() -> None:
    with pytest.raises(TypeError, match="TemporalDriftNoiseConfig"):
        generate_temporal_drift_noise_stream({})
