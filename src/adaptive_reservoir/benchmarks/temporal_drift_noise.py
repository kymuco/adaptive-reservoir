"""Deterministic data generation for TDNR-01.

This module is intentionally independent of reservoir models, readouts,
benchmark runners, and evaluation metrics. It generates aligned training
and evaluation data without changing existing benchmark behavior.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Scenario = Literal["stationary", "abrupt", "gradual", "recurring"]
SignalKind = Literal["sinusoidal", "ar1"]
FloatArray = NDArray[np.float64]

SCENARIOS = ("stationary", "abrupt", "gradual", "recurring")
SIGNAL_KINDS = ("sinusoidal", "ar1")

# Stable component identifiers. Changing these changes generated datasets.
_SIGNAL_STREAM = 1
_INPUT_NOISE_STREAM = 2
_TARGET_NOISE_STREAM = 3


@dataclass(frozen=True, slots=True)
class TemporalDriftNoiseConfig:
    """Configuration for a deterministic noisy temporal-drift dataset.

    Noise levels are non-negative standard deviation multipliers relative
    to the standard deviation of the latent signal before drift.

    The drift index is zero-based and excludes generator history.
    For gradual drift, interpolation begins at drift_at and reaches the
    new target function at drift_at + gradual_steps.
    For recurring drift, return_at is the first sample of the restored
    original regime. If omitted, it defaults to the midpoint of the
    remaining stream.
    """

    seed: int = 0
    n_samples: int = 4000
    drift_at: int = 2000
    delay_before: int = 3
    delay_after: int = 12
    input_noise: float = 0.0
    target_noise: float = 0.0
    scenario: Scenario = "abrupt"
    signal_kind: SignalKind = "sinusoidal"
    gradual_steps: int = 400
    return_at: int | None = None

    def __post_init__(self) -> None:
        _validate_int("seed", self.seed, minimum=0)
        _validate_int("n_samples", self.n_samples, minimum=3)
        _validate_int("drift_at", self.drift_at, minimum=2)
        _validate_int("delay_before", self.delay_before, minimum=1)
        _validate_int("delay_after", self.delay_after, minimum=1)
        _validate_int("gradual_steps", self.gradual_steps, minimum=1)
        _validate_noise("input_noise", self.input_noise)
        _validate_noise("target_noise", self.target_noise)

        if self.drift_at >= self.n_samples:
            raise ValueError("drift_at must be smaller than n_samples")
        if self.delay_before == self.delay_after:
            raise ValueError("delay_before and delay_after must differ")
        if self.scenario not in SCENARIOS:
            raise ValueError(f"unsupported scenario: {self.scenario!r}")
        if self.signal_kind not in SIGNAL_KINDS:
            raise ValueError(f"unsupported signal_kind: {self.signal_kind!r}")

        if self.scenario == "gradual":
            if self.drift_at + self.gradual_steps >= self.n_samples:
                raise ValueError("gradual drift must finish before the stream ends")

        if self.return_at is not None:
            _validate_int("return_at", self.return_at, minimum=1)
            if self.scenario != "recurring":
                raise ValueError("return_at is only valid for recurring drift")

        if self.scenario == "recurring":
            return_at = self.resolved_return_at
            if not self.drift_at < return_at < self.n_samples:
                raise ValueError("return_at must be after drift_at and before n_samples")

    @property
    def history_size(self) -> int:
        """Number of generated history samples preceding sample zero.

        Both target functions use a delayed sample and its predecessor.
        Keeping history independent of scenario and noise settings makes
        comparisons on the same latent realization possible.
        """
        return max(self.delay_before, self.delay_after) + 1

    @property
    def resolved_return_at(self) -> int:
        """Return the explicit or default recurring-drift boundary."""
        if self.return_at is not None:
            return self.return_at
        return self.drift_at + (self.n_samples - self.drift_at) // 2


@dataclass(frozen=True, slots=True, eq=False)
class TemporalDriftNoiseStream:
    """Aligned deterministic training data and offline evaluation references.

    inputs has shape (n_samples, 2), containing the current noisy
    observation and its first difference.

    clean_targets, training_targets, and drift_weight have shape
    (n_samples,). The latent and observed signals include history, so
    sample t corresponds to signal index history_size + t.

    Latent values, clean targets, and drift weights are evaluation-only
    references. Online models must consume inputs and training targets,
    not those references or future observations.

    Arrays returned by the generator are marked read-only.
    """

    config: TemporalDriftNoiseConfig
    latent_signal: FloatArray
    observed_signal: FloatArray
    inputs: FloatArray
    clean_targets: FloatArray
    training_targets: FloatArray
    drift_weight: FloatArray
    reference_scale: float

    @property
    def history_size(self) -> int:
        return self.config.history_size

    @property
    def n_samples(self) -> int:
        return self.config.n_samples

    def iter_training(self) -> Iterator[tuple[tuple[float, float], float]]:
        """Yield chronological observations and supervised targets.

        Consumers must predict before updating with the yielded target.
        This iterator performs no model fitting or evaluation itself.
        """
        for sample, target in zip(self.inputs, self.training_targets, strict=True):
            yield (float(sample[0]), float(sample[1])), float(target)


def generate_temporal_drift_noise_stream(
    config: TemporalDriftNoiseConfig | None = None,
) -> TemporalDriftNoiseStream:
    """Generate one deterministic TDNR-01 dataset.

    Independent RNG streams isolate latent signal generation, observation
    noise, and training-target noise. The latent signal and clean targets
    are therefore invariant to changes in either noise level.

    No global RNG state is read or modified. No reservoir is instantiated.
    """
    if config is None:
        config = TemporalDriftNoiseConfig()
    if not isinstance(config, TemporalDriftNoiseConfig):
        raise TypeError("config must be a TemporalDriftNoiseConfig")

    history = config.history_size
    length = history + config.n_samples

    latent = _generate_latent_signal(
        kind=config.signal_kind,
        length=length,
        rng=_make_rng(config.seed, _SIGNAL_STREAM),
    )

    reference_scale = float(np.std(latent[history : history + config.drift_at]))
    if not math.isfinite(reference_scale) or reference_scale <= 0.0:
        raise ValueError("latent pre-drift standard deviation must be positive and finite")

    input_noise = _make_rng(config.seed, _INPUT_NOISE_STREAM).standard_normal(length)
    target_noise = _make_rng(config.seed, _TARGET_NOISE_STREAM).standard_normal(
        config.n_samples,
    )

    weights = _generate_drift_weights(config)
    clean_targets = _generate_clean_targets(latent, config, weights)

    with np.errstate(over="ignore", invalid="ignore"):
        observed = latent + float(config.input_noise) * reference_scale * input_noise
        training_targets = (
            clean_targets + float(config.target_noise) * reference_scale * target_noise
        )
        current = observed[history:]
        previous = observed[history - 1 : -1]
        inputs = np.column_stack((current, current - previous))

    arrays = (latent, observed, inputs, clean_targets, training_targets, weights)
    for array in arrays:
        if not np.all(np.isfinite(array)):
            raise ValueError("generated stream contains non-finite values")
        array.setflags(write=False)

    return TemporalDriftNoiseStream(
        config=config,
        latent_signal=latent,
        observed_signal=observed,
        inputs=inputs,
        clean_targets=clean_targets,
        training_targets=training_targets,
        drift_weight=weights,
        reference_scale=reference_scale,
    )


def _make_rng(seed: int, component: int) -> np.random.Generator:
    """Create an explicitly selected, component-local random generator."""
    sequence = np.random.SeedSequence([seed, component])
    return np.random.Generator(np.random.PCG64(sequence))


def _generate_latent_signal(
    *,
    kind: SignalKind,
    length: int,
    rng: np.random.Generator,
) -> FloatArray:
    if kind == "sinusoidal":
        phase_a, phase_b = rng.uniform(0.0, 2.0 * np.pi, size=2)
        time = np.arange(length, dtype=np.float64)
        signal = (
            0.65 * np.sin(0.071 * time + phase_a)
            + 0.25 * np.sin(0.017 * time + phase_b)
        )
        return np.tanh(signal)

    if kind == "ar1":
        # Stationary unit-variance initialization before tanh compression.
        coefficient = 0.8
        innovation_scale = math.sqrt(1.0 - coefficient**2)
        innovations = rng.standard_normal(length)
        signal = np.empty(length, dtype=np.float64)
        signal[0] = innovations[0]
        for index in range(1, length):
            signal[index] = (
                coefficient * signal[index - 1]
                + innovation_scale * innovations[index]
            )
        return np.tanh(signal)

    raise ValueError(f"unsupported signal kind: {kind!r}")


def _generate_drift_weights(config: TemporalDriftNoiseConfig) -> FloatArray:
    """Return the interpolation weight of the post-drift target function."""
    weights = np.zeros(config.n_samples, dtype=np.float64)

    if config.scenario == "stationary":
        return weights

    if config.scenario == "abrupt":
        weights[config.drift_at :] = 1.0
        return weights

    if config.scenario == "gradual":
        start = config.drift_at
        stop = start + config.gradual_steps
        weights[start:stop] = (
            np.arange(config.gradual_steps, dtype=np.float64) / config.gradual_steps
        )
        weights[stop:] = 1.0
        return weights

    if config.scenario == "recurring":
        weights[config.drift_at : config.resolved_return_at] = 1.0
        return weights

    raise ValueError(f"unsupported scenario: {config.scenario!r}")


def _generate_clean_targets(
    latent: FloatArray,
    config: TemporalDriftNoiseConfig,
    weights: FloatArray,
) -> FloatArray:
    indices = np.arange(
        config.history_size,
        config.history_size + config.n_samples,
    )
    before = _target_for_delay(latent, indices, config.delay_before)
    after = _target_for_delay(latent, indices, config.delay_after)
    return (1.0 - weights) * before + weights * after


def _target_for_delay(
    latent: FloatArray,
    indices: NDArray[np.int_],
    delay: int,
) -> FloatArray:
    delayed = 0.75 * latent[indices - delay] + 0.25 * latent[indices - delay - 1]
    return np.tanh(delayed)


def _validate_int(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _validate_noise(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        numeric_value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")


__all__ = [
    "TemporalDriftNoiseConfig",
    "TemporalDriftNoiseStream",
    "generate_temporal_drift_noise_stream",
]
