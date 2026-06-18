"""Deterministic delayed XOR benchmark."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace

import numpy as np

from adaptive_reservoir import AdaptiveReservoir, ReadoutConfig, ReservoirConfig
from adaptive_reservoir.benchmarks.common import (
    BenchmarkResult,
    calculate_adapt_steps,
    readout_sparsity,
)

BENCHMARK_NAME = "delayed-xor"
MODEL_NAME = "adaptive_reservoir"
_RECOVERY_THRESHOLD = 0.75


def run_delayed_xor_benchmark(
    config: ReservoirConfig | None = None,
    *,
    seed: int = 0,
    n_samples: int = 1200,
    delay_a: int = 3,
    delay_b: int = 7,
    score_window: int = 128,
) -> BenchmarkResult:
    """Run a deterministic delayed XOR memory + nonlinearity benchmark."""

    reservoir_config = _config_for_seed(config, seed=seed)
    _validate_benchmark_args(
        config=reservoir_config,
        n_samples=n_samples,
        delay_a=delay_a,
        delay_b=delay_b,
        score_window=score_window,
    )

    model = AdaptiveReservoir(reservoir_config)
    predictions: list[float] = []
    targets: list[float] = []

    for sample, target in _generate_delayed_xor_stream(
        seed=seed,
        n_samples=n_samples,
        delay_a=delay_a,
        delay_b=delay_b,
        input_dim=reservoir_config.input_dim,
    ):
        result = model.step(sample, target=target)
        predictions.append(result.prediction)
        targets.append(target)

    mid_start = (n_samples - score_window) // 2
    pre_score = _binary_accuracy_score(
        predictions[:score_window],
        targets[:score_window],
    )
    post_score = _binary_accuracy_score(
        predictions[mid_start : mid_start + score_window],
        targets[mid_start : mid_start + score_window],
    )
    final_score = _binary_accuracy_score(
        predictions[-score_window:],
        targets[-score_window:],
    )
    rolling_scores = _rolling_binary_accuracy_scores(
        predictions,
        targets,
        window=score_window,
    )
    adapt_steps = calculate_adapt_steps(
        rolling_scores,
        start_index=score_window - 1,
        threshold=_RECOVERY_THRESHOLD,
        origin_index=0,
    )
    metrics = model.metrics_snapshot()

    return BenchmarkResult(
        benchmark=BENCHMARK_NAME,
        model=MODEL_NAME,
        topology=reservoir_config.topology,
        feature_mode=reservoir_config.feature_mode,
        readout=reservoir_config.readout.name,
        seed=seed,
        pre_score=pre_score,
        post_score=post_score,
        final_score=final_score,
        adapt_steps=adapt_steps,
        us_per_sample=metrics.us_per_sample_avg,
        saturation_rate=metrics.saturation_rate_avg,
        readout_sparsity=readout_sparsity(model),
        samples_seen=metrics.samples_seen,
    )


def _config_for_seed(config: ReservoirConfig | None, *, seed: int) -> ReservoirConfig:
    if config is None:
        return _default_config(seed=seed)
    return replace(config, seed=seed)


def _default_config(*, seed: int) -> ReservoirConfig:
    return ReservoirConfig(
        input_dim=1,
        n_cells=96,
        topology="modular_small_world",
        feature_mode="multi_raw",
        seed=seed,
        readout=ReadoutConfig(name="sliding_ridge", update_interval=1),
    )


def _validate_benchmark_args(
    *,
    config: ReservoirConfig,
    n_samples: int,
    delay_a: int,
    delay_b: int,
    score_window: int,
) -> None:
    if not isinstance(config, ReservoirConfig):
        msg = "config must be a ReservoirConfig"
        raise TypeError(msg)
    if config.input_dim < 1:
        msg = "config.input_dim must be at least 1"
        raise ValueError(msg)
    if n_samples <= 0:
        msg = "n_samples must be positive"
        raise ValueError(msg)
    if score_window <= 0:
        msg = "score_window must be positive"
        raise ValueError(msg)
    if delay_a <= 0 or delay_b <= 0:
        msg = "delays must be positive"
        raise ValueError(msg)
    if delay_a == delay_b:
        msg = "delay_a and delay_b must differ"
        raise ValueError(msg)
    if max(delay_a, delay_b) >= score_window:
        msg = "score_window must be larger than both delays"
        raise ValueError(msg)
    if n_samples < score_window * 3:
        msg = "n_samples must fit early, middle, and final score windows"
        raise ValueError(msg)


def _generate_delayed_xor_stream(
    *,
    seed: int,
    n_samples: int,
    delay_a: int,
    delay_b: int,
    input_dim: int,
) -> Iterator[tuple[tuple[float, ...], float]]:
    warmup = max(delay_a, delay_b, input_dim - 1)
    bits = _generate_bits(seed=seed, length=n_samples + warmup)
    for index in range(n_samples):
        bit_index = index + warmup
        yield (
            _sample_at(bits, bit_index, input_dim=input_dim),
            _target_at(bits, bit_index, delay_a=delay_a, delay_b=delay_b),
        )


def _generate_bits(*, seed: int, length: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=length, dtype=np.int8)


def _sample_at(
    bits: np.ndarray,
    index: int,
    *,
    input_dim: int,
) -> tuple[float, ...]:
    return tuple(float(bits[index - offset]) for offset in range(input_dim))


def _target_at(
    bits: np.ndarray,
    index: int,
    *,
    delay_a: int,
    delay_b: int,
) -> float:
    left = int(bits[index - delay_a])
    right = int(bits[index - delay_b])
    return float(left ^ right)


def _binary_accuracy_score(
    predictions: Sequence[float],
    targets: Sequence[float],
) -> float:
    prediction_array = _finite_vector(predictions, "predictions")
    target_array = _finite_vector(targets, "targets")
    if prediction_array.size != target_array.size:
        msg = "predictions and targets must have the same length"
        raise ValueError(msg)
    if prediction_array.size == 0:
        msg = "predictions and targets must not be empty"
        raise ValueError(msg)
    predicted_bits = prediction_array >= 0.5
    target_bits = target_array >= 0.5
    return float(np.mean(predicted_bits == target_bits))


def _rolling_binary_accuracy_scores(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    window: int,
) -> tuple[float | None, ...]:
    if window <= 0:
        msg = "window must be positive"
        raise ValueError(msg)
    prediction_array = _finite_vector(predictions, "predictions")
    target_array = _finite_vector(targets, "targets")
    if prediction_array.size != target_array.size:
        msg = "predictions and targets must have the same length"
        raise ValueError(msg)
    scores: list[float | None] = []
    for index in range(prediction_array.size):
        if index + 1 < window:
            scores.append(None)
            continue
        start = index + 1 - window
        scores.append(
            _binary_accuracy_score(
                prediction_array[start : index + 1],
                target_array[start : index + 1],
            )
        )
    return tuple(scores)


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        msg = f"{name} must be a numeric sequence"
        raise ValueError(msg)
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must contain only numeric values"
        raise ValueError(msg) from exc
    if array.ndim != 1:
        msg = f"{name} must be a 1D sequence"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return array


__all__ = ["BENCHMARK_NAME", "MODEL_NAME", "run_delayed_xor_benchmark"]
