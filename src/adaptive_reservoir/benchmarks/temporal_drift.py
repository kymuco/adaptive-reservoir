"""Deterministic temporal-drift benchmark."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import numpy as np

from adaptive_reservoir import AdaptiveReservoir, ReadoutConfig, ReservoirConfig
from adaptive_reservoir.benchmarks.common import (
    BenchmarkResult,
    calculate_adapt_steps,
    readout_sparsity,
    regression_score,
    rolling_regression_scores,
)

BENCHMARK_NAME = "temporal-drift"
MODEL_NAME = "adaptive_reservoir"


def run_temporal_drift_benchmark(
    config: ReservoirConfig | None = None,
    *,
    seed: int = 0,
    n_samples: int = 1200,
    drift_at: int = 600,
    delay_before: int = 3,
    delay_after: int = 12,
    score_window: int = 96,
) -> BenchmarkResult:
    """Run a deterministic benchmark where the predictive delay changes."""

    reservoir_config = _config_for_seed(config, seed=seed)
    _validate_benchmark_args(
        config=reservoir_config,
        n_samples=n_samples,
        drift_at=drift_at,
        delay_before=delay_before,
        delay_after=delay_after,
        score_window=score_window,
    )

    model = AdaptiveReservoir(reservoir_config)
    predictions: list[float] = []
    targets: list[float] = []

    for sample, target in _generate_temporal_drift_stream(
        seed=seed,
        n_samples=n_samples,
        drift_at=drift_at,
        delay_before=delay_before,
        delay_after=delay_after,
        input_dim=reservoir_config.input_dim,
    ):
        result = model.step(sample, target=target)
        predictions.append(result.prediction)
        targets.append(target)

    pre_score = regression_score(
        predictions[drift_at - score_window : drift_at],
        targets[drift_at - score_window : drift_at],
    )
    post_score = regression_score(
        predictions[drift_at : drift_at + score_window],
        targets[drift_at : drift_at + score_window],
    )
    final_score = regression_score(
        predictions[-score_window:],
        targets[-score_window:],
    )
    rolling_scores = rolling_regression_scores(
        predictions,
        targets,
        window=score_window,
    )
    recovery_search_start = drift_at + score_window - 1
    adapt_steps = calculate_adapt_steps(
        rolling_scores,
        start_index=recovery_search_start,
        threshold=max(0.70, pre_score * 0.85),
        origin_index=drift_at,
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
        input_dim=2,
        n_cells=96,
        topology="ring_shortcuts",
        feature_mode="multi_raw",
        seed=seed,
        readout=ReadoutConfig(name="sliding_ridge", update_interval=1),
    )


def _validate_benchmark_args(
    *,
    config: ReservoirConfig,
    n_samples: int,
    drift_at: int,
    delay_before: int,
    delay_after: int,
    score_window: int,
) -> None:
    if not isinstance(config, ReservoirConfig):
        msg = "config must be a ReservoirConfig"
        raise TypeError(msg)
    if config.input_dim < 2:
        msg = "config.input_dim must be at least 2"
        raise ValueError(msg)
    if n_samples <= 0:
        msg = "n_samples must be positive"
        raise ValueError(msg)
    if score_window <= 0:
        msg = "score_window must be positive"
        raise ValueError(msg)
    if delay_before <= 0 or delay_after <= 0:
        msg = "delays must be positive"
        raise ValueError(msg)
    if delay_before == delay_after:
        msg = "delay_before and delay_after must differ"
        raise ValueError(msg)
    if max(delay_before, delay_after) >= score_window:
        msg = "score_window must be larger than both delays"
        raise ValueError(msg)
    if drift_at <= score_window:
        msg = "drift_at must leave room for the pre-drift score window"
        raise ValueError(msg)
    if drift_at + score_window > n_samples:
        msg = "drift_at must leave room for the post-drift score window"
        raise ValueError(msg)


def _generate_temporal_drift_stream(
    *,
    seed: int,
    n_samples: int,
    drift_at: int,
    delay_before: int,
    delay_after: int,
    input_dim: int,
) -> Iterator[tuple[tuple[float, ...], float]]:
    warmup = max(delay_before, delay_after) + 1
    signal = _generate_signal(seed=seed, length=n_samples + warmup)
    for index in range(n_samples):
        signal_index = index + warmup
        delay = delay_before if index < drift_at else delay_after
        yield (
            _sample_at(signal, signal_index, input_dim=input_dim),
            _target_at(signal, signal_index, delay=delay),
        )


def _generate_signal(*, seed: int, length: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phase_a = float(rng.uniform(0.0, 2.0 * np.pi))
    phase_b = float(rng.uniform(0.0, 2.0 * np.pi))
    noise = rng.normal(0.0, 0.08, size=length)
    time = np.arange(length, dtype=np.float64)
    signal = (
        0.65 * np.sin(0.071 * time + phase_a)
        + 0.25 * np.sin(0.017 * time + phase_b)
        + 0.10 * noise
    )
    return np.tanh(signal).astype(np.float64)


def _sample_at(
    signal: np.ndarray,
    index: int,
    *,
    input_dim: int,
) -> tuple[float, ...]:
    current = float(signal[index])
    previous = float(signal[index - 1])
    values = [current, current - previous]
    for offset in range(2, input_dim):
        history_index = max(0, index - offset)
        values.append(float(signal[history_index]))
    return tuple(values)


def _target_at(signal: np.ndarray, index: int, *, delay: int) -> float:
    delayed = 0.75 * signal[index - delay] + 0.25 * signal[index - delay - 1]
    return float(np.tanh(delayed))


__all__ = ["BENCHMARK_NAME", "MODEL_NAME", "run_temporal_drift_benchmark"]
