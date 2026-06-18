"""Deterministic concept-drift benchmark."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from adaptive_reservoir import AdaptiveReservoir, ReadoutConfig, ReservoirConfig
from adaptive_reservoir.benchmarks.common import (
    BenchmarkResult,
    calculate_adapt_steps,
    readout_sparsity,
    regression_score,
    rolling_regression_scores,
)

BENCHMARK_NAME = "concept-drift"
MODEL_NAME = "adaptive_reservoir"


def run_concept_drift_benchmark(
    config: ReservoirConfig | None = None,
    *,
    seed: int = 0,
    n_samples: int = 900,
    drift_at: int = 450,
    score_window: int = 64,
) -> BenchmarkResult:
    """Run a deterministic abrupt concept-drift benchmark."""

    reservoir_config = config or _default_config(seed=seed)
    _validate_benchmark_args(
        config=reservoir_config,
        n_samples=n_samples,
        drift_at=drift_at,
        score_window=score_window,
    )

    model = AdaptiveReservoir(reservoir_config)
    predictions: list[float] = []
    targets: list[float] = []

    for sample, target in _generate_concept_drift_stream(
        seed=seed,
        n_samples=n_samples,
        drift_at=drift_at,
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
        threshold=max(0.75, pre_score * 0.9),
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


def _default_config(*, seed: int) -> ReservoirConfig:
    return ReservoirConfig(
        input_dim=2,
        n_cells=64,
        topology="ring_shortcuts",
        feature_mode="state_slow_raw",
        seed=seed,
        readout=ReadoutConfig(name="sliding_ridge", update_interval=1),
    )


def _validate_benchmark_args(
    *,
    config: ReservoirConfig,
    n_samples: int,
    drift_at: int,
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
    if drift_at <= score_window:
        msg = "drift_at must leave room for the pre-drift score window"
        raise ValueError(msg)
    if drift_at + score_window > n_samples:
        msg = "drift_at must leave room for the post-drift score window"
        raise ValueError(msg)


def _generate_concept_drift_stream(
    *,
    seed: int,
    n_samples: int,
    drift_at: int,
    input_dim: int,
) -> Iterator[tuple[tuple[float, ...], float]]:
    rng = np.random.default_rng(seed)
    for index in range(n_samples):
        sample = rng.uniform(-1.0, 1.0, size=input_dim)
        if index < drift_at:
            target = _target_before_drift(sample)
        else:
            target = _target_after_drift(sample)
        yield tuple(float(value) for value in sample), target


def _target_before_drift(sample: np.ndarray) -> float:
    return float(np.tanh(0.8 * sample[0] + 0.2 * sample[1]))


def _target_after_drift(sample: np.ndarray) -> float:
    return float(np.tanh(-0.8 * sample[0] + 0.2 * sample[1]))


__all__ = ["BENCHMARK_NAME", "MODEL_NAME", "run_concept_drift_benchmark"]
