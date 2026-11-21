"""Deterministic, prequential TDNR-01 Stage 2B comparison.

Compare AdaptiveReservoir and three causal baselines on identical
Stage 1 abrupt-drift streams. Models receive only the current input
and training target. Clean targets are reserved for offline scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from adaptive_reservoir import AdaptiveReservoir, ReadoutConfig, ReservoirConfig
from adaptive_reservoir.benchmarks.tdnr_baselines import (
    LagBufferRidgeBaseline,
    PersistenceBaseline,
    ZeroBaseline,
)
from adaptive_reservoir.benchmarks.temporal_drift_noise import (
    TemporalDriftNoiseConfig,
    TemporalDriftNoiseStream,
    generate_temporal_drift_noise_stream,
)

DEFAULT_SEEDS = (0, 1, 2)
MODEL_NAMES = (
    "adaptive_reservoir",
    "zero",
    "persistence",
    "lag_buffer_ridge",
)
WINDOW_NAMES = ("pre", "early_post", "late_post")


@dataclass(frozen=True, slots=True)
class WindowScore:
    """Offline clean-target errors over a half-open interval [start, stop)."""

    name: str
    start: int
    stop: int
    mae: float
    rmse: float


@dataclass(frozen=True, slots=True)
class ModelComparison:
    """One model's chronological predictions and disjoint window scores."""

    model: str
    seed: int
    predictions: tuple[float, ...]
    windows: tuple[WindowScore, ...]


def run_tdnr_comparison(
    config: TemporalDriftNoiseConfig | None = None,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    score_window: int = 128,
    reservoir_config: ReservoirConfig | None = None,
) -> tuple[ModelComparison, ...]:
    """Compare four models for each explicitly fixed dataset seed.

    The default Stage 1 stream has 4000 samples, abrupt drift at 2000,
    input noise 0.2 and target noise 0.0. Explicit seeds replace the seed
    stored in config. Models are reconstructed independently for each seed.

    Results are ordered by seed, then MODEL_NAMES. No experiments are
    performed at import time. Timing is intentionally not scored.
    """
    if config is None:
        config = TemporalDriftNoiseConfig(input_noise=0.2)
    if not isinstance(config, TemporalDriftNoiseConfig):
        raise TypeError("config must be a TemporalDriftNoiseConfig")
    _validate_seeds(seeds)
    _validate_window(config.n_samples, config.drift_at, score_window)
    _validate_reservoir_config(reservoir_config)
    if config.scenario != "abrupt":
        raise ValueError("TDNR Stage 2B requires an abrupt scenario")

    results: list[ModelComparison] = []
    for seed in seeds:
        stream = generate_temporal_drift_noise_stream(replace(config, seed=seed))
        results.extend(
            compare_tdnr_stream(
                stream,
                score_window=score_window,
                reservoir_config=reservoir_config,
            )
        )
    return tuple(results)


def compare_tdnr_stream(
    stream: TemporalDriftNoiseStream,
    *,
    score_window: int = 128,
    reservoir_config: ReservoirConfig | None = None,
) -> tuple[ModelComparison, ...]:
    """Compare models on one existing Stage 1 abrupt-drift stream.

    This entrypoint permits testing training/score separation without
    modifying the Stage 1 generator. Each model receives a fresh iterator
    of exactly (input, training_target) pairs. Evaluation-only fields
    are accessed after predictions have been collected.
    """
    if not isinstance(stream, TemporalDriftNoiseStream):
        raise TypeError("stream must be a TemporalDriftNoiseStream")
    _validate_window(stream.n_samples, stream.config.drift_at, score_window)
    _validate_reservoir_config(reservoir_config)
    if stream.config.scenario != "abrupt":
        raise ValueError("TDNR Stage 2B requires an abrupt scenario")
    if stream.inputs.shape != (stream.n_samples, 2):
        raise ValueError("inputs must have shape (n_samples, 2)")
    if stream.training_targets.shape != (stream.n_samples,):
        raise ValueError("training_targets must have shape (n_samples,)")
    if stream.clean_targets.shape != (stream.n_samples,):
        raise ValueError("clean_targets must have shape (n_samples,)")
    if not all(
        np.all(np.isfinite(values))
        for values in (stream.inputs, stream.training_targets, stream.clean_targets)
    ):
        raise ValueError("comparison arrays must contain only finite values")

    reservoir = AdaptiveReservoir(
        _reservoir_config(stream.config.seed, reservoir_config)
    )

    def reservoir_step(sample: tuple[float, float], target: float) -> float:
        return float(reservoir.step(sample, target=target).prediction)

    models = (
        ("adaptive_reservoir", reservoir_step),
        ("zero", ZeroBaseline().step),
        ("persistence", PersistenceBaseline().step),
        ("lag_buffer_ridge", LagBufferRidgeBaseline().step),
    )

    results: list[ModelComparison] = []
    for name, step in models:
        predictions = tuple(
            float(step(sample, target))
            for sample, target in stream.iter_training()
        )
        if len(predictions) != stream.n_samples:
            raise ValueError("prediction count differs from stream length")
        if not np.all(np.isfinite(predictions)):
            raise ValueError(f"{name} produced non-finite predictions")
        results.append(
            ModelComparison(
                model=name,
                seed=stream.config.seed,
                predictions=predictions,
                windows=_score_windows(
                    predictions,
                    stream.clean_targets,
                    stream.config.drift_at,
                    score_window,
                ),
            )
        )
    return tuple(results)


def _reservoir_config(
    seed: int,
    override: ReservoirConfig | None,
) -> ReservoirConfig:
    if override is not None:
        return replace(override, seed=seed)
    return ReservoirConfig(
        input_dim=2,
        n_cells=96,
        topology="ring_shortcuts",
        feature_mode="multi_raw",
        seed=seed,
        dtype="float64",
        readout=ReadoutConfig(name="sliding_ridge", update_interval=1),
    )


def _score_windows(
    predictions: tuple[float, ...],
    clean_targets: np.ndarray,
    drift_at: int,
    width: int,
) -> tuple[WindowScore, ...]:
    length = len(predictions)
    boundaries = (
        ("pre", drift_at - width, drift_at),
        ("early_post", drift_at, drift_at + width),
        ("late_post", length - width, length),
    )
    scores: list[WindowScore] = []
    for name, start, stop in boundaries:
        errors = np.asarray(predictions[start:stop]) - clean_targets[start:stop]
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        if not np.isfinite(mae) or not np.isfinite(rmse):
            raise ValueError(f"non-finite score in window {name}")
        scores.append(WindowScore(name, start, stop, mae, rmse))
    return tuple(scores)


def _validate_window(n_samples: int, drift_at: int, width: int) -> None:
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("score_window must be a positive integer")
    if drift_at < width:
        raise ValueError("score_window exceeds the available pre-drift samples")
    if n_samples - drift_at < 2 * width:
        raise ValueError("two disjoint post-drift windows must fit")


def _validate_seeds(seeds: tuple[int, ...]) -> None:
    if not isinstance(seeds, tuple) or not seeds:
        raise ValueError("seeds must be a nonempty tuple")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")


def _validate_reservoir_config(config: ReservoirConfig | None) -> None:
    if config is None:
        return
    if not isinstance(config, ReservoirConfig):
        raise TypeError("reservoir_config must be a ReservoirConfig")
    if config.input_dim != 2:
        raise ValueError("reservoir_config.input_dim must equal 2")


__all__ = [
    "DEFAULT_SEEDS",
    "MODEL_NAMES",
    "WINDOW_NAMES",
    "WindowScore",
    "ModelComparison",
    "run_tdnr_comparison",
    "compare_tdnr_stream",
]
