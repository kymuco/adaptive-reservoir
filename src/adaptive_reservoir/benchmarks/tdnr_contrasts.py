"""Offline paired model contrasts for TDNR-01 Stage 3C.

Compare AdaptiveReservoir with each causal baseline using the same
synthetic seeds, noise conditions and scoring windows.

Positive error advantage means the adaptive model has lower error.
Positive noise advantage means it degrades less relative to zero noise.
These are different estimands and must not be conflated.

Bootstrap intervals are descriptive, unadjusted for multiple comparisons,
and do not establish real-world superiority or statistical significance.
No experiments run at import time.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from adaptive_reservoir.benchmarks.tdnr_compare import MODEL_NAMES, WINDOW_NAMES
from adaptive_reservoir.benchmarks.tdnr_noise_sweep import NOISE_AXES, PairedNoiseScore
from adaptive_reservoir.benchmarks.tdnr_statistics import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    summarize_paired_noise,
)

ADAPTIVE_MODEL = "adaptive_reservoir"
BASELINE_MODELS = tuple(name for name in MODEL_NAMES if name != ADAPTIVE_MODEL)


@dataclass(frozen=True, slots=True)
class BaselineContrast:
    """Seed-paired adaptive advantage against one baseline.

    Error advantage is baseline error minus adaptive error.
    Noise advantage is baseline degradation minus adaptive degradation.
    Positive values favor the adaptive model for the specified metric.
    """

    partition: str
    axis: str
    level: float
    baseline: str
    window: str
    n_seeds: int
    confidence: float
    mean_mae_advantage: float
    mae_ci_low: float
    mae_ci_high: float
    mean_rmse_advantage: float
    rmse_ci_low: float
    rmse_ci_high: float
    mean_noise_mae_advantage: float
    noise_mae_ci_low: float
    noise_mae_ci_high: float
    mean_noise_rmse_advantage: float
    noise_rmse_ci_low: float
    noise_rmse_ci_high: float


def summarize_baseline_contrasts(
    rows: Iterable[PairedNoiseScore],
    *,
    partition: str = "exploratory",
    confidence: float = 0.95,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = 0,
) -> tuple[BaselineContrast, ...]:
    """Calculate seed-paired model contrasts from Stage 3A scores.

    Stage 3B validates scores, references, seed sets and partition rules.
    This stage additionally requires every model and scoring window for
    every supplied axis/level/seed. All supplied axes must use the same
    noise levels. A single axis is permitted for exploratory analysis.

    Whole seeds are resampled with one shared bootstrap index matrix.
    Development and holdout results are never pooled. The function does
    not select configurations, tune parameters or run model inference.
    """
    materialized = tuple(rows)
    summarize_paired_noise(
        materialized,
        partition=partition,
        confidence=confidence,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )

    seeds = tuple(sorted({row.seed for row in materialized}))
    indexed = {
        (row.axis, float(row.level), row.model, row.window, row.seed): row
        for row in materialized
    }
    levels_by_axis: dict[str, set[float]] = {}
    for row in materialized:
        levels_by_axis.setdefault(row.axis, set()).add(float(row.level))

    level_sets = {tuple(sorted(levels)) for levels in levels_by_axis.values()}
    if len(level_sets) != 1:
        raise ValueError("all noise axes must use identical levels")

    for axis, levels in levels_by_axis.items():
        for level in levels:
            for model in MODEL_NAMES:
                for window in WINDOW_NAMES:
                    if any(
                        (axis, level, model, window, seed) not in indexed
                        for seed in seeds
                    ):
                        raise ValueError("incomplete model/window/seed grid")

    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0, len(seeds), size=(bootstrap_replicates, len(seeds))
    )
    tail = (1.0 - float(confidence)) / 2.0

    results: list[BaselineContrast] = []
    for axis in NOISE_AXES:
        if axis not in levels_by_axis:
            continue
        for level in sorted(levels_by_axis[axis]):
            for baseline in BASELINE_MODELS:
                for window in WINDOW_NAMES:
                    adaptive = [
                        indexed[(axis, level, ADAPTIVE_MODEL, window, seed)]
                        for seed in seeds
                    ]
                    reference = [
                        indexed[(axis, level, baseline, window, seed)]
                        for seed in seeds
                    ]
                    pairs = tuple(zip(adaptive, reference, strict=True))

                    mae = _estimate(
                        [b.mae - a.mae for a, b in pairs], indices, tail
                    )
                    rmse = _estimate(
                        [b.rmse - a.rmse for a, b in pairs], indices, tail
                    )
                    noise_mae = _estimate(
                        [b.delta_mae - a.delta_mae for a, b in pairs],
                        indices,
                        tail,
                    )
                    noise_rmse = _estimate(
                        [b.delta_rmse - a.delta_rmse for a, b in pairs],
                        indices,
                        tail,
                    )

                    results.append(
                        BaselineContrast(
                            partition=partition,
                            axis=axis,
                            level=level,
                            baseline=baseline,
                            window=window,
                            n_seeds=len(seeds),
                            confidence=float(confidence),
                            mean_mae_advantage=mae[0],
                            mae_ci_low=mae[1],
                            mae_ci_high=mae[2],
                            mean_rmse_advantage=rmse[0],
                            rmse_ci_low=rmse[1],
                            rmse_ci_high=rmse[2],
                            mean_noise_mae_advantage=noise_mae[0],
                            noise_mae_ci_low=noise_mae[1],
                            noise_mae_ci_high=noise_mae[2],
                            mean_noise_rmse_advantage=noise_rmse[0],
                            noise_rmse_ci_low=noise_rmse[1],
                            noise_rmse_ci_high=noise_rmse[2],
                        )
                    )
    return tuple(results)


def _estimate(
    differences: list[float],
    indices: np.ndarray,
    tail: float,
) -> tuple[float, float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("paired contrast contains non-finite differences")

    bootstrap_means = np.mean(values[indices], axis=1)
    bounds = np.quantile(bootstrap_means, (tail, 1.0 - tail))
    result = (
        float(np.mean(values)),
        float(bounds[0]),
        float(bounds[1]),
    )
    if not all(math.isfinite(value) for value in result):
        raise ValueError("paired contrast contains non-finite aggregates")
    return result


__all__ = [
    "ADAPTIVE_MODEL",
    "BASELINE_MODELS",
    "BaselineContrast",
    "summarize_baseline_contrasts",
]
