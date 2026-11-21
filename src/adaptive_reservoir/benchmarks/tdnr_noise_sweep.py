"""Paired noise-robustness experiments for TDNR-01 Stage 3A.

Vary input and target noise independently while holding dataset seeds,
latent realizations, target functions, model configurations, and scoring
windows fixed. Reuse the causal Stage 2B comparison without changing its
model training or scoring behavior.

The sweep produces per-seed results, not statistical significance claims.
No experiment runs at import time. Full-scale sweeps may be expensive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Real

from adaptive_reservoir import ReservoirConfig
from adaptive_reservoir.benchmarks.tdnr_compare import (
    DEFAULT_SEEDS,
    ModelComparison,
    WindowScore,
    run_tdnr_comparison,
)
from adaptive_reservoir.benchmarks.temporal_drift_noise import (
    TemporalDriftNoiseConfig,
)

DEFAULT_NOISE_LEVELS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)
NOISE_AXES = ("input", "target")


@dataclass(frozen=True, slots=True)
class PairedNoiseScore:
    """One model/seed/window score paired with its zero-noise reference.

    delta_mae and delta_rmse are noisy minus zero-noise errors. Positive
    deltas indicate degradation; negative deltas indicate improvement.
    These are absolute error differences, not percentages.
    """

    axis: str
    level: float
    seed: int
    model: str
    window: str
    mae: float
    rmse: float
    baseline_mae: float
    baseline_rmse: float
    delta_mae: float
    delta_rmse: float


def run_tdnr_noise_sweep(
    config: TemporalDriftNoiseConfig | None = None,
    *,
    levels: tuple[float, ...] = DEFAULT_NOISE_LEVELS,
    axes: tuple[str, ...] = NOISE_AXES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    score_window: int = 128,
    reservoir_config: ReservoirConfig | None = None,
) -> tuple[PairedNoiseScore, ...]:
    """Return deterministic, paired, prequential noise comparisons.

    A zero-noise abrupt stream is the shared reference for both axes.
    For the input axis, only input_noise changes; for the target axis,
    only target_noise changes. The other channel stays at zero.

    Each level uses the same explicit seeds and independently initialized
    models. The Stage 1 generator uses component-local random streams,
    preserving the latent realization and clean targets across levels.

    Models are trained using training_targets and scored against
    clean_targets by Stage 2B. This function only reads its completed
    WindowScore values; it never supplies evaluation-only data to models.

    Output order is axis, level, seed, model, window. Zero-noise model
    runs are computed once and reused as the paired reference for every
    axis. The full default experiment is intentionally not a CI smoke test.
    """
    if config is None:
        config = TemporalDriftNoiseConfig()
    if not isinstance(config, TemporalDriftNoiseConfig):
        raise TypeError("config must be a TemporalDriftNoiseConfig")
    if config.scenario != "abrupt":
        raise ValueError("noise sweep requires an abrupt scenario")
    if config.input_noise != 0.0 or config.target_noise != 0.0:
        raise ValueError("base config must have zero input_noise and target_noise")

    checked_levels = _validate_levels(levels)
    _validate_axes(axes)

    reference_results = run_tdnr_comparison(
        config,
        seeds=seeds,
        score_window=score_window,
        reservoir_config=reservoir_config,
    )
    reference_scores = _index_scores(reference_results)

    rows: list[PairedNoiseScore] = []
    for axis in axes:
        for level in checked_levels:
            if level == 0.0:
                comparisons = reference_results
            else:
                variant = replace(
                    config,
                    input_noise=level if axis == "input" else 0.0,
                    target_noise=level if axis == "target" else 0.0,
                )
                comparisons = run_tdnr_comparison(
                    variant,
                    seeds=seeds,
                    score_window=score_window,
                    reservoir_config=reservoir_config,
                )

            for comparison in comparisons:
                for score in comparison.windows:
                    key = (comparison.seed, comparison.model, score.name)
                    reference = reference_scores[key]
                    rows.append(
                        PairedNoiseScore(
                            axis=axis,
                            level=level,
                            seed=comparison.seed,
                            model=comparison.model,
                            window=score.name,
                            mae=score.mae,
                            rmse=score.rmse,
                            baseline_mae=reference.mae,
                            baseline_rmse=reference.rmse,
                            delta_mae=score.mae - reference.mae,
                            delta_rmse=score.rmse - reference.rmse,
                        )
                    )

    return tuple(rows)


def _index_scores(
    comparisons: tuple[ModelComparison, ...],
) -> dict[tuple[int, str, str], WindowScore]:
    return {
        (comparison.seed, comparison.model, score.name): score
        for comparison in comparisons
        for score in comparison.windows
    }


def _validate_levels(levels: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(levels, tuple) or not levels:
        raise ValueError("levels must be a nonempty tuple")

    checked: list[float] = []
    for level in levels:
        if isinstance(level, bool) or not isinstance(level, Real):
            raise ValueError("levels must contain finite non-negative numbers")
        try:
            numeric = float(level)
        except OverflowError as exc:
            raise ValueError("levels must be finite") from exc
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError("levels must contain finite non-negative numbers")
        if checked and numeric <= checked[-1]:
            raise ValueError("levels must be strictly increasing and unique")
        checked.append(numeric)

    if checked[0] != 0.0:
        raise ValueError("levels must begin with the zero-noise reference")
    return tuple(checked)


def _validate_axes(axes: tuple[str, ...]) -> None:
    if not isinstance(axes, tuple) or not axes:
        raise ValueError("axes must be a nonempty tuple")
    if any(axis not in NOISE_AXES for axis in axes):
        raise ValueError("axes must contain only 'input' and 'target'")
    if len(set(axes)) != len(axes):
        raise ValueError("axes must be unique")


__all__ = [
    "DEFAULT_NOISE_LEVELS",
    "NOISE_AXES",
    "PairedNoiseScore",
    "run_tdnr_noise_sweep",
]
