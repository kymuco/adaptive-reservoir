"""Offline paired-seed statistics for TDNR-01 Stage 3B.

Aggregate Stage 3A noise-sweep results without rerunning models.
Bootstrap whole seed pairs, not individual time steps. Keep development
and holdout seed partitions separate. Confidence intervals describe
variation across synthetic seeds, not real-world generalization.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

import numpy as np

from adaptive_reservoir.benchmarks.tdnr_compare import MODEL_NAMES, WINDOW_NAMES
from adaptive_reservoir.benchmarks.tdnr_noise_sweep import (
    NOISE_AXES,
    PairedNoiseScore,
)

DEVELOPMENT_SEEDS = tuple(range(10))
HOLDOUT_SEEDS = tuple(range(100, 130))
DEFAULT_BOOTSTRAP_REPLICATES = 2000


@dataclass(frozen=True, slots=True)
class NoiseAggregate:
    """Across-seed summary for one axis, level, model and score window.

    Deltas are noisy minus zero-noise errors, paired within each seed.
    Positive deltas indicate degradation. Confidence bounds are
    percentile bootstrap intervals for the mean paired delta.
    """

    partition: str
    axis: str
    level: float
    model: str
    window: str
    n_seeds: int
    confidence: float
    mean_mae: float
    mean_rmse: float
    mean_delta_mae: float
    delta_mae_ci_low: float
    delta_mae_ci_high: float
    mean_delta_rmse: float
    delta_rmse_ci_low: float
    delta_rmse_ci_high: float


def summarize_paired_noise(
    rows: Iterable[PairedNoiseScore],
    *,
    partition: str = "exploratory",
    confidence: float = 0.95,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = 0,
) -> tuple[NoiseAggregate, ...]:
    """Summarize complete Stage 3A groups using paired seed bootstrap.

    Every group must contain the same set of at least two seeds and a
    corresponding zero-noise reference for each axis/model/window.
    Duplicate rows, missing seeds, inconsistent references and invalid
    scores are rejected rather than silently averaged.

    partition='development' requires exactly seeds 0..9;
    partition='holdout' requires exactly seeds 100..129.
    'exploratory' accepts any common seed set with at least two seeds.
    These labels do not select or run experiments.

    One bootstrap index matrix is shared across all groups, preserving
    their common seed ordering. A local NumPy generator is used, so the
    global RNG is untouched. Results are independent of input row order.
    """
    _validate_options(partition, confidence, bootstrap_replicates, bootstrap_seed)
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("rows must not be empty")

    groups: dict[tuple[str, float, str, str], dict[int, PairedNoiseScore]] = {}
    for row in materialized:
        _validate_row(row)
        key = (row.axis, float(row.level), row.model, row.window)
        members = groups.setdefault(key, {})
        if row.seed in members:
            raise ValueError("duplicate axis/level/model/window/seed row")
        members[row.seed] = row

    seed_sets = {frozenset(members) for members in groups.values()}
    if len(seed_sets) != 1:
        raise ValueError("all groups must contain identical seed sets")
    seeds = tuple(sorted(next(iter(seed_sets))))
    if len(seeds) < 2:
        raise ValueError("at least two paired seeds are required")
    if partition == "development" and seeds != DEVELOPMENT_SEEDS:
        raise ValueError("development partition requires exactly seeds 0..9")
    if partition == "holdout" and seeds != HOLDOUT_SEEDS:
        raise ValueError("holdout partition requires exactly seeds 100..129")

    for (axis, level, model, window), members in groups.items():
        reference = groups.get((axis, 0.0, model, window))
        if reference is None:
            raise ValueError("each group requires a zero-noise reference")
        for seed in seeds:
            row = members[seed]
            zero = reference[seed]
            if not _close(row.baseline_mae, zero.mae):
                raise ValueError("baseline_mae differs from zero-noise reference")
            if not _close(row.baseline_rmse, zero.rmse):
                raise ValueError("baseline_rmse differs from zero-noise reference")
            if level == 0.0 and (
                not _close(row.delta_mae, 0.0)
                or not _close(row.delta_rmse, 0.0)
            ):
                raise ValueError("zero-noise paired deltas must be zero")

    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0, len(seeds), size=(bootstrap_replicates, len(seeds))
    )
    tail = (1.0 - float(confidence)) / 2.0

    ordered_keys = sorted(
        groups,
        key=lambda key: (
            NOISE_AXES.index(key[0]),
            key[1],
            MODEL_NAMES.index(key[2]),
            WINDOW_NAMES.index(key[3]),
        ),
    )

    summaries: list[NoiseAggregate] = []
    for axis, level, model, window in ordered_keys:
        members = groups[(axis, level, model, window)]
        ordered = [members[seed] for seed in seeds]
        mae = np.asarray([row.mae for row in ordered], dtype=np.float64)
        rmse = np.asarray([row.rmse for row in ordered], dtype=np.float64)
        delta_mae = np.asarray(
            [row.delta_mae for row in ordered], dtype=np.float64
        )
        delta_rmse = np.asarray(
            [row.delta_rmse for row in ordered], dtype=np.float64
        )

        mae_bounds = np.quantile(
            np.mean(delta_mae[indices], axis=1),
            (tail, 1.0 - tail),
        )
        rmse_bounds = np.quantile(
            np.mean(delta_rmse[indices], axis=1),
            (tail, 1.0 - tail),
        )
        values = (
            float(np.mean(mae)),
            float(np.mean(rmse)),
            float(np.mean(delta_mae)),
            float(mae_bounds[0]),
            float(mae_bounds[1]),
            float(np.mean(delta_rmse)),
            float(rmse_bounds[0]),
            float(rmse_bounds[1]),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("non-finite aggregate or bootstrap interval")

        summaries.append(
            NoiseAggregate(
                partition=partition,
                axis=axis,
                level=level,
                model=model,
                window=window,
                n_seeds=len(seeds),
                confidence=float(confidence),
                mean_mae=values[0],
                mean_rmse=values[1],
                mean_delta_mae=values[2],
                delta_mae_ci_low=values[3],
                delta_mae_ci_high=values[4],
                mean_delta_rmse=values[5],
                delta_rmse_ci_low=values[6],
                delta_rmse_ci_high=values[7],
            )
        )
    return tuple(summaries)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-10)


def _validate_row(row: PairedNoiseScore) -> None:
    if not isinstance(row, PairedNoiseScore):
        raise ValueError("rows must contain PairedNoiseScore instances")
    if row.axis not in NOISE_AXES:
        raise ValueError("invalid noise axis")
    if row.model not in MODEL_NAMES:
        raise ValueError("invalid model name")
    if row.window not in WINDOW_NAMES:
        raise ValueError("invalid scoring window")
    if isinstance(row.seed, bool) or not isinstance(row.seed, int) or row.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    level = _finite_real("level", row.level)
    if level < 0.0:
        raise ValueError("level must be non-negative")

    for name in (
        "mae", "rmse", "baseline_mae", "baseline_rmse",
        "delta_mae", "delta_rmse",
    ):
        value = _finite_real(name, getattr(row, name))
        if name in ("mae", "rmse", "baseline_mae", "baseline_rmse") and value < 0.0:
            raise ValueError(f"{name} must be non-negative")

    if not _close(row.delta_mae, row.mae - row.baseline_mae):
        raise ValueError("delta_mae does not match paired score difference")
    if not _close(row.delta_rmse, row.rmse - row.baseline_rmse):
        raise ValueError("delta_rmse does not match paired score difference")


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _validate_options(
    partition: str,
    confidence: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> None:
    if partition not in ("exploratory", "development", "holdout"):
        raise ValueError("invalid partition")
    numeric_confidence = _finite_real("confidence", confidence)
    if not 0.0 < numeric_confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or not 100 <= bootstrap_replicates <= 20000
    ):
        raise ValueError("bootstrap_replicates must be an integer in [100, 20000]")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a non-negative integer")


__all__ = [
    "DEVELOPMENT_SEEDS",
    "HOLDOUT_SEEDS",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "NoiseAggregate",
    "summarize_paired_noise",
]
