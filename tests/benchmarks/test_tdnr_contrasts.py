"""TDNR-01 Stage 3C paired baseline contrast tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from adaptive_reservoir import ReadoutConfig, ReservoirConfig
from adaptive_reservoir.benchmarks.tdnr_compare import MODEL_NAMES, WINDOW_NAMES
from adaptive_reservoir.benchmarks.tdnr_contrasts import (
    ADAPTIVE_MODEL,
    BASELINE_MODELS,
    BaselineContrast,
    summarize_baseline_contrasts,
)
from adaptive_reservoir.benchmarks.tdnr_noise_sweep import (
    NOISE_AXES,
    PairedNoiseScore,
    run_tdnr_noise_sweep,
)
from adaptive_reservoir.benchmarks.tdnr_statistics import (
    DEVELOPMENT_SEEDS,
    HOLDOUT_SEEDS,
)
from adaptive_reservoir.benchmarks.temporal_drift_noise import (
    TemporalDriftNoiseConfig,
)


def _rows(
    *,
    seeds=(0, 1, 2),
    axes=("input",),
    levels=(0.0, 0.2),
):
    rows = []
    for axis in axes:
        for level in levels:
            for seed in seeds:
                for model in MODEL_NAMES:
                    for window in WINDOW_NAMES:
                        model_index = MODEL_NAMES.index(model)
                        window_index = WINDOW_NAMES.index(window)
                        base_mae = 1.0 + 0.1 * model_index
                        base_mae += 0.02 * window_index + 0.01 * (seed % 17)
                        base_rmse = base_mae + 0.5
                        axis_factor = 2.0 if axis == "target" else 1.0
                        change = (
                            level * (model_index + 1) * (seed % 5 + 1)
                            * 0.03 * axis_factor
                        )
                        mae = base_mae + change
                        rmse = base_rmse + 1.5 * change
                        rows.append(
                            PairedNoiseScore(
                                axis=axis,
                                level=level,
                                seed=seed,
                                model=model,
                                window=window,
                                mae=mae,
                                rmse=rmse,
                                baseline_mae=base_mae,
                                baseline_rmse=base_rmse,
                                delta_mae=mae - base_mae,
                                delta_rmse=rmse - base_rmse,
                            )
                        )
    return tuple(rows)


def _summarize(rows, **kwargs):
    return summarize_baseline_contrasts(
        rows, bootstrap_replicates=200, bootstrap_seed=7, **kwargs
    )


def test_constants_and_frozen_result() -> None:
    assert ADAPTIVE_MODEL in MODEL_NAMES
    assert tuple(name for name in MODEL_NAMES if name != ADAPTIVE_MODEL) == BASELINE_MODELS
    assert len(BASELINE_MODELS) == 3
    assert BaselineContrast.__dataclass_params__.frozen


def test_exact_seed_paired_bootstrap() -> None:
    rows = _rows()
    results = _summarize(rows)
    selected = next(
        result for result in results
        if result.axis == "input"
        and result.level == 0.2
        and result.baseline == "zero"
        and result.window == "pre"
    )

    indexed = {
        (row.model, row.seed): row
        for row in rows if row.level == 0.2 and row.window == "pre"
    }
    mae_differences = np.asarray([
        indexed[("zero", seed)].mae - indexed[(ADAPTIVE_MODEL, seed)].mae
        for seed in (0, 1, 2)
    ])
    noise_differences = np.asarray([
        indexed[("zero", seed)].delta_mae
        - indexed[(ADAPTIVE_MODEL, seed)].delta_mae
        for seed in (0, 1, 2)
    ])
    indices = np.random.default_rng(7).integers(0, 3, size=(200, 3))

    assert selected.n_seeds == 3
    assert selected.confidence == 0.95
    assert selected.mean_mae_advantage == pytest.approx(mae_differences.mean())
    assert selected.mean_noise_mae_advantage == pytest.approx(
        noise_differences.mean()
    )
    np.testing.assert_allclose(
        (selected.mae_ci_low, selected.mae_ci_high),
        np.quantile(mae_differences[indices].mean(axis=1), (0.025, 0.975)),
    )
    np.testing.assert_allclose(
        (selected.noise_mae_ci_low, selected.noise_mae_ci_high),
        np.quantile(noise_differences[indices].mean(axis=1), (0.025, 0.975)),
    )


def test_zero_noise_has_zero_degradation_contrast() -> None:
    results = _summarize(_rows())
    zero = [result for result in results if result.level == 0.0]
    assert zero
    for result in zero:
        assert result.mean_noise_mae_advantage == 0.0
        assert result.noise_mae_ci_low == 0.0
        assert result.noise_mae_ci_high == 0.0
        assert result.mean_noise_rmse_advantage == 0.0
        assert result.noise_rmse_ci_low == 0.0
        assert result.noise_rmse_ci_high == 0.0


def test_complete_grid_and_canonical_order() -> None:
    results = _summarize(_rows(axes=NOISE_AXES))
    assert len(results) == 2 * 2 * len(BASELINE_MODELS) * len(WINDOW_NAMES)
    assert [(result.axis, result.level) for result in results[::9]] == [
        ("input", 0.0),
        ("input", 0.2),
        ("target", 0.0),
        ("target", 0.2),
    ]
    assert [result.baseline for result in results[:9:3]] == list(BASELINE_MODELS)
    assert [result.window for result in results[:3]] == list(WINDOW_NAMES)


def test_input_order_does_not_change_results() -> None:
    rows = _rows(axes=NOISE_AXES, seeds=(7, 2, 5, 1))
    assert _summarize(rows) == _summarize(reversed(rows))


def test_development_and_holdout_partitions_remain_separate() -> None:
    development = _summarize(
        _rows(seeds=DEVELOPMENT_SEEDS), partition="development"
    )
    holdout = _summarize(_rows(seeds=HOLDOUT_SEEDS), partition="holdout")
    assert all(result.n_seeds == 10 for result in development)
    assert all(result.partition == "development" for result in development)
    assert all(result.n_seeds == 30 for result in holdout)
    assert all(result.partition == "holdout" for result in holdout)

    with pytest.raises(ValueError, match="holdout partition"):
        _summarize(_rows(seeds=DEVELOPMENT_SEEDS), partition="holdout")


def test_missing_model_group_is_rejected() -> None:
    rows = tuple(
        row for row in _rows()
        if not (
            row.level == 0.2
            and row.model == "persistence"
            and row.window == "pre"
        )
    )
    with pytest.raises(ValueError, match="incomplete model/window/seed grid"):
        _summarize(rows)


def test_missing_seed_pair_is_rejected() -> None:
    rows = tuple(
        row for row in _rows()
        if not (
            row.level == 0.2
            and row.model == "zero"
            and row.window == "pre"
            and row.seed == 2
        )
    )
    with pytest.raises(ValueError, match="identical seed sets"):
        _summarize(rows)


def test_different_noise_levels_across_axes_are_rejected() -> None:
    input_rows = _rows(axes=("input",))
    target_rows = _rows(axes=("target",), levels=(0.0,))
    with pytest.raises(ValueError, match="identical levels"):
        _summarize(input_rows + target_rows)


def test_duplicate_and_inconsistent_reference_are_rejected() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="duplicate"):
        _summarize(rows + (rows[0],))

    modified = list(rows)
    index = next(i for i, row in enumerate(modified) if row.level == 0.2)
    modified[index] = replace(
        modified[index],
        mae=modified[index].mae + 1.0,
        baseline_mae=modified[index].baseline_mae + 1.0,
    )
    with pytest.raises(ValueError, match="baseline_mae"):
        _summarize(modified)


def test_empty_and_invalid_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        _summarize(())
    with pytest.raises(ValueError, match="confidence"):
        _summarize(_rows(), confidence=1.0)
    with pytest.raises(ValueError, match="holdout partition"):
        _summarize(_rows(), partition="holdout")


def test_noise_sweep_integration() -> None:
    config = TemporalDriftNoiseConfig(
        n_samples=48,
        drift_at=24,
        delay_before=3,
        delay_after=12,
        scenario="abrupt",
        input_noise=0.0,
        target_noise=0.0,
    )
    reservoir = ReservoirConfig(
        input_dim=2,
        n_cells=6,
        topology="ring_shortcuts",
        feature_mode="state_raw",
        readout=ReadoutConfig(name="sliding_ridge", window_size=12),
    )
    rows = run_tdnr_noise_sweep(
        config,
        levels=(0.0, 0.2),
        axes=("input",),
        seeds=(1, 2),
        score_window=6,
        reservoir_config=reservoir,
    )
    results = _summarize(rows)
    assert len(results) == 2 * len(BASELINE_MODELS) * len(WINDOW_NAMES)
    assert all(result.n_seeds == 2 for result in results)

    for result in results:
        baseline = [
            row for row in rows
            if row.axis == result.axis
            and row.level == result.level
            and row.model == result.baseline
            and row.window == result.window
        ]
        adaptive = [
            row for row in rows
            if row.axis == result.axis
            and row.level == result.level
            and row.model == ADAPTIVE_MODEL
            and row.window == result.window
        ]
        baseline.sort(key=lambda row: row.seed)
        adaptive.sort(key=lambda row: row.seed)
        expected = np.mean([
            b.mae - a.mae for a, b in zip(adaptive, baseline, strict=True)
        ])
        assert result.mean_mae_advantage == pytest.approx(expected)


def test_global_numpy_rng_is_unchanged() -> None:
    before = np.random.get_state()
    _summarize(_rows())
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
