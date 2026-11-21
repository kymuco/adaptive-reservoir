"""TDNR-01 Stage 3B paired bootstrap and partition tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from adaptive_reservoir import ReadoutConfig, ReservoirConfig
from adaptive_reservoir.benchmarks.tdnr_compare import MODEL_NAMES, WINDOW_NAMES
from adaptive_reservoir.benchmarks.tdnr_noise_sweep import (
    NOISE_AXES,
    PairedNoiseScore,
    run_tdnr_noise_sweep,
)
from adaptive_reservoir.benchmarks.tdnr_statistics import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEVELOPMENT_SEEDS,
    HOLDOUT_SEEDS,
    NoiseAggregate,
    summarize_paired_noise,
)
from adaptive_reservoir.benchmarks.temporal_drift_noise import (
    TemporalDriftNoiseConfig,
)


def _rows(
    *,
    seeds=(0, 1, 2),
    levels=(0.0, 0.2),
    axes=("input",),
    models=("adaptive_reservoir",),
    windows=("pre",),
):
    result = []
    for axis in axes:
        for level in levels:
            for seed in seeds:
                for model in models:
                    for window in windows:
                        base_mae = 1.0 + 0.01 * (seed % 17)
                        base_mae += 0.05 * MODEL_NAMES.index(model)
                        base_rmse = base_mae + 0.5
                        axis_factor = 2.0 if axis == "target" else 1.0
                        delta = level * (seed % 5 - 2) * 0.2 * axis_factor
                        result.append(
                            PairedNoiseScore(
                                axis=axis,
                                level=level,
                                seed=seed,
                                model=model,
                                window=window,
                                mae=base_mae + delta,
                                rmse=base_rmse + 1.5 * delta,
                                baseline_mae=base_mae,
                                baseline_rmse=base_rmse,
                                delta_mae=delta,
                                delta_rmse=1.5 * delta,
                            )
                        )
    return tuple(result)


def _summarize(rows, **kwargs):
    return summarize_paired_noise(
        rows, bootstrap_replicates=200, bootstrap_seed=11, **kwargs
    )


def test_constants_and_frozen_result() -> None:
    assert tuple(range(10)) == DEVELOPMENT_SEEDS
    assert tuple(range(100, 130)) == HOLDOUT_SEEDS
    assert DEFAULT_BOOTSTRAP_REPLICATES == 2000
    assert NoiseAggregate.__dataclass_params__.frozen


def test_exact_paired_mean_and_percentile_bootstrap() -> None:
    rows = _rows()
    summaries = _summarize(rows)
    assert len(summaries) == 2
    zero, noisy = summaries
    assert zero.level == 0.0
    assert zero.mean_delta_mae == 0.0
    assert zero.delta_mae_ci_low == 0.0
    assert zero.delta_mae_ci_high == 0.0
    assert zero.mean_delta_rmse == 0.0

    selected = [row for row in rows if row.level == 0.2]
    mae = np.asarray([row.mae for row in selected])
    rmse = np.asarray([row.rmse for row in selected])
    delta_mae = np.asarray([row.delta_mae for row in selected])
    delta_rmse = np.asarray([row.delta_rmse for row in selected])
    indices = np.random.default_rng(11).integers(0, 3, size=(200, 3))
    expected_mae_bounds = np.quantile(
        delta_mae[indices].mean(axis=1), (0.025, 0.975)
    )
    expected_rmse_bounds = np.quantile(
        delta_rmse[indices].mean(axis=1), (0.025, 0.975)
    )

    assert noisy.n_seeds == 3
    assert noisy.confidence == 0.95
    assert noisy.mean_mae == pytest.approx(mae.mean())
    assert noisy.mean_rmse == pytest.approx(rmse.mean())
    assert noisy.mean_delta_mae == pytest.approx(delta_mae.mean())
    assert noisy.mean_delta_rmse == pytest.approx(delta_rmse.mean())
    np.testing.assert_allclose(
        (noisy.delta_mae_ci_low, noisy.delta_mae_ci_high),
        expected_mae_bounds,
    )
    np.testing.assert_allclose(
        (noisy.delta_rmse_ci_low, noisy.delta_rmse_ci_high),
        expected_rmse_bounds,
    )


def test_row_order_and_seed_order_do_not_change_results() -> None:
    rows = _rows(seeds=(7, 2, 5, 1), axes=NOISE_AXES)
    assert _summarize(rows) == _summarize(reversed(rows))


def test_canonical_group_order_and_complete_grid() -> None:
    rows = _rows(axes=NOISE_AXES, models=MODEL_NAMES, windows=WINDOW_NAMES)
    summaries = _summarize(reversed(rows))
    assert len(summaries) == 2 * 2 * len(MODEL_NAMES) * len(WINDOW_NAMES)
    assert len({(r.axis, r.level, r.model, r.window) for r in summaries}) == len(
        summaries
    )
    assert [(r.axis, r.level) for r in summaries[::24]] == [
        ("input", 0.0),
        ("input", 0.2),
        ("target", 0.0),
        ("target", 0.2),
    ]
    assert all(r.n_seeds == 3 for r in summaries)


def test_development_and_holdout_partitions() -> None:
    development = _summarize(
        _rows(seeds=DEVELOPMENT_SEEDS), partition="development"
    )
    holdout = _summarize(_rows(seeds=HOLDOUT_SEEDS), partition="holdout")
    assert all(row.partition == "development" for row in development)
    assert all(row.n_seeds == 10 for row in development)
    assert all(row.partition == "holdout" for row in holdout)
    assert all(row.n_seeds == 30 for row in holdout)

    with pytest.raises(ValueError, match="development partition"):
        _summarize(_rows(seeds=(0, 1)), partition="development")
    with pytest.raises(ValueError, match="holdout partition"):
        _summarize(_rows(seeds=(100, 101)), partition="holdout")
    with pytest.raises(ValueError, match="holdout partition"):
        _summarize(_rows(seeds=DEVELOPMENT_SEEDS), partition="holdout")


def test_duplicate_rows_are_rejected() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="duplicate"):
        _summarize(rows + (rows[0],))


def test_missing_seed_is_rejected() -> None:
    rows = tuple(
        row for row in _rows()
        if not (row.level == 0.2 and row.seed == 2)
    )
    with pytest.raises(ValueError, match="identical seed sets"):
        _summarize(rows)


def test_missing_zero_noise_reference_is_rejected() -> None:
    rows = tuple(row for row in _rows() if row.level != 0.0)
    with pytest.raises(ValueError, match="zero-noise reference"):
        _summarize(rows)


def test_inconsistent_reference_is_rejected() -> None:
    rows = list(_rows())
    index = next(i for i, row in enumerate(rows) if row.level == 0.2)
    rows[index] = replace(
        rows[index],
        mae=rows[index].mae + 1.0,
        baseline_mae=rows[index].baseline_mae + 1.0,
    )
    with pytest.raises(ValueError, match="baseline_mae"):
        _summarize(rows)


def test_incorrect_delta_is_rejected() -> None:
    rows = list(_rows())
    rows[-1] = replace(rows[-1], delta_mae=rows[-1].delta_mae + 1.0)
    with pytest.raises(ValueError, match="delta_mae"):
        _summarize(rows)


def test_nonfinite_scores_are_rejected() -> None:
    rows = list(_rows())
    rows[-1] = replace(rows[-1], mae=float("nan"))
    with pytest.raises(ValueError, match="mae"):
        _summarize(rows)


def test_single_seed_and_empty_input_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        _summarize(())
    with pytest.raises(ValueError, match="at least two"):
        _summarize(_rows(seeds=(0,)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence": 0.0},
        {"confidence": 1.0},
        {"confidence": float("nan")},
        {"confidence": True},
        {"bootstrap_replicates": 0},
        {"bootstrap_replicates": 99},
        {"bootstrap_replicates": True},
        {"bootstrap_replicates": 20001},
        {"bootstrap_seed": -1},
        {"bootstrap_seed": True},
        {"partition": "unknown"},
    ],
)
def test_invalid_options_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        summarize_paired_noise(_rows(), **kwargs)


def test_noise_sweep_integration_with_small_stream() -> None:
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
    summaries = _summarize(rows)
    assert len(summaries) == 2 * len(MODEL_NAMES) * len(WINDOW_NAMES)
    assert all(row.n_seeds == 2 for row in summaries)
    for summary in summaries:
        matching = [
            row for row in rows
            if (row.axis, row.level, row.model, row.window)
            == (summary.axis, summary.level, summary.model, summary.window)
        ]
        assert summary.mean_delta_mae == pytest.approx(
            np.mean([row.delta_mae for row in matching])
        )
        assert summary.mean_delta_rmse == pytest.approx(
            np.mean([row.delta_rmse for row in matching])
        )


def test_bootstrap_does_not_modify_global_numpy_rng() -> None:
    before = np.random.get_state()
    _summarize(_rows())
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
