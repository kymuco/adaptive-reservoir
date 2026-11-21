"""TDNR-01 Stage 3A paired noise-sweep contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from adaptive_reservoir import ReadoutConfig, ReservoirConfig
from adaptive_reservoir.benchmarks import tdnr_noise_sweep as sweep_module
from adaptive_reservoir.benchmarks.tdnr_compare import (
    MODEL_NAMES,
    WINDOW_NAMES,
    ModelComparison,
    WindowScore,
    run_tdnr_comparison,
)
from adaptive_reservoir.benchmarks.tdnr_noise_sweep import (
    DEFAULT_NOISE_LEVELS,
    NOISE_AXES,
    PairedNoiseScore,
    run_tdnr_noise_sweep,
)
from adaptive_reservoir.benchmarks.temporal_drift_noise import (
    TemporalDriftNoiseConfig,
    generate_temporal_drift_noise_stream,
)


def _config() -> TemporalDriftNoiseConfig:
    return TemporalDriftNoiseConfig(
        seed=17,
        n_samples=48,
        drift_at=24,
        delay_before=3,
        delay_after=12,
        input_noise=0.0,
        target_noise=0.0,
        scenario="abrupt",
    )


def _reservoir() -> ReservoirConfig:
    return ReservoirConfig(
        input_dim=2,
        n_cells=8,
        topology="ring_shortcuts",
        feature_mode="state_raw",
        seed=999,
        readout=ReadoutConfig(
            name="sliding_ridge",
            window_size=12,
            update_interval=1,
        ),
    )


def _run(
    *,
    levels=(0.0, 0.2),
    axes=("input",),
    seeds=(1,),
):
    return run_tdnr_noise_sweep(
        _config(),
        levels=levels,
        axes=axes,
        seeds=seeds,
        score_window=6,
        reservoir_config=_reservoir(),
    )


def _index(rows):
    return {
        (row.axis, row.level, row.seed, row.model, row.window): row
        for row in rows
    }


def test_public_constants_and_result_shape() -> None:
    assert DEFAULT_NOISE_LEVELS == (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)
    assert NOISE_AXES == ("input", "target")
    assert PairedNoiseScore.__dataclass_params__.frozen

    rows = _run(axes=NOISE_AXES, seeds=(1, 2))
    assert len(rows) == 2 * 2 * 2 * len(MODEL_NAMES) * len(WINDOW_NAMES)
    assert len(_index(rows)) == len(rows)
    assert [(row.axis, row.level) for row in rows[::24]] == [
        ("input", 0.0),
        ("input", 0.2),
        ("target", 0.0),
        ("target", 0.2),
    ]
    assert all(np.isfinite(row.mae) and np.isfinite(row.rmse) for row in rows)
    assert all(np.isfinite(row.delta_mae) and np.isfinite(row.delta_rmse) for row in rows)


def test_zero_noise_rows_are_shared_references() -> None:
    rows = _run(axes=NOISE_AXES)
    indexed = _index(rows)
    for model in MODEL_NAMES:
        for window in WINDOW_NAMES:
            input_row = indexed[("input", 0.0, 1, model, window)]
            target_row = indexed[("target", 0.0, 1, model, window)]
            assert input_row.mae == target_row.mae
            assert input_row.rmse == target_row.rmse
            assert input_row.mae == input_row.baseline_mae
            assert input_row.rmse == input_row.baseline_rmse
            assert input_row.delta_mae == 0.0
            assert input_row.delta_rmse == 0.0
            assert target_row.delta_mae == 0.0
            assert target_row.delta_rmse == 0.0


def test_paired_deltas_match_direct_stage2b_scores() -> None:
    config = _config()
    rows = _run(axes=("input",), seeds=(3,))
    reference = run_tdnr_comparison(
        config,
        seeds=(3,),
        score_window=6,
        reservoir_config=_reservoir(),
    )
    noisy = run_tdnr_comparison(
        replace(config, input_noise=0.2),
        seeds=(3,),
        score_window=6,
        reservoir_config=_reservoir(),
    )
    reference_index = {
        (item.model, score.name): score
        for item in reference
        for score in item.windows
    }
    noisy_index = {
        (item.model, score.name): score
        for item in noisy
        for score in item.windows
    }

    for row in rows:
        reference_score = reference_index[(row.model, row.window)]
        actual_score = (
            reference_score
            if row.level == 0.0
            else noisy_index[(row.model, row.window)]
        )
        assert row.seed == 3
        assert row.mae == actual_score.mae
        assert row.rmse == actual_score.rmse
        assert row.baseline_mae == reference_score.mae
        assert row.baseline_rmse == reference_score.rmse
        assert row.delta_mae == actual_score.mae - reference_score.mae
        assert row.delta_rmse == actual_score.rmse - reference_score.rmse


def test_repeated_sweeps_are_deterministic() -> None:
    first = _run()
    second = _run()
    assert first == second


def test_seed_order_changes_order_not_results() -> None:
    first = _run(seeds=(2, 5))
    second = _run(seeds=(5, 2))
    assert _index(first) == _index(second)


def test_noise_axes_change_only_the_selected_channel(monkeypatch) -> None:
    calls = []

    def fake_compare(config, *, seeds, score_window, reservoir_config):
        calls.append((config.input_noise, config.target_noise, seeds))
        return tuple(
            ModelComparison(
                model="zero",
                seed=seed,
                predictions=(0.0,),
                windows=(WindowScore("pre", 0, 1, 1.0, 1.0),),
            )
            for seed in seeds
        )

    monkeypatch.setattr(sweep_module, "run_tdnr_comparison", fake_compare)
    rows = _run(axes=NOISE_AXES, levels=(0.0, 0.25), seeds=(4,))

    assert calls == [
        (0.0, 0.0, (4,)),
        (0.25, 0.0, (4,)),
        (0.0, 0.25, (4,)),
    ]
    assert len(rows) == 4
    assert all(row.delta_mae == 0.0 for row in rows)


def test_stage1_clean_targets_are_invariant_to_noise_levels() -> None:
    config = _config()
    reference = generate_temporal_drift_noise_stream(config)
    noisy_input = generate_temporal_drift_noise_stream(
        replace(config, input_noise=0.4)
    )
    noisy_target = generate_temporal_drift_noise_stream(
        replace(config, target_noise=0.4)
    )

    for variant in (noisy_input, noisy_target):
        np.testing.assert_array_equal(reference.latent_signal, variant.latent_signal)
        np.testing.assert_array_equal(reference.clean_targets, variant.clean_targets)
        np.testing.assert_array_equal(reference.drift_weight, variant.drift_weight)

    np.testing.assert_array_equal(
        reference.training_targets,
        noisy_input.training_targets,
    )
    np.testing.assert_array_equal(reference.inputs, noisy_target.inputs)
    assert not np.array_equal(reference.inputs, noisy_input.inputs)
    assert not np.array_equal(
        reference.training_targets,
        noisy_target.training_targets,
    )


def test_zero_baseline_is_invariant_to_noise() -> None:
    rows = _run(axes=NOISE_AXES)
    for row in rows:
        if row.model == "zero":
            assert row.delta_mae == 0.0
            assert row.delta_rmse == 0.0


def test_invalid_base_noise_is_rejected() -> None:
    for changes in (
        {"input_noise": 0.1},
        {"target_noise": 0.1},
        {"input_noise": 0.1, "target_noise": 0.2},
    ):
        with pytest.raises(ValueError, match="zero"):
            run_tdnr_noise_sweep(
                replace(_config(), **changes),
                levels=(0.0,),
                seeds=(1,),
                score_window=6,
                reservoir_config=_reservoir(),
            )


def test_non_abrupt_scenario_is_rejected() -> None:
    with pytest.raises(ValueError, match="abrupt"):
        run_tdnr_noise_sweep(
            replace(_config(), scenario="stationary"),
            levels=(0.0,),
            seeds=(1,),
            score_window=6,
            reservoir_config=_reservoir(),
        )


@pytest.mark.parametrize(
    "levels",
    [
        (),
        (0.1,),
        (-0.1, 0.0),
        (0.0, -0.1),
        (0.0, 0.0),
        (0.0, 0.2, 0.1),
        (0.0, float("nan")),
        (0.0, float("inf")),
        (0.0, True),
        (0.0, "0.2"),
        [0.0, 0.2],
    ],
)
def test_invalid_noise_levels_are_rejected(levels) -> None:
    with pytest.raises(ValueError, match="levels"):
        _run(levels=levels)


@pytest.mark.parametrize(
    "axes",
    [(), ("both",), ("input", "input"), ("input", "other"), ["input"]],
)
def test_invalid_noise_axes_are_rejected(axes) -> None:
    with pytest.raises(ValueError, match="axes"):
        _run(axes=axes)


@pytest.mark.parametrize("seeds", [(), (1, 1), (-1,), (True,), [1, 2]])
def test_invalid_seeds_are_rejected(seeds) -> None:
    with pytest.raises(ValueError, match="seeds"):
        _run(seeds=seeds)


def test_invalid_score_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="score_window"):
        run_tdnr_noise_sweep(
            _config(),
            levels=(0.0,),
            seeds=(1,),
            score_window=25,
            reservoir_config=_reservoir(),
        )


def test_invalid_config_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="TemporalDriftNoiseConfig"):
        run_tdnr_noise_sweep("invalid", levels=(0.0,))


def test_global_numpy_rng_state_is_unchanged() -> None:
    before = np.random.get_state()
    _run(levels=(0.0,), seeds=(1,))
    after = np.random.get_state()

    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
