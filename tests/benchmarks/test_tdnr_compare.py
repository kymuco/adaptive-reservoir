"""TDNR-01 Stage 2B comparison, scoring, and causality tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from adaptive_reservoir import AdaptiveReservoir, ReadoutConfig, ReservoirConfig
from adaptive_reservoir.benchmarks.tdnr_baselines import (
    LagBufferRidgeBaseline,
    PersistenceBaseline,
    ZeroBaseline,
)
from adaptive_reservoir.benchmarks.tdnr_compare import (
    DEFAULT_SEEDS,
    MODEL_NAMES,
    WINDOW_NAMES,
    compare_tdnr_stream,
    run_tdnr_comparison,
)
from adaptive_reservoir.benchmarks.temporal_drift_noise import (
    TemporalDriftNoiseConfig,
    generate_temporal_drift_noise_stream,
)


def _config(**changes):
    base = TemporalDriftNoiseConfig(
        seed=17,
        n_samples=64,
        drift_at=32,
        delay_before=3,
        delay_after=12,
        input_noise=0.2,
        target_noise=0.1,
    )
    return replace(base, **changes)


def _reservoir():
    return ReservoirConfig(
        input_dim=2,
        n_cells=8,
        topology="ring_shortcuts",
        feature_mode="multi_raw",
        seed=999,
        readout=ReadoutConfig(name="sliding_ridge", window_size=16),
    )


def _stream(**changes):
    return generate_temporal_drift_noise_stream(_config(**changes))


def _compare(stream=None, *, width=8):
    if stream is None:
        stream = _stream()
    return compare_tdnr_stream(
        stream,
        score_window=width,
        reservoir_config=_reservoir(),
    )


def _by_name(results):
    return {result.model: result for result in results}


def test_fixed_seeds_and_model_order() -> None:
    assert DEFAULT_SEEDS == (0, 1, 2)
    assert MODEL_NAMES == (
        "adaptive_reservoir",
        "zero",
        "persistence",
        "lag_buffer_ridge",
    )
    assert WINDOW_NAMES == ("pre", "early_post", "late_post")

    results = run_tdnr_comparison(
        _config(seed=999),
        seeds=(4, 7),
        score_window=8,
        reservoir_config=_reservoir(),
    )
    assert [(item.seed, item.model) for item in results] == [
        (seed, model) for seed in (4, 7) for model in MODEL_NAMES
    ]
    assert all(len(item.predictions) == 64 for item in results)


def test_repeated_comparison_is_deterministic() -> None:
    stream = _stream()
    first = _compare(stream)
    second = _compare(stream)
    assert first == second
    for result in first:
        assert np.all(np.isfinite(result.predictions))
        assert all(np.isfinite(w.mae) and np.isfinite(w.rmse) for w in result.windows)


def test_seed_order_does_not_change_individual_results() -> None:
    first = run_tdnr_comparison(
        _config(), seeds=(2, 3), score_window=8, reservoir_config=_reservoir()
    )
    second = run_tdnr_comparison(
        _config(), seeds=(3, 2), score_window=8, reservoir_config=_reservoir()
    )
    assert {(r.seed, r.model): r for r in first} == {
        (r.seed, r.model): r for r in second
    }


def test_window_boundaries_are_exact_and_disjoint() -> None:
    for result in _compare():
        windows = result.windows
        assert [(w.name, w.start, w.stop) for w in windows] == [
            ("pre", 24, 32),
            ("early_post", 32, 40),
            ("late_post", 56, 64),
        ]
        assert windows[0].stop <= windows[1].start
        assert windows[1].stop <= windows[2].start
        assert all(w.stop - w.start == 8 for w in windows)


def test_scoring_uses_clean_targets_and_correct_formulas() -> None:
    stream = _stream()
    assert not np.array_equal(stream.training_targets, stream.clean_targets)
    for result in _compare(stream):
        for window in result.windows:
            errors = (
                np.asarray(result.predictions[window.start : window.stop])
                - stream.clean_targets[window.start : window.stop]
            )
            assert window.mae == pytest.approx(float(np.mean(np.abs(errors))), abs=1e-14)
            assert window.rmse == pytest.approx(
                float(np.sqrt(np.mean(errors**2))), abs=1e-14
            )


def test_all_models_predict_before_supervised_update() -> None:
    stream = _stream()
    results = _by_name(_compare(stream))
    assert all(result.predictions[0] == 0.0 for result in results.values())
    np.testing.assert_array_equal(results["zero"].predictions, 0.0)
    expected_persistence = np.concatenate(([0.0], stream.training_targets[:-1]))
    np.testing.assert_array_equal(
        results["persistence"].predictions,
        expected_persistence,
    )


def test_comparison_matches_direct_model_steps() -> None:
    stream = _stream()
    results = _by_name(_compare(stream))
    reservoir = AdaptiveReservoir(replace(_reservoir(), seed=stream.config.seed))
    direct_reservoir = tuple(
        reservoir.step(sample, target=target).prediction
        for sample, target in stream.iter_training()
    )
    assert results["adaptive_reservoir"].predictions == direct_reservoir

    for name, model in (
        ("zero", ZeroBaseline()),
        ("persistence", PersistenceBaseline()),
        ("lag_buffer_ridge", LagBufferRidgeBaseline()),
    ):
        direct = tuple(
            model.step(sample, target) for sample, target in stream.iter_training()
        )
        assert results[name].predictions == direct


def test_current_training_target_cannot_change_its_own_prediction() -> None:
    stream = _stream()
    altered_targets = stream.training_targets.copy()
    index = 12
    altered_targets[index] += 50.0
    changed = replace(stream, training_targets=altered_targets)
    original_results = _by_name(_compare(stream))
    changed_results = _by_name(_compare(changed))

    for name in MODEL_NAMES:
        assert (
            original_results[name].predictions[: index + 1]
            == changed_results[name].predictions[: index + 1]
        )
    assert (
        original_results["persistence"].predictions[index + 1]
        != changed_results["persistence"].predictions[index + 1]
    )


def test_future_input_changes_cannot_affect_past_predictions() -> None:
    stream = _stream()
    altered_inputs = stream.inputs.copy()
    boundary = 20
    altered_inputs[boundary:, :] += 3.0
    changed = replace(stream, inputs=altered_inputs)
    original_results = _by_name(_compare(stream))
    changed_results = _by_name(_compare(changed))

    for name in MODEL_NAMES:
        assert (
            original_results[name].predictions[:boundary]
            == changed_results[name].predictions[:boundary]
        )


def test_clean_targets_are_not_used_for_training_or_prediction() -> None:
    stream = _stream()
    changed = replace(stream, clean_targets=stream.clean_targets + 10.0)
    original_results = _by_name(_compare(stream))
    changed_results = _by_name(_compare(changed))

    for name in MODEL_NAMES:
        assert original_results[name].predictions == changed_results[name].predictions
        assert original_results[name].windows != changed_results[name].windows


def test_latent_signal_and_drift_weights_are_not_model_inputs() -> None:
    stream = _stream()
    changed = replace(
        stream,
        latent_signal=stream.latent_signal + 5.0,
        drift_weight=np.zeros_like(stream.drift_weight),
    )
    assert _compare(stream) == _compare(changed)


def test_training_targets_not_clean_targets_drive_persistence() -> None:
    stream = _stream()
    changed = replace(stream, training_targets=np.zeros(stream.n_samples))
    original = _by_name(_compare(stream))
    altered = _by_name(_compare(changed))

    assert original["persistence"].predictions != altered["persistence"].predictions
    np.testing.assert_array_equal(altered["persistence"].predictions, 0.0)
    assert original["zero"] == altered["zero"]


def test_adjacent_post_windows_are_allowed() -> None:
    stream = _stream(n_samples=48)
    results = _compare(stream)
    for result in results:
        assert [(w.start, w.stop) for w in result.windows] == [
            (24, 32), (32, 40), (40, 48)
        ]


@pytest.mark.parametrize("width", [0, -1, 1.5, True, 33])
def test_invalid_score_window_rejected(width) -> None:
    with pytest.raises(ValueError, match="score_window"):
        _compare(width=width)


def test_overlapping_post_windows_rejected() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        _compare(_stream(n_samples=47))


def test_non_abrupt_scenario_rejected() -> None:
    with pytest.raises(ValueError, match="abrupt"):
        _compare(_stream(scenario="stationary"))
    with pytest.raises(ValueError, match="abrupt"):
        run_tdnr_comparison(
            _config(scenario="stationary"),
            seeds=(0,),
            score_window=8,
            reservoir_config=_reservoir(),
        )


@pytest.mark.parametrize("seeds", [(), (0, 0), (-1,), (True,), (1.5,), [0, 1]])
def test_invalid_seed_sets_rejected(seeds) -> None:
    with pytest.raises(ValueError, match="seeds"):
        run_tdnr_comparison(
            _config(), seeds=seeds, score_window=8, reservoir_config=_reservoir()
        )


def test_invalid_reservoir_configuration_rejected() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        _compare_reservoir_config(ReservoirConfig(input_dim=3))
    with pytest.raises(TypeError, match="ReservoirConfig"):
        _compare_reservoir_config("invalid")


def _compare_reservoir_config(config) -> None:
    compare_tdnr_stream(_stream(), score_window=8, reservoir_config=config)


def test_invalid_stream_rejected() -> None:
    with pytest.raises(TypeError, match="TemporalDriftNoiseStream"):
        compare_tdnr_stream(None, score_window=8, reservoir_config=_reservoir())


def test_invalid_stream_arrays_rejected() -> None:
    stream = _stream()
    with pytest.raises(ValueError, match="training_targets"):
        _compare(replace(stream, training_targets=stream.training_targets[:-1]))
    with pytest.raises(ValueError, match="inputs"):
        _compare(replace(stream, inputs=stream.inputs[:, :1]))
    with pytest.raises(ValueError, match="finite"):
        bad = stream.clean_targets.copy()
        bad[0] = float("nan")
        _compare(replace(stream, clean_targets=bad))
