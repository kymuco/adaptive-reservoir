from __future__ import annotations

import pytest

from adaptive_reservoir import ReadoutConfig, ReservoirConfig
from adaptive_reservoir.benchmarks import BenchmarkResult, run_delayed_xor_benchmark
from adaptive_reservoir.benchmarks.delayed_xor import (
    BENCHMARK_NAME,
    _binary_accuracy_score,
    _generate_delayed_xor_stream,
    _rolling_binary_accuracy_scores,
)


def test_delayed_xor_benchmark_runs_with_small_sample_count() -> None:
    result = run_delayed_xor_benchmark(
        _config(),
        seed=42,
        n_samples=240,
        delay_a=2,
        delay_b=5,
        score_window=40,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.benchmark == BENCHMARK_NAME
    assert result.model == "adaptive_reservoir"
    assert result.topology == "modular_small_world"
    assert result.feature_mode == "multi_raw"
    assert result.readout == "sliding_ridge"
    assert result.seed == 42
    assert result.samples_seen == 240
    assert result.us_per_sample >= 0.0
    assert 0.0 <= result.pre_score <= 1.0
    assert 0.0 <= result.post_score <= 1.0
    assert 0.0 <= result.final_score <= 1.0
    assert 0.0 <= result.saturation_rate <= 1.0
    assert result.readout_sparsity is None or (
        0.0 <= result.readout_sparsity <= 1.0
    )


def test_delayed_xor_benchmark_is_deterministic_except_timing() -> None:
    first = run_delayed_xor_benchmark(
        _config(),
        seed=7,
        n_samples=240,
        delay_a=2,
        delay_b=5,
        score_window=40,
    )
    second = run_delayed_xor_benchmark(
        _config(),
        seed=7,
        n_samples=240,
        delay_a=2,
        delay_b=5,
        score_window=40,
    )

    assert first.pre_score == second.pre_score
    assert first.post_score == second.post_score
    assert first.final_score == second.final_score
    assert first.adapt_steps == second.adapt_steps
    assert first.saturation_rate == second.saturation_rate
    assert first.readout_sparsity == second.readout_sparsity
    assert first.samples_seen == second.samples_seen


def test_delayed_xor_benchmark_applies_seed_to_supplied_config() -> None:
    stale_seed_result = run_delayed_xor_benchmark(
        _config(seed=123),
        seed=7,
        n_samples=240,
        delay_a=2,
        delay_b=5,
        score_window=40,
    )
    matching_seed_result = run_delayed_xor_benchmark(
        _config(seed=7),
        seed=7,
        n_samples=240,
        delay_a=2,
        delay_b=5,
        score_window=40,
    )

    assert stale_seed_result.pre_score == matching_seed_result.pre_score
    assert stale_seed_result.post_score == matching_seed_result.post_score
    assert stale_seed_result.final_score == matching_seed_result.final_score
    assert stale_seed_result.adapt_steps == matching_seed_result.adapt_steps
    assert stale_seed_result.seed == 7


def test_delayed_xor_stream_changes_with_seed() -> None:
    first = list(
        _generate_delayed_xor_stream(
            seed=1,
            n_samples=8,
            delay_a=1,
            delay_b=3,
            input_dim=2,
        )
    )
    second = list(
        _generate_delayed_xor_stream(
            seed=2,
            n_samples=8,
            delay_a=1,
            delay_b=3,
            input_dim=2,
        )
    )

    assert first != second


def test_delayed_xor_stream_uses_actual_delayed_xor_targets() -> None:
    stream = list(
        _generate_delayed_xor_stream(
            seed=3,
            n_samples=20,
            delay_a=1,
            delay_b=4,
            input_dim=5,
        )
    )

    for sample, target in stream:
        assert len(sample) == 5
        expected = float(int(sample[1]) ^ int(sample[4]))
        assert target == expected


def test_delayed_xor_benchmark_default_config_uses_seed() -> None:
    result = run_delayed_xor_benchmark(
        seed=5,
        n_samples=240,
        delay_a=2,
        delay_b=5,
        score_window=40,
    )

    assert result.seed == 5
    assert result.topology == "modular_small_world"
    assert result.feature_mode == "multi_raw"
    assert result.samples_seen == 240


@pytest.mark.parametrize(
    ("n_samples", "score_window", "match"),
    [
        (0, 40, "n_samples"),
        (240, 0, "score_window"),
        (119, 40, "early, middle, and final"),
    ],
)
def test_delayed_xor_benchmark_rejects_invalid_windows(
    n_samples: int,
    score_window: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_delayed_xor_benchmark(
            _config(),
            seed=42,
            n_samples=n_samples,
            delay_a=2,
            delay_b=5,
            score_window=score_window,
        )


@pytest.mark.parametrize(
    ("delay_a", "delay_b", "score_window", "match"),
    [
        (0, 5, 40, "delays"),
        (2, 0, 40, "delays"),
        (4, 4, 40, "differ"),
        (2, 40, 40, "larger"),
    ],
)
def test_delayed_xor_benchmark_rejects_invalid_delays(
    delay_a: int,
    delay_b: int,
    score_window: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_delayed_xor_benchmark(
            _config(),
            seed=42,
            n_samples=240,
            delay_a=delay_a,
            delay_b=delay_b,
            score_window=score_window,
        )


def test_delayed_xor_benchmark_rejects_too_small_input_dim() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        run_delayed_xor_benchmark(
            ReservoirConfig(input_dim=0),
            n_samples=240,
            delay_a=2,
            delay_b=5,
            score_window=40,
        )


def test_delayed_xor_accuracy_score_is_bounded() -> None:
    assert _binary_accuracy_score([0.1, 0.9], [0.0, 1.0]) == 1.0
    assert _binary_accuracy_score([0.9, 0.1], [0.0, 1.0]) == 0.0


def test_delayed_xor_rolling_accuracy_scores_align_to_indices() -> None:
    scores = _rolling_binary_accuracy_scores(
        [0.1, 0.9, 0.8, 0.2],
        [0.0, 1.0, 1.0, 0.0],
        window=2,
    )

    assert scores == (None, 1.0, 1.0, 1.0)


def test_delayed_xor_benchmark_row_contains_future_report_columns() -> None:
    result = run_delayed_xor_benchmark(
        _config(),
        seed=42,
        n_samples=240,
        delay_a=2,
        delay_b=5,
        score_window=40,
    )

    assert set(result.to_row()) == {
        "adapt_steps",
        "benchmark",
        "feature_mode",
        "final_score",
        "model",
        "post_score",
        "pre_score",
        "readout",
        "readout_sparsity",
        "samples_seen",
        "saturation_rate",
        "seed",
        "topology",
        "us_per_sample",
    }


def _config(*, seed: int = 123) -> ReservoirConfig:
    return ReservoirConfig(
        input_dim=5,
        n_cells=8,
        topology="modular_small_world",
        feature_mode="multi_raw",
        seed=seed,
        readout=ReadoutConfig(name="sliding_ridge", update_interval=1),
    )
