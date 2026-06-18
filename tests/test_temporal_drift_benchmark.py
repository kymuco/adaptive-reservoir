from __future__ import annotations

import pytest

from adaptive_reservoir import ReadoutConfig, ReservoirConfig
from adaptive_reservoir.benchmarks import BenchmarkResult, run_temporal_drift_benchmark
from adaptive_reservoir.benchmarks.temporal_drift import (
    BENCHMARK_NAME,
    _generate_temporal_drift_stream,
)


def test_temporal_drift_benchmark_runs_with_small_sample_count() -> None:
    result = run_temporal_drift_benchmark(
        _config(),
        seed=42,
        n_samples=260,
        drift_at=130,
        delay_before=2,
        delay_after=8,
        score_window=32,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.benchmark == BENCHMARK_NAME
    assert result.model == "adaptive_reservoir"
    assert result.topology == "ring_shortcuts"
    assert result.feature_mode == "multi_raw"
    assert result.readout == "sliding_ridge"
    assert result.seed == 42
    assert result.samples_seen == 260
    assert result.us_per_sample >= 0.0
    assert 0.0 <= result.pre_score <= 1.0
    assert 0.0 <= result.post_score <= 1.0
    assert 0.0 <= result.final_score <= 1.0
    assert 0.0 <= result.saturation_rate <= 1.0
    assert result.readout_sparsity is None or (
        0.0 <= result.readout_sparsity <= 1.0
    )


def test_temporal_drift_benchmark_is_deterministic_except_timing() -> None:
    first = run_temporal_drift_benchmark(
        _config(),
        seed=7,
        n_samples=260,
        drift_at=130,
        delay_before=2,
        delay_after=8,
        score_window=32,
    )
    second = run_temporal_drift_benchmark(
        _config(),
        seed=7,
        n_samples=260,
        drift_at=130,
        delay_before=2,
        delay_after=8,
        score_window=32,
    )

    assert first.pre_score == second.pre_score
    assert first.post_score == second.post_score
    assert first.final_score == second.final_score
    assert first.adapt_steps == second.adapt_steps
    assert first.saturation_rate == second.saturation_rate
    assert first.readout_sparsity == second.readout_sparsity
    assert first.samples_seen == second.samples_seen


def test_temporal_drift_benchmark_applies_seed_to_supplied_config() -> None:
    stale_seed_result = run_temporal_drift_benchmark(
        _config(seed=123),
        seed=7,
        n_samples=260,
        drift_at=130,
        delay_before=2,
        delay_after=8,
        score_window=32,
    )
    matching_seed_result = run_temporal_drift_benchmark(
        _config(seed=7),
        seed=7,
        n_samples=260,
        drift_at=130,
        delay_before=2,
        delay_after=8,
        score_window=32,
    )

    assert stale_seed_result.pre_score == matching_seed_result.pre_score
    assert stale_seed_result.post_score == matching_seed_result.post_score
    assert stale_seed_result.final_score == matching_seed_result.final_score
    assert stale_seed_result.adapt_steps == matching_seed_result.adapt_steps
    assert stale_seed_result.seed == 7


def test_temporal_drift_stream_changes_with_seed() -> None:
    first = list(
        _generate_temporal_drift_stream(
            seed=1,
            n_samples=8,
            drift_at=4,
            delay_before=1,
            delay_after=3,
            input_dim=2,
        )
    )
    second = list(
        _generate_temporal_drift_stream(
            seed=2,
            n_samples=8,
            drift_at=4,
            delay_before=1,
            delay_after=3,
            input_dim=2,
        )
    )

    assert first != second


def test_temporal_drift_stream_changes_delay_rule_at_drift() -> None:
    stream = list(
        _generate_temporal_drift_stream(
            seed=3,
            n_samples=12,
            drift_at=6,
            delay_before=1,
            delay_after=4,
            input_dim=3,
        )
    )

    assert len(stream[0][0]) == 3
    assert stream[5][1] != stream[6][1]


def test_temporal_drift_benchmark_default_config_uses_seed() -> None:
    result = run_temporal_drift_benchmark(
        seed=5,
        n_samples=260,
        drift_at=130,
        delay_before=2,
        delay_after=8,
        score_window=32,
    )

    assert result.seed == 5
    assert result.feature_mode == "multi_raw"
    assert result.samples_seen == 260


@pytest.mark.parametrize(
    ("n_samples", "drift_at", "score_window", "match"),
    [
        (0, 10, 4, "n_samples"),
        (100, 10, 10, "pre-drift"),
        (100, 90, 20, "post-drift"),
        (100, 50, 0, "score_window"),
    ],
)
def test_temporal_drift_benchmark_rejects_invalid_windows(
    n_samples: int,
    drift_at: int,
    score_window: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_temporal_drift_benchmark(
            _config(),
            seed=42,
            n_samples=n_samples,
            drift_at=drift_at,
            delay_before=2,
            delay_after=8,
            score_window=score_window,
        )


@pytest.mark.parametrize(
    ("delay_before", "delay_after", "score_window", "match"),
    [
        (0, 8, 32, "delays"),
        (2, 0, 32, "delays"),
        (4, 4, 32, "differ"),
        (2, 32, 32, "larger"),
    ],
)
def test_temporal_drift_benchmark_rejects_invalid_delays(
    delay_before: int,
    delay_after: int,
    score_window: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_temporal_drift_benchmark(
            _config(),
            seed=42,
            n_samples=260,
            drift_at=130,
            delay_before=delay_before,
            delay_after=delay_after,
            score_window=score_window,
        )


def test_temporal_drift_benchmark_rejects_too_small_input_dim() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        run_temporal_drift_benchmark(
            ReservoirConfig(input_dim=1),
            n_samples=260,
            drift_at=130,
            delay_before=2,
            delay_after=8,
            score_window=32,
        )


def test_temporal_drift_benchmark_row_contains_future_report_columns() -> None:
    result = run_temporal_drift_benchmark(
        _config(),
        seed=42,
        n_samples=260,
        drift_at=130,
        delay_before=2,
        delay_after=8,
        score_window=32,
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
        input_dim=2,
        n_cells=8,
        topology="ring_shortcuts",
        feature_mode="multi_raw",
        seed=seed,
        readout=ReadoutConfig(name="sliding_ridge", update_interval=1),
    )
