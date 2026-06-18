from __future__ import annotations

import pytest

from adaptive_reservoir import ReadoutConfig, ReservoirConfig
from adaptive_reservoir.benchmarks import BenchmarkResult, run_concept_drift_benchmark
from adaptive_reservoir.benchmarks.concept_drift import (
    BENCHMARK_NAME,
    _generate_concept_drift_stream,
)


def test_concept_drift_benchmark_runs_with_small_sample_count() -> None:
    result = run_concept_drift_benchmark(
        _config(),
        seed=42,
        n_samples=220,
        drift_at=110,
        score_window=24,
    )

    assert isinstance(result, BenchmarkResult)
    assert result.benchmark == BENCHMARK_NAME
    assert result.model == "adaptive_reservoir"
    assert result.topology == "ring_shortcuts"
    assert result.feature_mode == "state_slow_raw"
    assert result.readout == "sliding_ridge"
    assert result.seed == 42
    assert result.samples_seen == 220
    assert result.us_per_sample >= 0.0
    assert 0.0 <= result.pre_score <= 1.0
    assert 0.0 <= result.post_score <= 1.0
    assert 0.0 <= result.final_score <= 1.0
    assert 0.0 <= result.saturation_rate <= 1.0
    assert result.readout_sparsity is None or 0.0 <= result.readout_sparsity <= 1.0


def test_concept_drift_benchmark_is_deterministic_except_timing() -> None:
    first = run_concept_drift_benchmark(
        _config(),
        seed=7,
        n_samples=220,
        drift_at=110,
        score_window=24,
    )
    second = run_concept_drift_benchmark(
        _config(),
        seed=7,
        n_samples=220,
        drift_at=110,
        score_window=24,
    )

    assert first.pre_score == second.pre_score
    assert first.post_score == second.post_score
    assert first.final_score == second.final_score
    assert first.adapt_steps == second.adapt_steps
    assert first.saturation_rate == second.saturation_rate
    assert first.readout_sparsity == second.readout_sparsity
    assert first.samples_seen == second.samples_seen


def test_concept_drift_stream_changes_with_seed() -> None:
    first = list(
        _generate_concept_drift_stream(
            seed=1,
            n_samples=8,
            drift_at=4,
            input_dim=2,
        )
    )
    second = list(
        _generate_concept_drift_stream(
            seed=2,
            n_samples=8,
            drift_at=4,
            input_dim=2,
        )
    )

    assert first != second


def test_concept_drift_benchmark_default_config_uses_seed() -> None:
    result = run_concept_drift_benchmark(
        seed=5,
        n_samples=220,
        drift_at=110,
        score_window=24,
    )

    assert result.seed == 5
    assert result.samples_seen == 220


@pytest.mark.parametrize(
    ("n_samples", "drift_at", "score_window", "match"),
    [
        (0, 10, 4, "n_samples"),
        (100, 10, 10, "pre-drift"),
        (100, 90, 20, "post-drift"),
        (100, 50, 0, "score_window"),
    ],
)
def test_concept_drift_benchmark_rejects_invalid_windows(
    n_samples: int,
    drift_at: int,
    score_window: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        run_concept_drift_benchmark(
            _config(),
            seed=42,
            n_samples=n_samples,
            drift_at=drift_at,
            score_window=score_window,
        )


def test_concept_drift_benchmark_rejects_too_small_input_dim() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        run_concept_drift_benchmark(
            ReservoirConfig(input_dim=1),
            n_samples=220,
            drift_at=110,
            score_window=24,
        )


def test_concept_drift_benchmark_row_contains_future_report_columns() -> None:
    result = run_concept_drift_benchmark(
        _config(),
        seed=42,
        n_samples=220,
        drift_at=110,
        score_window=24,
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


def _config() -> ReservoirConfig:
    return ReservoirConfig(
        input_dim=2,
        n_cells=8,
        topology="ring_shortcuts",
        feature_mode="state_slow_raw",
        seed=123,
        readout=ReadoutConfig(name="sliding_ridge", update_interval=1),
    )
