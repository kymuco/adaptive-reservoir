from __future__ import annotations

import pytest

from adaptive_reservoir import AdaptiveReservoir, ReservoirConfig
from adaptive_reservoir.benchmarks.common import (
    BenchmarkResult,
    calculate_adapt_steps,
    readout_sparsity,
    regression_score,
    rolling_regression_scores,
)


def test_benchmark_result_to_row_is_stable() -> None:
    result = BenchmarkResult(
        benchmark="concept-drift",
        model="adaptive_reservoir",
        topology="ring_shortcuts",
        feature_mode="state_slow_raw",
        readout="sliding_ridge",
        seed=42,
        pre_score=0.8,
        post_score=0.2,
        final_score=0.7,
        adapt_steps=10,
        us_per_sample=12.5,
        saturation_rate=0.1,
        readout_sparsity=0.25,
        samples_seen=100,
    )

    assert list(result.to_row()) == [
        "benchmark",
        "model",
        "topology",
        "feature_mode",
        "readout",
        "seed",
        "pre_score",
        "post_score",
        "final_score",
        "adapt_steps",
        "us_per_sample",
        "saturation_rate",
        "readout_sparsity",
        "samples_seen",
    ]
    assert result.to_row()["final_score"] == 0.7


def test_benchmark_result_rejects_invalid_scores() -> None:
    with pytest.raises(ValueError, match="final_score"):
        BenchmarkResult(
            benchmark="concept-drift",
            model="adaptive_reservoir",
            topology="ring_shortcuts",
            feature_mode="state_slow_raw",
            readout="sliding_ridge",
            seed=42,
            pre_score=0.8,
            post_score=0.2,
            final_score=1.1,
            adapt_steps=None,
            us_per_sample=12.5,
            saturation_rate=0.1,
            readout_sparsity=None,
            samples_seen=100,
        )


def test_regression_score_perfect_predictions_are_one() -> None:
    assert regression_score([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]) == 1.0


def test_regression_score_is_bounded_for_bad_predictions() -> None:
    score = regression_score([100.0, 100.0, 100.0], [0.0, 1.0, 2.0])

    assert 0.0 <= score <= 1.0


def test_regression_score_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        regression_score([1.0], [1.0, 2.0])


def test_rolling_regression_scores_align_to_indices() -> None:
    scores = rolling_regression_scores(
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        window=2,
    )

    assert scores == (None, 1.0, 1.0, 1.0)


def test_calculate_adapt_steps_returns_none_if_threshold_never_reached() -> None:
    steps = calculate_adapt_steps(
        [None, 0.1, 0.2, 0.3],
        start_index=1,
        threshold=0.8,
    )

    assert steps is None


def test_calculate_adapt_steps_returns_first_recovery() -> None:
    steps = calculate_adapt_steps(
        [None, 0.1, 0.4, 0.8, 0.9],
        start_index=1,
        threshold=0.75,
    )

    assert steps == 3


def test_calculate_adapt_steps_can_count_from_earlier_origin() -> None:
    steps = calculate_adapt_steps(
        [None, 0.1, 0.4, 0.8, 0.9],
        start_index=3,
        threshold=0.75,
        origin_index=1,
    )

    assert steps == 3


def test_readout_sparsity_returns_none_for_object_without_weights() -> None:
    assert readout_sparsity(object()) is None


def test_readout_sparsity_reports_fraction_of_near_zero_weights() -> None:
    model = AdaptiveReservoir(ReservoirConfig(input_dim=2, n_cells=4, seed=42))

    sparsity = readout_sparsity(model)

    assert sparsity == 1.0
