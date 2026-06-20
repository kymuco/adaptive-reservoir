from __future__ import annotations

import math

import pytest

from benchmarks.perf_smoke import (
    PerfSmokeResult,
    format_markdown,
    main,
    run_perf_smoke,
)


def test_run_perf_smoke_returns_expected_scenarios() -> None:
    results = run_perf_smoke(seed=3, samples=8, warmup=2, n_cells=6, input_dim=2)

    assert tuple(result.scenario for result in results) == (
        "core_ring_shortcuts",
        "core_random_sparse",
        "model_ring_sliding_ridge",
        "model_ring_replay_ridge",
    )
    for result in results:
        assert result.samples == 8
        assert result.warmup == 2
        assert result.seed == 3
        assert result.n_cells == 6
        assert result.input_dim == 2
        assert math.isfinite(result.us_per_sample)
        assert result.us_per_sample > 0.0


def test_perf_smoke_result_to_row_is_stable() -> None:
    result = PerfSmokeResult(
        scenario="smoke",
        samples=8,
        warmup=2,
        seed=3,
        n_cells=6,
        input_dim=2,
        us_per_sample=12.3456,
    )

    assert result.to_row() == {
        "scenario": "smoke",
        "samples": 8,
        "warmup": 2,
        "seed": 3,
        "n_cells": 6,
        "input_dim": 2,
        "us_per_sample": 12.3456,
    }


def test_format_markdown_outputs_stable_table() -> None:
    output = format_markdown(
        [
            PerfSmokeResult(
                scenario="core_ring_shortcuts",
                samples=8,
                warmup=2,
                seed=3,
                n_cells=6,
                input_dim=2,
                us_per_sample=12.34567,
            )
        ]
    )

    assert output.splitlines()[0] == (
        "| scenario | samples | warmup | seed | n_cells | input_dim | us_per_sample |"
    )
    assert "| core_ring_shortcuts | 8 | 2 | 3 | 6 | 2 | 12.3457 |" in output


def test_main_prints_perf_smoke_table(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--seed", "3", "--samples", "8", "--warmup", "2", "--cells", "6"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "us_per_sample" in output
    assert "core_ring_shortcuts" in output
    assert "model_ring_replay_ridge" in output


@pytest.mark.parametrize(
    "argv",
    [
        ["--samples", "0"],
        ["--warmup", "-1"],
        ["--cells", "1"],
        ["--input-dim", "1"],
    ],
)
def test_main_rejects_invalid_arguments(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(argv)

    assert exc_info.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_format_markdown_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="results must not be empty"):
        format_markdown([])
