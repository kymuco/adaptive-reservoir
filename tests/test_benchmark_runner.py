from __future__ import annotations

import argparse

import pytest

from adaptive_reservoir.benchmarks.common import BenchmarkResult
from adaptive_reservoir.benchmarks.runner import (
    BENCHMARKS,
    build_config_from_args,
    build_parser,
    format_result,
    main,
    normalize_benchmark_name,
    run_benchmark_from_args,
)


def test_runner_registers_all_benchmarks() -> None:
    assert set(BENCHMARKS) == {"concept-drift", "temporal-drift", "delayed-xor"}


def test_normalize_benchmark_name_accepts_underscore_aliases() -> None:
    assert normalize_benchmark_name(" Temporal_Drift ") == "temporal-drift"
    assert normalize_benchmark_name("concept_drift") == "concept-drift"
    assert normalize_benchmark_name("delayed_xor") == "delayed-xor"


def test_build_config_from_args_uses_benchmark_defaults() -> None:
    args = argparse.Namespace(
        cells=None,
        feature_mode=None,
        input_dim=None,
        readout=None,
        seed=42,
        topology=None,
    )

    temporal = build_config_from_args(args, benchmark_name="temporal-drift")
    delayed = build_config_from_args(args, benchmark_name="delayed-xor")

    assert temporal.input_dim == 2
    assert temporal.n_cells == 96
    assert temporal.topology == "ring_shortcuts"
    assert temporal.feature_mode == "multi_raw"
    assert delayed.input_dim == 1
    assert delayed.topology == "modular_small_world"


def test_build_config_from_args_respects_overrides() -> None:
    args = argparse.Namespace(
        cells=12,
        feature_mode="state_slow_raw",
        input_dim=3,
        readout="nlms",
        seed=7,
        topology="ring_shortcuts",
    )

    config = build_config_from_args(args, benchmark_name="delayed-xor")

    assert config.input_dim == 3
    assert config.n_cells == 12
    assert config.feature_mode == "state_slow_raw"
    assert config.readout.name == "nlms"
    assert config.seed == 7


def test_format_result_outputs_stable_key_value_rows() -> None:
    result = BenchmarkResult(
        benchmark="smoke",
        model="adaptive_reservoir",
        topology="ring_shortcuts",
        feature_mode="multi_raw",
        readout="sliding_ridge",
        seed=3,
        pre_score=0.1,
        post_score=0.2,
        final_score=0.3,
        adapt_steps=None,
        us_per_sample=12.345678,
        saturation_rate=0.0,
        readout_sparsity=1.0,
        samples_seen=10,
    )

    output = format_result(result)

    assert "benchmark: smoke" in output
    assert "adapt_steps: none" in output
    assert "us_per_sample: 12.3457" in output
    assert output.splitlines()[0] == "benchmark: smoke"


def test_run_benchmark_from_args_dispatches_temporal_drift() -> None:
    args = build_parser().parse_args(
        [
            "temporal-drift",
            "--seed",
            "3",
            "--samples",
            "260",
            "--drift-at",
            "130",
            "--delay-before",
            "2",
            "--delay-after",
            "8",
            "--score-window",
            "32",
            "--cells",
            "8",
        ]
    )

    result = run_benchmark_from_args(args)

    assert result.benchmark == "temporal-drift"
    assert result.seed == 3
    assert result.samples_seen == 260


def test_main_runs_concept_drift(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "concept-drift",
            "--seed",
            "3",
            "--samples",
            "220",
            "--drift-at",
            "110",
            "--score-window",
            "24",
            "--cells",
            "8",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "benchmark: concept-drift" in output
    assert "samples_seen: 220" in output


def test_main_runs_temporal_drift(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "temporal_drift",
            "--seed",
            "3",
            "--samples",
            "260",
            "--drift-at",
            "130",
            "--delay-before",
            "2",
            "--delay-after",
            "8",
            "--score-window",
            "32",
            "--cells",
            "8",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "benchmark: temporal-drift" in output
    assert "samples_seen: 260" in output


def test_main_runs_delayed_xor(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "delayed-xor",
            "--seed",
            "3",
            "--samples",
            "240",
            "--delay-a",
            "2",
            "--delay-b",
            "5",
            "--score-window",
            "40",
            "--cells",
            "8",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "benchmark: delayed-xor" in output
    assert "samples_seen: 240" in output


def test_main_rejects_unknown_benchmark(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["unknown"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "unknown benchmark" in captured.err
    assert "temporal-drift" in captured.err
