from __future__ import annotations

import json

import pytest

from adaptive_reservoir.benchmarks.rls_sweep import (
    RLS_SWEEP_REPORT_COLUMNS,
    RLSSweepResult,
    format_rls_sweep_csv_report,
    format_rls_sweep_json_report,
    format_rls_sweep_markdown_report,
    run_rls_sweep,
)
from adaptive_reservoir.benchmarks.runner import (
    build_parser,
    format_benchmark_output,
    run_benchmark_from_args,
)
from adaptive_reservoir.core.config import READOUT_NAMES, ReadoutConfig
from adaptive_reservoir.experimental.rls import RLS_READOUT_NAME


def _small_sweep(**overrides: object) -> tuple[RLSSweepResult, ...]:
    kwargs = {
        "seed": 7,
        "n_samples": 96,
        "drift_at": 48,
        "score_window": 16,
        "input_dim": 2,
        "n_cells": 24,
        "forgetting_factors": (1.0,),
        "covariance_scales": (100.0,),
        "feature_modes": ("state_slow_raw",),
        "topologies": ("ring_shortcuts",),
    }
    kwargs.update(overrides)
    return run_rls_sweep(**kwargs)  # type: ignore[arg-type]


def _stable_rows(results: tuple[RLSSweepResult, ...]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for result in results:
        row = result.to_row()
        row.pop("us_per_sample")
        rows.append(row)
    return tuple(rows)


def test_run_rls_sweep_returns_cartesian_product() -> None:
    results = _small_sweep(
        forgetting_factors=(1.0, 0.99),
        covariance_scales=(10.0, 100.0),
        feature_modes=("state_slow_raw",),
        topologies=("ring_shortcuts", "modular_small_world"),
    )

    assert len(results) == 8
    assert [result.topology for result in results[:4]] == ["ring_shortcuts"] * 4
    assert {result.forgetting_factor for result in results} == {1.0, 0.99}
    assert {result.covariance_scale for result in results} == {10.0, 100.0}


def test_run_rls_sweep_rows_include_lambda_and_covariance_scale() -> None:
    result = _small_sweep(forgetting_factors=(0.99,), covariance_scales=(250.0,))[0]

    row = result.to_row()

    assert row["benchmark"] == "rls-sweep"
    assert row["base_benchmark"] == "concept-drift"
    assert row["readout"] == RLS_READOUT_NAME
    assert row["lambda"] == 0.99
    assert row["covariance_scale"] == 250.0
    assert result.samples_seen == 96


def test_run_rls_sweep_is_deterministic_for_same_seed_except_timing() -> None:
    first = _small_sweep(forgetting_factors=(1.0, 0.99))
    second = _small_sweep(forgetting_factors=(1.0, 0.99))

    assert _stable_rows(first) == _stable_rows(second)


@pytest.mark.parametrize("forgetting_factor", [0.0, -0.1, 1.01, float("inf")])
def test_run_rls_sweep_rejects_invalid_lambda(forgetting_factor: float) -> None:
    with pytest.raises(ValueError):
        _small_sweep(forgetting_factors=(forgetting_factor,))


@pytest.mark.parametrize("covariance_scale", [0.0, -1.0, float("inf")])
def test_run_rls_sweep_rejects_invalid_covariance_scale(covariance_scale: float) -> None:
    with pytest.raises(ValueError):
        _small_sweep(covariance_scales=(covariance_scale,))


def test_run_rls_sweep_rejects_invalid_feature_mode() -> None:
    with pytest.raises(ValueError):
        _small_sweep(feature_modes=("unknown",))


def test_run_rls_sweep_rejects_invalid_topology() -> None:
    with pytest.raises(ValueError):
        _small_sweep(topologies=("unknown",))


def test_rls_sweep_csv_output_has_expected_columns() -> None:
    output = format_rls_sweep_csv_report(_small_sweep())

    assert output.splitlines()[0] == ",".join(RLS_SWEEP_REPORT_COLUMNS)
    assert "rls-sweep" in output
    assert "experimental_rls" in output


def test_rls_sweep_markdown_output_has_expected_columns() -> None:
    output = format_rls_sweep_markdown_report(_small_sweep())

    assert output.startswith("| benchmark | base_benchmark | model | topology |")
    assert "| lambda | covariance_scale |" in output
    assert "experimental_rls" in output


def test_rls_sweep_json_output_is_valid() -> None:
    output = format_rls_sweep_json_report(_small_sweep())
    rows = json.loads(output)

    assert len(rows) == 1
    assert rows[0]["benchmark"] == "rls-sweep"
    assert rows[0]["lambda"] == 1.0
    assert rows[0]["covariance_scale"] == 100.0


@pytest.mark.parametrize("output_format", ["csv", "markdown", "json"])
def test_runner_accepts_rls_sweep_with_table_formats(output_format: str) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "rls-sweep",
            "--format",
            output_format,
            "--samples",
            "96",
            "--drift-at",
            "48",
            "--score-window",
            "16",
            "--cells",
            "24",
            "--lambda",
            "1.0,0.99",
            "--covariance-scale",
            "100",
            "--feature-mode",
            "state_slow_raw",
            "--topology",
            "ring_shortcuts",
        ]
    )

    result = run_benchmark_from_args(args)
    output = format_benchmark_output(result, output_format=args.format)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert "rls-sweep" in output
    assert "experimental_rls" in output


def test_runner_rejects_rls_sweep_text_output() -> None:
    parser = build_parser()
    args = parser.parse_args(["rls-sweep"])

    with pytest.raises(ValueError, match="text output is not supported"):
        run_benchmark_from_args(args)


def test_runner_rejects_rls_sweep_options_for_stable_benchmarks() -> None:
    parser = build_parser()
    args = parser.parse_args(["concept-drift", "--lambda", "0.99"])

    with pytest.raises(ValueError, match="--lambda is not supported"):
        run_benchmark_from_args(args)


def test_rls_sweep_does_not_register_experimental_rls_as_stable_readout() -> None:
    assert RLS_READOUT_NAME not in READOUT_NAMES

    with pytest.raises(ValueError):
        ReadoutConfig(name=RLS_READOUT_NAME)  # type: ignore[arg-type]
