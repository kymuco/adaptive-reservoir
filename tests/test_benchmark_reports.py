from __future__ import annotations

import csv
import json
from io import StringIO

from adaptive_reservoir.benchmarks.common import BenchmarkResult
from adaptive_reservoir.benchmarks.reports import (
    REPORT_COLUMNS,
    format_csv_report,
    format_json_report,
    format_markdown_report,
    result_to_report_row,
    results_to_report_rows,
)


def test_report_columns_are_stable() -> None:
    assert REPORT_COLUMNS == (
        "benchmark",
        "model",
        "topology",
        "feature_mode",
        "readout",
        "seed",
        "final_score",
        "adapt_steps",
        "us_per_sample",
        "saturation_rate",
        "readout_sparsity",
    )


def test_result_to_report_row_keeps_only_report_columns() -> None:
    row = result_to_report_row(_result())

    assert tuple(row) == REPORT_COLUMNS
    assert "pre_score" not in row
    assert "post_score" not in row
    assert "samples_seen" not in row
    assert row["final_score"] == 0.987654321


def test_results_to_report_rows_preserves_order() -> None:
    rows = results_to_report_rows([
        _result(benchmark="concept-drift", seed=1),
        _result(benchmark="temporal-drift", seed=2),
    ])

    assert rows[0]["benchmark"] == "concept-drift"
    assert rows[1]["benchmark"] == "temporal-drift"
    assert rows[0]["seed"] == 1
    assert rows[1]["seed"] == 2


def test_csv_report_includes_header_and_formatted_values() -> None:
    output = format_csv_report([_result(adapt_steps=None, readout_sparsity=None)])
    reader = csv.DictReader(StringIO(output))
    rows = list(reader)

    assert reader.fieldnames == list(REPORT_COLUMNS)
    assert rows == [
        {
            "benchmark": "temporal-drift",
            "model": "adaptive_reservoir",
            "topology": "ring_shortcuts",
            "feature_mode": "multi_raw",
            "readout": "sliding_ridge",
            "seed": "7",
            "final_score": "0.987654",
            "adapt_steps": "",
            "us_per_sample": "12.3457",
            "saturation_rate": "0.125",
            "readout_sparsity": "",
        }
    ]


def test_markdown_report_includes_summary_table_and_escapes_cells() -> None:
    output = format_markdown_report([
        _result(topology="ring|shortcuts", feature_mode="multi\nraw"),
    ])

    assert output.splitlines()[0].startswith("| benchmark | model | topology |")
    assert "| --- | --- | --- |" in output
    assert "ring\\|shortcuts" in output
    assert "multi raw" in output


def test_json_report_keeps_numeric_and_null_types() -> None:
    output = format_json_report([_result(adapt_steps=None, readout_sparsity=None)])
    rows = json.loads(output)

    assert rows == [
        {
            "benchmark": "temporal-drift",
            "model": "adaptive_reservoir",
            "topology": "ring_shortcuts",
            "feature_mode": "multi_raw",
            "readout": "sliding_ridge",
            "seed": 7,
            "final_score": 0.987654321,
            "adapt_steps": None,
            "us_per_sample": 12.345678,
            "saturation_rate": 0.125,
            "readout_sparsity": None,
        }
    ]


def test_empty_reports_are_still_well_formed() -> None:
    assert format_csv_report([]) == ",".join(REPORT_COLUMNS) + "\n"
    assert format_json_report([]) == "[]"

    markdown_lines = format_markdown_report([]).splitlines()
    assert len(markdown_lines) == 2
    assert markdown_lines[0].startswith("| benchmark |")
    assert markdown_lines[1].startswith("| --- |")


def _result(
    *,
    benchmark: str = "temporal-drift",
    seed: int = 7,
    topology: str = "ring_shortcuts",
    feature_mode: str = "multi_raw",
    adapt_steps: int | None = 42,
    readout_sparsity: float | None = 0.25,
) -> BenchmarkResult:
    return BenchmarkResult(
        benchmark=benchmark,
        model="adaptive_reservoir",
        topology=topology,
        feature_mode=feature_mode,
        readout="sliding_ridge",
        seed=seed,
        pre_score=0.1,
        post_score=0.2,
        final_score=0.987654321,
        adapt_steps=adapt_steps,
        us_per_sample=12.345678,
        saturation_rate=0.125,
        readout_sparsity=readout_sparsity,
        samples_seen=1200,
    )
