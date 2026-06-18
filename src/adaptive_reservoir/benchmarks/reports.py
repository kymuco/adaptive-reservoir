"""Report formatting helpers for benchmark results."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from io import StringIO

from adaptive_reservoir.benchmarks.common import BenchmarkResult

REPORT_COLUMNS = (
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


def result_to_report_row(result: BenchmarkResult) -> dict[str, object]:
    """Return a compact report row for one benchmark result."""

    row = result.to_row()
    return {column: row[column] for column in REPORT_COLUMNS}


def results_to_report_rows(
    results: Sequence[BenchmarkResult],
) -> tuple[dict[str, object], ...]:
    """Return compact report rows for benchmark results."""

    return tuple(result_to_report_row(result) for result in results)


def format_csv_report(results: Sequence[BenchmarkResult]) -> str:
    """Format benchmark results as a CSV report."""

    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(REPORT_COLUMNS)
    for row in results_to_report_rows(results):
        writer.writerow([format_report_value(row[column]) for column in REPORT_COLUMNS])
    return output.getvalue()


def format_markdown_report(results: Sequence[BenchmarkResult]) -> str:
    """Format benchmark results as a Markdown summary table."""

    lines = [
        "| " + " | ".join(REPORT_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in REPORT_COLUMNS) + " |",
    ]
    for row in results_to_report_rows(results):
        cells = [_markdown_cell(row[column]) for column in REPORT_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_json_report(results: Sequence[BenchmarkResult]) -> str:
    """Format benchmark results as a JSON report."""

    return json.dumps(
        list(results_to_report_rows(results)),
        indent=2,
        allow_nan=False,
    )


def format_report_value(value: object) -> str:
    """Format a report cell value for text table formats."""

    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_cell(value: object) -> str:
    return format_report_value(value).replace("\n", " ").replace("|", "\\|")


__all__ = [
    "REPORT_COLUMNS",
    "format_csv_report",
    "format_json_report",
    "format_markdown_report",
    "format_report_value",
    "result_to_report_row",
    "results_to_report_rows",
]
