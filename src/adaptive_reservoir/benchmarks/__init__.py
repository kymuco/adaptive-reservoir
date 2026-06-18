"""Benchmark helpers and deterministic benchmark scenarios."""

from adaptive_reservoir.benchmarks.common import BenchmarkResult
from adaptive_reservoir.benchmarks.concept_drift import run_concept_drift_benchmark
from adaptive_reservoir.benchmarks.delayed_xor import run_delayed_xor_benchmark
from adaptive_reservoir.benchmarks.reports import (
    REPORT_COLUMNS,
    format_csv_report,
    format_json_report,
    format_markdown_report,
    result_to_report_row,
    results_to_report_rows,
)
from adaptive_reservoir.benchmarks.temporal_drift import run_temporal_drift_benchmark

__all__ = [
    "BenchmarkResult",
    "REPORT_COLUMNS",
    "format_csv_report",
    "format_json_report",
    "format_markdown_report",
    "result_to_report_row",
    "results_to_report_rows",
    "run_concept_drift_benchmark",
    "run_delayed_xor_benchmark",
    "run_temporal_drift_benchmark",
]
