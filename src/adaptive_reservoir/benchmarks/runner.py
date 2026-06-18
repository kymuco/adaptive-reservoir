"""Command-line runner for built-in adaptive reservoir benchmarks."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from adaptive_reservoir import ReadoutConfig, ReservoirConfig
from adaptive_reservoir.benchmarks.common import BenchmarkResult
from adaptive_reservoir.benchmarks.concept_drift import run_concept_drift_benchmark
from adaptive_reservoir.benchmarks.delayed_xor import run_delayed_xor_benchmark
from adaptive_reservoir.benchmarks.temporal_drift import run_temporal_drift_benchmark

BenchmarkRunner = Callable[..., BenchmarkResult]

BENCHMARKS: dict[str, BenchmarkRunner] = {
    "concept-drift": run_concept_drift_benchmark,
    "temporal-drift": run_temporal_drift_benchmark,
    "delayed-xor": run_delayed_xor_benchmark,
}

_BENCHMARK_SPECIFIC_OPTIONS: dict[str, dict[str, str]] = {
    "concept-drift": {"drift_at": "--drift-at"},
    "temporal-drift": {
        "drift_at": "--drift-at",
        "delay_before": "--delay-before",
        "delay_after": "--delay-after",
    },
    "delayed-xor": {"delay_a": "--delay-a", "delay_b": "--delay-b"},
}


class _BenchmarkArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"error: {message}\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_benchmark_from_args(args)
    except (TypeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(format_result(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark runner argument parser."""

    parser = _BenchmarkArgumentParser(
        prog="adaptive-reservoir-bench",
        description="Run built-in adaptive-reservoir benchmarks.",
    )
    parser.add_argument(
        "benchmark",
        help="benchmark name: concept-drift, temporal-drift, or delayed-xor",
    )
    parser.add_argument("--seed", type=int, default=0, help="benchmark and model seed")
    parser.add_argument("--samples", type=int, default=None, help="number of benchmark samples")
    parser.add_argument(
        "--score-window",
        type=int,
        default=None,
        help="score/evaluation window size",
    )
    parser.add_argument("--cells", type=int, default=None, help="reservoir cell count")
    parser.add_argument("--input-dim", type=int, default=None, help="input vector dimension")
    parser.add_argument("--topology", default=None, help="reservoir topology")
    parser.add_argument("--feature-mode", default=None, help="reservoir feature mode")
    parser.add_argument("--readout", default=None, help="readout name")
    parser.add_argument(
        "--drift-at",
        type=int,
        default=None,
        help="drift sample index for drift benchmarks",
    )
    parser.add_argument(
        "--delay-before",
        type=int,
        default=None,
        help="pre-drift delay for temporal-drift",
    )
    parser.add_argument(
        "--delay-after",
        type=int,
        default=None,
        help="post-drift delay for temporal-drift",
    )
    parser.add_argument("--delay-a", type=int, default=None, help="first delayed XOR lag")
    parser.add_argument("--delay-b", type=int, default=None, help="second delayed XOR lag")
    return parser


def run_benchmark_from_args(args: argparse.Namespace) -> BenchmarkResult:
    """Run the selected benchmark from parsed CLI arguments."""

    benchmark_name = normalize_benchmark_name(args.benchmark)
    if benchmark_name not in BENCHMARKS:
        supported = ", ".join(sorted(BENCHMARKS))
        msg = f"unknown benchmark {args.benchmark!r}; supported benchmarks: {supported}"
        raise ValueError(msg)
    _reject_irrelevant_options(args, benchmark_name=benchmark_name)

    config = build_config_from_args(args, benchmark_name=benchmark_name)
    kwargs = _common_kwargs(args)
    if benchmark_name == "concept-drift":
        _set_optional(kwargs, "drift_at", args.drift_at)
        return run_concept_drift_benchmark(config, **kwargs)
    if benchmark_name == "temporal-drift":
        _set_optional(kwargs, "drift_at", args.drift_at)
        _set_optional(kwargs, "delay_before", args.delay_before)
        _set_optional(kwargs, "delay_after", args.delay_after)
        return run_temporal_drift_benchmark(config, **kwargs)
    _set_optional(kwargs, "delay_a", args.delay_a)
    _set_optional(kwargs, "delay_b", args.delay_b)
    return run_delayed_xor_benchmark(config, **kwargs)


def build_config_from_args(
    args: argparse.Namespace,
    *,
    benchmark_name: str,
) -> ReservoirConfig:
    """Build a reservoir config from CLI options."""

    return ReservoirConfig(
        input_dim=_int_or_default(args.input_dim, _default_input_dim(benchmark_name)),
        n_cells=_int_or_default(args.cells, _default_cells(benchmark_name)),
        topology=args.topology or _default_topology(benchmark_name),
        feature_mode=args.feature_mode or _default_feature_mode(benchmark_name),
        seed=args.seed,
        readout=ReadoutConfig(name=args.readout or "sliding_ridge", update_interval=1),
    )


def format_result(result: BenchmarkResult) -> str:
    """Format a benchmark result as stable key-value text."""

    return "\n".join(
        f"{key}: {_format_value(value)}" for key, value in result.to_row().items()
    )


def normalize_benchmark_name(value: str) -> str:
    """Normalize CLI benchmark names to canonical dash-separated names."""

    return value.strip().lower().replace("_", "-")


def _common_kwargs(args: argparse.Namespace) -> dict[str, int]:
    kwargs = {"seed": args.seed}
    _set_optional(kwargs, "n_samples", args.samples)
    _set_optional(kwargs, "score_window", args.score_window)
    return kwargs


def _set_optional(kwargs: dict[str, int], key: str, value: int | None) -> None:
    if value is not None:
        kwargs[key] = value


def _reject_irrelevant_options(args: argparse.Namespace, *, benchmark_name: str) -> None:
    allowed = _BENCHMARK_SPECIFIC_OPTIONS[benchmark_name]
    rejected: list[str] = []
    for option_name, cli_flag in _all_benchmark_specific_options().items():
        if option_name in allowed:
            continue
        if getattr(args, option_name) is not None:
            rejected.append(cli_flag)
    if rejected:
        flags = ", ".join(rejected)
        msg = f"{flags} is not supported for {benchmark_name}"
        raise ValueError(msg)


def _all_benchmark_specific_options() -> dict[str, str]:
    options: dict[str, str] = {}
    for benchmark_options in _BENCHMARK_SPECIFIC_OPTIONS.values():
        options.update(benchmark_options)
    return options


def _int_or_default(value: int | None, default: int) -> int:
    if value is None:
        return default
    return value


def _default_input_dim(benchmark_name: str) -> int:
    if benchmark_name == "delayed-xor":
        return 1
    return 2


def _default_cells(benchmark_name: str) -> int:
    if benchmark_name == "concept-drift":
        return 64
    return 96


def _default_topology(benchmark_name: str) -> str:
    if benchmark_name == "delayed-xor":
        return "modular_small_world"
    return "ring_shortcuts"


def _default_feature_mode(benchmark_name: str) -> str:
    if benchmark_name == "concept-drift":
        return "state_slow_raw"
    return "multi_raw"


def _format_value(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


__all__ = [
    "BENCHMARKS",
    "build_config_from_args",
    "build_parser",
    "format_result",
    "main",
    "normalize_benchmark_name",
    "run_benchmark_from_args",
]
