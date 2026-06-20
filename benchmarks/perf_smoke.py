"""Lightweight performance smoke benchmark for adaptive-reservoir.

This developer script tracks wall-clock microseconds per sample across PRs.
It is intentionally not a CI performance gate and does not define thresholds.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))

from adaptive_reservoir import AdaptiveReservoir, ReadoutConfig, ReservoirConfig  # noqa: E402
from adaptive_reservoir.core.reservoir import ReservoirCore  # noqa: E402

InputStream = tuple[tuple[float, ...], ...]
TargetStream = tuple[float, ...]
MeasuredStep = Callable[[tuple[float, ...], float], None]

DEFAULT_SEED = 0
DEFAULT_SAMPLES = 5_000
DEFAULT_WARMUP = 500
DEFAULT_CELLS = 64
DEFAULT_INPUT_DIM = 2
_SCENARIO_NAMES = (
    "core_ring_shortcuts",
    "core_random_sparse",
    "model_ring_sliding_ridge",
    "model_ring_replay_ridge",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PerfSmokeResult:
    """One wall-clock performance smoke measurement."""

    scenario: str
    samples: int
    warmup: int
    seed: int
    n_cells: int
    input_dim: int
    us_per_sample: float

    def __post_init__(self) -> None:
        _validate_non_empty_string("scenario", self.scenario)
        _validate_positive_int("samples", self.samples)
        _validate_non_negative_int("warmup", self.warmup)
        _validate_min_int("n_cells", self.n_cells, minimum=2)
        _validate_min_int("input_dim", self.input_dim, minimum=2)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            msg = "seed must be an integer"
            raise ValueError(msg)
        if not isinstance(self.us_per_sample, float) or not math.isfinite(self.us_per_sample):
            msg = "us_per_sample must be finite"
            raise ValueError(msg)
        if self.us_per_sample <= 0.0:
            msg = "us_per_sample must be positive"
            raise ValueError(msg)

    def to_row(self) -> dict[str, object]:
        """Return a stable table-friendly result row."""

        return {
            "scenario": self.scenario,
            "samples": self.samples,
            "warmup": self.warmup,
            "seed": self.seed,
            "n_cells": self.n_cells,
            "input_dim": self.input_dim,
            "us_per_sample": self.us_per_sample,
        }


def run_perf_smoke(
    *,
    seed: int = DEFAULT_SEED,
    samples: int = DEFAULT_SAMPLES,
    warmup: int = DEFAULT_WARMUP,
    n_cells: int = DEFAULT_CELLS,
    input_dim: int = DEFAULT_INPUT_DIM,
) -> tuple[PerfSmokeResult, ...]:
    """Run deterministic wall-clock smoke scenarios and return timing rows."""

    _validate_positive_int("samples", samples)
    _validate_non_negative_int("warmup", warmup)
    _validate_min_int("n_cells", n_cells, minimum=2)
    _validate_min_int("input_dim", input_dim, minimum=2)
    if not isinstance(seed, int) or isinstance(seed, bool):
        msg = "seed must be an integer"
        raise ValueError(msg)

    inputs = _generate_inputs(seed=seed, samples=samples + warmup, input_dim=input_dim)
    targets = tuple(_target(sample) for sample in inputs)
    scenarios = (
        _core_ring_shortcuts_step(seed=seed, n_cells=n_cells, input_dim=input_dim),
        _core_random_sparse_step(seed=seed, n_cells=n_cells, input_dim=input_dim),
        _model_ring_step(
            seed=seed,
            n_cells=n_cells,
            input_dim=input_dim,
            readout=ReadoutConfig(name="sliding_ridge", update_interval=1),
        ),
        _model_ring_step(
            seed=seed,
            n_cells=n_cells,
            input_dim=input_dim,
            readout=ReadoutConfig(name="replay_ridge", update_interval=1),
        ),
    )

    return tuple(
        PerfSmokeResult(
            scenario=scenario_name,
            samples=samples,
            warmup=warmup,
            seed=seed,
            n_cells=n_cells,
            input_dim=input_dim,
            us_per_sample=_measure_us_per_sample(
                inputs=inputs,
                targets=targets,
                warmup=warmup,
                step=scenario_step,
            ),
        )
        for scenario_name, scenario_step in zip(_SCENARIO_NAMES, scenarios, strict=True)
    )


def format_markdown(results: Sequence[PerfSmokeResult]) -> str:
    """Format performance smoke results as a stable Markdown table."""

    if not results:
        msg = "results must not be empty"
        raise ValueError(msg)
    lines = [
        "| scenario | samples | warmup | seed | n_cells | input_dim | us_per_sample |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.scenario} | "
            f"{result.samples} | "
            f"{result.warmup} | "
            f"{result.seed} | "
            f"{result.n_cells} | "
            f"{result.input_dim} | "
            f"{result.us_per_sample:.4f} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the performance smoke CLI and return a process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        results = run_perf_smoke(
            seed=args.seed,
            samples=args.samples,
            warmup=args.warmup,
            n_cells=args.cells,
            input_dim=args.input_dim,
        )
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(format_markdown(results))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="perf_smoke.py",
        description="Track adaptive-reservoir wall-clock microseconds per sample.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="stream/model seed")
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="measured samples per scenario",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="unmeasured warmup samples per scenario",
    )
    parser.add_argument("--cells", type=int, default=DEFAULT_CELLS, help="reservoir cells")
    parser.add_argument(
        "--input-dim",
        type=int,
        default=DEFAULT_INPUT_DIM,
        help="input vector dimension",
    )
    return parser


def _core_ring_shortcuts_step(
    *,
    seed: int,
    n_cells: int,
    input_dim: int,
) -> MeasuredStep:
    core = ReservoirCore.from_config(
        ReservoirConfig(
            input_dim=input_dim,
            n_cells=n_cells,
            topology="ring_shortcuts",
            feature_mode="state_slow_raw",
            seed=seed,
        )
    )

    def step(sample: tuple[float, ...], _target_value: float) -> None:
        core.step(sample)

    return step


def _core_random_sparse_step(
    *,
    seed: int,
    n_cells: int,
    input_dim: int,
) -> MeasuredStep:
    core = ReservoirCore.from_config(
        ReservoirConfig(
            input_dim=input_dim,
            n_cells=n_cells,
            topology="random_sparse",
            feature_mode="state_slow_raw",
            seed=seed,
        )
    )

    def step(sample: tuple[float, ...], _target_value: float) -> None:
        core.step(sample)

    return step


def _model_ring_step(
    *,
    seed: int,
    n_cells: int,
    input_dim: int,
    readout: ReadoutConfig,
) -> MeasuredStep:
    model = AdaptiveReservoir(
        ReservoirConfig(
            input_dim=input_dim,
            n_cells=n_cells,
            topology="ring_shortcuts",
            feature_mode="state_slow_raw",
            seed=seed,
            readout=readout,
        )
    )

    def step(sample: tuple[float, ...], target_value: float) -> None:
        model.step(sample, target=target_value)

    return step


def _measure_us_per_sample(
    *,
    inputs: InputStream,
    targets: TargetStream,
    warmup: int,
    step: MeasuredStep,
) -> float:
    for sample, target in zip(inputs[:warmup], targets[:warmup], strict=True):
        step(sample, target)
    measured_inputs = inputs[warmup:]
    measured_targets = targets[warmup:]
    start = perf_counter()
    for sample, target in zip(measured_inputs, measured_targets, strict=True):
        step(sample, target)
    elapsed_seconds = perf_counter() - start
    return float(elapsed_seconds * 1_000_000.0 / len(measured_inputs))


def _generate_inputs(*, seed: int, samples: int, input_dim: int) -> InputStream:
    rng = np.random.default_rng(seed)
    values = rng.uniform(-1.0, 1.0, size=(samples, input_dim))
    return tuple(tuple(float(value) for value in row) for row in values)


def _target(sample: Sequence[float]) -> float:
    return float(np.tanh(0.8 * sample[0] + 0.2 * sample[1]))


def _validate_non_empty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{name} must be a non-empty string"
        raise ValueError(msg)


def _validate_positive_int(name: str, value: int) -> None:
    _validate_min_int(name, value, minimum=1)


def _validate_non_negative_int(name: str, value: int) -> None:
    _validate_min_int(name, value, minimum=0)


def _validate_min_int(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        msg = f"{name} must be an integer >= {minimum}"
        raise ValueError(msg)


__all__ = [
    "DEFAULT_CELLS",
    "DEFAULT_INPUT_DIM",
    "DEFAULT_SAMPLES",
    "DEFAULT_SEED",
    "DEFAULT_WARMUP",
    "PerfSmokeResult",
    "format_markdown",
    "main",
    "run_perf_smoke",
]

if __name__ == "__main__":
    raise SystemExit(main())
