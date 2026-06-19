"""Experimental RLS sweep benchmark helpers."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from io import StringIO
from typing import cast

import numpy as np

from adaptive_reservoir import AdaptiveReservoir, ReadoutConfig, ReservoirConfig
from adaptive_reservoir.benchmarks.common import (
    calculate_adapt_steps,
    readout_sparsity,
    regression_score,
    rolling_regression_scores,
)
from adaptive_reservoir.core.config import (
    FEATURE_MODES,
    TOPOLOGY_NAMES,
    FeatureMode,
    TopologyName,
)
from adaptive_reservoir.experimental.rls import RLS_READOUT_NAME, RLSReadout
from adaptive_reservoir.features import extract_features

BENCHMARK_NAME = "rls-sweep"
BASE_BENCHMARK_NAME = "concept-drift"
MODEL_NAME = "adaptive_reservoir_experimental_rls"

DEFAULT_LAMBDAS = (1.0, 0.99)
DEFAULT_COVARIANCE_SCALES = (100.0, 1_000.0)
DEFAULT_FEATURE_MODES = ("state_slow_raw", "multi_raw")
DEFAULT_TOPOLOGIES = ("ring_shortcuts", "modular_small_world")

RLS_SWEEP_REPORT_COLUMNS = (
    "benchmark",
    "base_benchmark",
    "model",
    "topology",
    "feature_mode",
    "readout",
    "lambda",
    "covariance_scale",
    "seed",
    "pre_score",
    "post_score",
    "final_score",
    "adapt_steps",
    "us_per_sample",
    "saturation_rate",
    "readout_sparsity",
    "samples_seen",
)

_OUTPUT_FORMATS = frozenset(("csv", "markdown", "json"))


@dataclass(frozen=True, slots=True, kw_only=True)
class RLSSweepResult:
    """Aggregate result for one deterministic experimental RLS sweep run."""

    benchmark: str
    base_benchmark: str
    model: str
    topology: str
    feature_mode: str
    readout: str
    forgetting_factor: float
    covariance_scale: float
    seed: int
    pre_score: float
    post_score: float
    final_score: float
    adapt_steps: int | None
    us_per_sample: float
    saturation_rate: float
    readout_sparsity: float | None
    samples_seen: int

    def __post_init__(self) -> None:
        _validate_non_empty_string("benchmark", self.benchmark)
        _validate_non_empty_string("base_benchmark", self.base_benchmark)
        _validate_non_empty_string("model", self.model)
        _validate_non_empty_string("topology", self.topology)
        _validate_non_empty_string("feature_mode", self.feature_mode)
        _validate_non_empty_string("readout", self.readout)
        _validate_forgetting_factor(self.forgetting_factor)
        _validate_positive_finite("covariance_scale", self.covariance_scale)
        _validate_int("seed", self.seed)
        _validate_score("pre_score", self.pre_score)
        _validate_score("post_score", self.post_score)
        _validate_score("final_score", self.final_score)
        _validate_optional_non_negative_int("adapt_steps", self.adapt_steps)
        _validate_non_negative_finite("us_per_sample", self.us_per_sample)
        _validate_score("saturation_rate", self.saturation_rate)
        _validate_optional_score("readout_sparsity", self.readout_sparsity)
        _validate_non_negative_int("samples_seen", self.samples_seen)

    def to_row(self) -> dict[str, object]:
        """Return a stable CSV/Markdown/JSON-friendly sweep result row."""

        return {
            "benchmark": self.benchmark,
            "base_benchmark": self.base_benchmark,
            "model": self.model,
            "topology": self.topology,
            "feature_mode": self.feature_mode,
            "readout": self.readout,
            "lambda": self.forgetting_factor,
            "covariance_scale": self.covariance_scale,
            "seed": self.seed,
            "pre_score": self.pre_score,
            "post_score": self.post_score,
            "final_score": self.final_score,
            "adapt_steps": self.adapt_steps,
            "us_per_sample": self.us_per_sample,
            "saturation_rate": self.saturation_rate,
            "readout_sparsity": self.readout_sparsity,
            "samples_seen": self.samples_seen,
        }


def run_rls_sweep(
    *,
    seed: int = 0,
    n_samples: int = 900,
    drift_at: int = 450,
    score_window: int = 64,
    input_dim: int = 2,
    n_cells: int = 64,
    forgetting_factors: Sequence[float] = DEFAULT_LAMBDAS,
    covariance_scales: Sequence[float] = DEFAULT_COVARIANCE_SCALES,
    feature_modes: Sequence[str] = DEFAULT_FEATURE_MODES,
    topologies: Sequence[str] = DEFAULT_TOPOLOGIES,
) -> tuple[RLSSweepResult, ...]:
    """Run a deterministic concept-drift sweep over experimental RLS settings."""

    _validate_sweep_args(
        seed=seed,
        n_samples=n_samples,
        drift_at=drift_at,
        score_window=score_window,
        input_dim=input_dim,
        n_cells=n_cells,
    )
    lambdas = _validate_float_grid(
        "lambda",
        forgetting_factors,
        validator=_validate_forgetting_factor,
    )
    scales = _validate_float_grid(
        "covariance_scale",
        covariance_scales,
        validator=lambda value: _validate_positive_finite("covariance_scale", value),
    )
    modes = _validate_choice_grid("feature_mode", feature_modes, allowed=FEATURE_MODES)
    topology_values = _validate_choice_grid("topology", topologies, allowed=TOPOLOGY_NAMES)

    results: list[RLSSweepResult] = []
    for topology in topology_values:
        for feature_mode in modes:
            for forgetting_factor in lambdas:
                for covariance_scale in scales:
                    results.append(
                        _run_single_rls_sweep_case(
                            seed=seed,
                            n_samples=n_samples,
                            drift_at=drift_at,
                            score_window=score_window,
                            input_dim=input_dim,
                            n_cells=n_cells,
                            topology=topology,
                            feature_mode=feature_mode,
                            forgetting_factor=forgetting_factor,
                            covariance_scale=covariance_scale,
                        )
                    )
    return tuple(results)


def format_rls_sweep_output(
    results: Sequence[RLSSweepResult],
    *,
    output_format: str,
) -> str:
    """Format experimental RLS sweep results as csv, markdown, or json."""

    _validate_rls_sweep_results(results)
    if output_format == "csv":
        return format_rls_sweep_csv_report(results)
    if output_format == "markdown":
        return format_rls_sweep_markdown_report(results)
    if output_format == "json":
        return format_rls_sweep_json_report(results)
    supported = ", ".join(sorted(_OUTPUT_FORMATS))
    msg = f"unsupported rls-sweep output format: {output_format}; use one of: {supported}"
    raise ValueError(msg)


def format_rls_sweep_csv_report(results: Sequence[RLSSweepResult]) -> str:
    """Format experimental RLS sweep results as CSV."""

    _validate_rls_sweep_results(results)
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(RLS_SWEEP_REPORT_COLUMNS)
    for row in _rls_sweep_rows(results):
        writer.writerow(
            [_format_report_value(row[column]) for column in RLS_SWEEP_REPORT_COLUMNS]
        )
    return output.getvalue()


def format_rls_sweep_markdown_report(results: Sequence[RLSSweepResult]) -> str:
    """Format experimental RLS sweep results as a Markdown table."""

    _validate_rls_sweep_results(results)
    lines = [
        "| " + " | ".join(RLS_SWEEP_REPORT_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in RLS_SWEEP_REPORT_COLUMNS) + " |",
    ]
    for row in _rls_sweep_rows(results):
        cells = [_markdown_cell(row[column]) for column in RLS_SWEEP_REPORT_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_rls_sweep_json_report(results: Sequence[RLSSweepResult]) -> str:
    """Format experimental RLS sweep results as JSON."""

    _validate_rls_sweep_results(results)
    return json.dumps(list(_rls_sweep_rows(results)), indent=2, allow_nan=False)


def _run_single_rls_sweep_case(
    *,
    seed: int,
    n_samples: int,
    drift_at: int,
    score_window: int,
    input_dim: int,
    n_cells: int,
    topology: str,
    feature_mode: str,
    forgetting_factor: float,
    covariance_scale: float,
) -> RLSSweepResult:
    config = ReservoirConfig(
        input_dim=input_dim,
        n_cells=n_cells,
        topology=cast(TopologyName, topology),
        feature_mode=cast(FeatureMode, feature_mode),
        seed=seed,
        dtype="float64",
        readout=ReadoutConfig(name="sliding_ridge", update_interval=1),
    )
    model = AdaptiveReservoir(config)
    _install_experimental_rls_readout(
        model,
        forgetting_factor=forgetting_factor,
        covariance_scale=covariance_scale,
    )

    predictions: list[float] = []
    targets: list[float] = []
    for sample, target in _generate_concept_drift_stream(
        seed=seed,
        n_samples=n_samples,
        drift_at=drift_at,
        input_dim=input_dim,
    ):
        result = model.step(sample, target=target)
        predictions.append(result.prediction)
        targets.append(target)

    pre_score = regression_score(
        predictions[drift_at - score_window : drift_at],
        targets[drift_at - score_window : drift_at],
    )
    post_score = regression_score(
        predictions[drift_at : drift_at + score_window],
        targets[drift_at : drift_at + score_window],
    )
    final_score = regression_score(predictions[-score_window:], targets[-score_window:])
    rolling_scores = rolling_regression_scores(
        predictions,
        targets,
        window=score_window,
    )
    recovery_search_start = drift_at + score_window - 1
    adapt_steps = calculate_adapt_steps(
        rolling_scores,
        start_index=recovery_search_start,
        threshold=max(0.75, pre_score * 0.9),
        origin_index=drift_at,
    )
    metrics = model.metrics_snapshot()

    return RLSSweepResult(
        benchmark=BENCHMARK_NAME,
        base_benchmark=BASE_BENCHMARK_NAME,
        model=MODEL_NAME,
        topology=topology,
        feature_mode=feature_mode,
        readout=RLS_READOUT_NAME,
        forgetting_factor=forgetting_factor,
        covariance_scale=covariance_scale,
        seed=seed,
        pre_score=pre_score,
        post_score=post_score,
        final_score=final_score,
        adapt_steps=adapt_steps,
        us_per_sample=metrics.us_per_sample_avg,
        saturation_rate=metrics.saturation_rate_avg,
        readout_sparsity=readout_sparsity(model),
        samples_seen=metrics.samples_seen,
    )


def _install_experimental_rls_readout(
    model: AdaptiveReservoir,
    *,
    forgetting_factor: float,
    covariance_scale: float,
) -> None:
    features = extract_features(model._core.state, model.config.feature_mode)
    model._readout = RLSReadout(
        feature_dim=int(features.size),
        forgetting_factor=forgetting_factor,
        covariance_scale=covariance_scale,
        dtype=model.config.dtype,
    )


def _generate_concept_drift_stream(
    *,
    seed: int,
    n_samples: int,
    drift_at: int,
    input_dim: int,
) -> Iterator[tuple[tuple[float, ...], float]]:
    rng = np.random.default_rng(seed)
    for index in range(n_samples):
        sample = rng.uniform(-1.0, 1.0, size=input_dim)
        target = (
            _target_before_drift(sample)
            if index < drift_at
            else _target_after_drift(sample)
        )
        yield tuple(float(value) for value in sample), target


def _target_before_drift(sample: np.ndarray) -> float:
    return float(np.tanh(0.8 * sample[0] + 0.2 * sample[1]))


def _target_after_drift(sample: np.ndarray) -> float:
    return float(np.tanh(-0.8 * sample[0] + 0.2 * sample[1]))


def _validate_sweep_args(
    *,
    seed: int,
    n_samples: int,
    drift_at: int,
    score_window: int,
    input_dim: int,
    n_cells: int,
) -> None:
    _validate_int("seed", seed)
    _validate_positive_int("n_samples", n_samples)
    _validate_positive_int("score_window", score_window)
    _validate_positive_int("input_dim", input_dim)
    _validate_positive_int("n_cells", n_cells)
    if input_dim < 2:
        msg = "input_dim must be at least 2"
        raise ValueError(msg)
    if drift_at <= score_window:
        msg = "drift_at must leave room for the pre-drift score window"
        raise ValueError(msg)
    if drift_at + score_window > n_samples:
        msg = "drift_at must leave room for the post-drift score window"
        raise ValueError(msg)


def _validate_float_grid(
    name: str,
    values: Sequence[float],
    *,
    validator: Callable[[float], float],
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        msg = f"{name} grid must be a numeric sequence"
        raise ValueError(msg)
    if not values:
        msg = f"{name} grid must not be empty"
        raise ValueError(msg)
    return tuple(validator(value) for value in values)


def _validate_choice_grid(
    name: str,
    values: Sequence[str],
    *,
    allowed: frozenset[str],
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        msg = f"{name} grid must be a sequence of strings"
        raise ValueError(msg)
    if not values:
        msg = f"{name} grid must not be empty"
        raise ValueError(msg)
    normalized = tuple(value.strip() for value in values)
    for value in normalized:
        if value not in allowed:
            allowed_values = ", ".join(sorted(allowed))
            msg = f"{name} must be one of: {allowed_values}; got {value!r}"
            raise ValueError(msg)
    return normalized


def _validate_rls_sweep_results(results: Sequence[RLSSweepResult]) -> None:
    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        msg = "results must be a sequence of RLSSweepResult values"
        raise ValueError(msg)
    if not results:
        msg = "results must not be empty"
        raise ValueError(msg)
    for result in results:
        if not isinstance(result, RLSSweepResult):
            msg = "results must contain only RLSSweepResult values"
            raise ValueError(msg)


def _rls_sweep_rows(results: Sequence[RLSSweepResult]) -> tuple[dict[str, object], ...]:
    return tuple(result.to_row() for result in results)


def _format_report_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_cell(value: object) -> str:
    return _format_report_value(value).replace("\n", " ").replace("|", "\\|")


def _validate_non_empty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{name} must be a non-empty string"
        raise ValueError(msg)


def _validate_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer"
        raise ValueError(msg)


def _validate_positive_int(name: str, value: int) -> None:
    _validate_int(name, value)
    if value <= 0:
        msg = f"{name} must be positive"
        raise ValueError(msg)


def _validate_non_negative_int(name: str, value: int) -> None:
    _validate_int(name, value)
    if value < 0:
        msg = f"{name} must be non-negative"
        raise ValueError(msg)


def _validate_optional_non_negative_int(name: str, value: int | None) -> None:
    if value is None:
        return
    _validate_non_negative_int(name, value)


def _validate_forgetting_factor(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0 or result > 1.0:
        msg = "lambda must be finite and in the interval (0, 1]"
        raise ValueError(msg)
    return result


def _validate_positive_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        msg = f"{name} must be finite and positive"
        raise ValueError(msg)
    return result


def _validate_non_negative_finite(name: str, value: float) -> None:
    if not isinstance(value, float) or not math.isfinite(value) or value < 0.0:
        msg = f"{name} must be finite and non-negative"
        raise ValueError(msg)


def _validate_score(name: str, value: float) -> None:
    if not isinstance(value, float) or not math.isfinite(value) or value < 0.0 or value > 1.0:
        msg = f"{name} must be a finite score in [0.0, 1.0]"
        raise ValueError(msg)


def _validate_optional_score(name: str, value: float | None) -> None:
    if value is None:
        return
    _validate_score(name, value)


__all__ = [
    "BASE_BENCHMARK_NAME",
    "BENCHMARK_NAME",
    "DEFAULT_COVARIANCE_SCALES",
    "DEFAULT_FEATURE_MODES",
    "DEFAULT_LAMBDAS",
    "DEFAULT_TOPOLOGIES",
    "MODEL_NAME",
    "RLS_SWEEP_REPORT_COLUMNS",
    "RLSSweepResult",
    "format_rls_sweep_csv_report",
    "format_rls_sweep_json_report",
    "format_rls_sweep_markdown_report",
    "format_rls_sweep_output",
    "run_rls_sweep",
]
