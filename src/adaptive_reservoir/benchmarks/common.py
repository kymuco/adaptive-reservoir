"""Common benchmark result and metric helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkResult:
    """Aggregate result for one deterministic benchmark run."""

    benchmark: str
    model: str
    topology: str
    feature_mode: str
    readout: str
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
        _validate_non_empty_string("model", self.model)
        _validate_non_empty_string("topology", self.topology)
        _validate_non_empty_string("feature_mode", self.feature_mode)
        _validate_non_empty_string("readout", self.readout)
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
        """Return a stable JSON/CSV/Markdown-friendly result row."""

        return {
            "benchmark": self.benchmark,
            "model": self.model,
            "topology": self.topology,
            "feature_mode": self.feature_mode,
            "readout": self.readout,
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


def regression_score(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return a bounded regression score where higher is better."""

    prediction_array = _finite_vector(predictions, "predictions")
    target_array = _finite_vector(targets, "targets")
    if prediction_array.size != target_array.size:
        msg = "predictions and targets must have the same length"
        raise ValueError(msg)
    if prediction_array.size == 0:
        msg = "predictions and targets must not be empty"
        raise ValueError(msg)
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        msg = "epsilon must be finite and positive"
        raise ValueError(msg)

    errors = prediction_array - target_array
    mse = float(np.mean(errors * errors))
    centered_targets = target_array - float(np.mean(target_array))
    baseline_mse = float(np.mean(centered_targets * centered_targets))
    if baseline_mse <= epsilon:
        baseline_mse = float(np.mean(target_array * target_array))
    denominator = max(baseline_mse, epsilon)
    return _clip01(1.0 - mse / denominator)


def rolling_regression_scores(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    window: int,
) -> tuple[float | None, ...]:
    """Return trailing-window regression scores aligned to sample indices."""

    if window <= 0:
        msg = "window must be positive"
        raise ValueError(msg)
    prediction_array = _finite_vector(predictions, "predictions")
    target_array = _finite_vector(targets, "targets")
    if prediction_array.size != target_array.size:
        msg = "predictions and targets must have the same length"
        raise ValueError(msg)
    scores: list[float | None] = []
    for index in range(prediction_array.size):
        if index + 1 < window:
            scores.append(None)
            continue
        start = index + 1 - window
        scores.append(
            regression_score(
                prediction_array[start : index + 1],
                target_array[start : index + 1],
            )
        )
    return tuple(scores)


def calculate_adapt_steps(
    scores: Sequence[float | None],
    *,
    start_index: int,
    threshold: float,
    origin_index: int | None = None,
) -> int | None:
    """Return steps from origin until score reaches threshold."""

    if start_index < 0:
        msg = "start_index must be non-negative"
        raise ValueError(msg)
    origin = start_index if origin_index is None else origin_index
    if origin < 0 or origin > start_index:
        msg = "origin_index must be between zero and start_index"
        raise ValueError(msg)
    _validate_score("threshold", threshold)
    for index in range(start_index, len(scores)):
        score = scores[index]
        if score is not None and score >= threshold:
            return index - origin + 1
    return None


def readout_sparsity(model: object, *, epsilon: float = 1e-8) -> float | None:
    """Return fraction of near-zero readout weights when available."""

    if epsilon < 0.0 or not math.isfinite(epsilon):
        msg = "epsilon must be finite and non-negative"
        raise ValueError(msg)
    readout = getattr(model, "_readout", None)
    if readout is None or not hasattr(readout, "weights"):
        return None
    try:
        weights = getattr(readout, "weights")
    except Exception:
        return None
    array = np.asarray(weights, dtype=np.float64)
    if array.size == 0:
        return None
    if not np.all(np.isfinite(array)):
        return None
    return float(np.mean(np.abs(array) <= epsilon))


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        msg = f"{name} must be a numeric sequence"
        raise ValueError(msg)
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must contain only numeric values"
        raise ValueError(msg) from exc
    if array.ndim != 1:
        msg = f"{name} must be a 1D sequence"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    return array


def _validate_non_empty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = f"{name} must be a non-empty string"
        raise ValueError(msg)


def _validate_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer"
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


def _validate_score(name: str, value: float) -> None:
    if not isinstance(value, float) or not math.isfinite(value) or value < 0.0 or value > 1.0:
        msg = f"{name} must be a finite score in [0.0, 1.0]"
        raise ValueError(msg)


def _validate_optional_score(name: str, value: float | None) -> None:
    if value is None:
        return
    _validate_score(name, value)


def _validate_non_negative_finite(name: str, value: float) -> None:
    if not isinstance(value, float) or not math.isfinite(value) or value < 0.0:
        msg = f"{name} must be finite and non-negative"
        raise ValueError(msg)


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        msg = "score value must be finite"
        raise ValueError(msg)
    return min(1.0, max(0.0, value))


__all__ = [
    "BenchmarkResult",
    "calculate_adapt_steps",
    "readout_sparsity",
    "regression_score",
    "rolling_regression_scores",
]
