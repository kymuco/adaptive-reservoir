"""Stateful calculator for adaptive state channels."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from adaptive_reservoir.core.config import ChannelConfig
from adaptive_reservoir.core.result import AdaptiveChannels
from adaptive_reservoir.core.serialization import (
    nested_numeric_sequence_to_tuple,
    numeric_sequence_to_tuple,
    optional_int_or_none,
    optional_numeric_sequence_to_tuple,
    require_int,
    require_mapping,
)
from adaptive_reservoir.core.state import ReservoirState
from adaptive_reservoir.readout.base import FloatArray, validate_features, validate_target

CHANNEL_CALCULATOR_SNAPSHOT_SCHEMA_VERSION = 1
_NOVELTY_BASELINE_MULTIPLIER = 3.0
_MAX_FLOAT64 = float(np.finfo(np.float64).max)


@dataclass(frozen=True, slots=True, kw_only=True)
class ChannelCalculatorSnapshot:
    """Immutable numeric checkpoint for adaptive channel calculator state."""

    samples_seen: int
    feature_dim: int | None
    feature_window: tuple[tuple[float, ...], ...]
    activation_window: tuple[tuple[float, ...], ...]
    state_delta_window: tuple[float, ...]
    prediction_window: tuple[float, ...]
    prediction_error_window: tuple[float, ...]
    previous_activations: tuple[float, ...] | None
    schema_version: int = CHANNEL_CALCULATOR_SNAPSHOT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly channel snapshot dictionary."""

        return {
            "schema_version": self.schema_version,
            "samples_seen": self.samples_seen,
            "feature_dim": self.feature_dim,
            "feature_window": [list(row) for row in self.feature_window],
            "activation_window": [list(row) for row in self.activation_window],
            "state_delta_window": list(self.state_delta_window),
            "prediction_window": list(self.prediction_window),
            "prediction_error_window": list(self.prediction_error_window),
            "previous_activations": (
                None
                if self.previous_activations is None
                else list(self.previous_activations)
            ),
        }

    @classmethod
    def from_dict(cls, data: object) -> ChannelCalculatorSnapshot:
        """Create a channel snapshot from a JSON-friendly mapping."""

        mapping = require_mapping(data, "channels")
        schema_version = require_int(mapping, "schema_version")
        if schema_version != CHANNEL_CALCULATOR_SNAPSHOT_SCHEMA_VERSION:
            msg = f"unsupported channel snapshot schema_version: {schema_version}"
            raise ValueError(msg)
        if "feature_dim" not in mapping:
            msg = "missing required field: feature_dim"
            raise ValueError(msg)
        if "previous_activations" not in mapping:
            msg = "missing required field: previous_activations"
            raise ValueError(msg)
        return cls(
            schema_version=schema_version,
            samples_seen=require_int(mapping, "samples_seen"),
            feature_dim=optional_int_or_none(mapping, "feature_dim", None),
            feature_window=nested_numeric_sequence_to_tuple(
                mapping.get("feature_window"),
                "feature_window",
            ),
            activation_window=nested_numeric_sequence_to_tuple(
                mapping.get("activation_window"),
                "activation_window",
            ),
            state_delta_window=numeric_sequence_to_tuple(
                mapping.get("state_delta_window"),
                "state_delta_window",
            ),
            prediction_window=numeric_sequence_to_tuple(
                mapping.get("prediction_window"),
                "prediction_window",
            ),
            prediction_error_window=numeric_sequence_to_tuple(
                mapping.get("prediction_error_window"),
                "prediction_error_window",
            ),
            previous_activations=optional_numeric_sequence_to_tuple(
                mapping.get("previous_activations"),
                "previous_activations",
            ),
        )


class AdaptiveChannelCalculator:
    """Stateful calculator for bounded numeric adaptive channel signals."""

    def __init__(self, *, config: ChannelConfig, dtype: str = "float64") -> None:
        if not isinstance(config, ChannelConfig):
            msg = "config must be a ChannelConfig"
            raise TypeError(msg)
        self.config = config
        self.dtype = _validate_dtype(dtype)
        self._samples_seen = 0
        self._feature_dim: int | None = None
        self._feature_window: list[FloatArray] = []
        self._activation_window: list[FloatArray] = []
        self._state_delta_window: list[float] = []
        self._prediction_window: list[float] = []
        self._prediction_error_window: list[float] = []
        self._previous_activations: FloatArray | None = None

    @property
    def samples_seen(self) -> int:
        """Number of calculator updates observed."""

        return self._samples_seen

    @property
    def feature_count(self) -> int:
        """Number of feature vectors retained in the novelty history."""

        return len(self._feature_window)

    @property
    def state_count(self) -> int:
        """Number of activation vectors retained in the novelty history."""

        return len(self._activation_window)

    @property
    def state_delta_count(self) -> int:
        """Number of state deltas retained in the stability history."""

        return len(self._state_delta_window)

    @property
    def prediction_count(self) -> int:
        """Number of predictions retained in the stability history."""

        return len(self._prediction_window)

    @property
    def prediction_error_count(self) -> int:
        """Number of supervised prediction errors retained in drift history."""

        return len(self._prediction_error_window)

    @property
    def feature_dim(self) -> int | None:
        """Feature dimension observed by the calculator, if initialized."""

        return self._feature_dim

    def reset(self) -> None:
        """Clear numeric runtime history without changing configuration."""

        self._samples_seen = 0
        self._feature_dim = None
        self._feature_window.clear()
        self._activation_window.clear()
        self._state_delta_window.clear()
        self._prediction_window.clear()
        self._prediction_error_window.clear()
        self._previous_activations = None

    def snapshot(self) -> ChannelCalculatorSnapshot:
        """Return an immutable numeric checkpoint of channel runtime state."""

        return ChannelCalculatorSnapshot(
            samples_seen=self._samples_seen,
            feature_dim=self._feature_dim,
            feature_window=_vector_window_to_tuple(self._feature_window),
            activation_window=_vector_window_to_tuple(self._activation_window),
            state_delta_window=tuple(self._state_delta_window),
            prediction_window=tuple(self._prediction_window),
            prediction_error_window=tuple(self._prediction_error_window),
            previous_activations=_optional_vector_to_tuple(self._previous_activations),
        )

    def restore(self, snapshot: ChannelCalculatorSnapshot) -> None:
        """Restore numeric channel runtime state from a snapshot."""

        if not isinstance(snapshot, ChannelCalculatorSnapshot):
            msg = "snapshot must be a ChannelCalculatorSnapshot"
            raise TypeError(msg)
        if snapshot.schema_version != CHANNEL_CALCULATOR_SNAPSHOT_SCHEMA_VERSION:
            msg = f"unsupported channel snapshot schema_version: {snapshot.schema_version}"
            raise ValueError(msg)
        if snapshot.samples_seen < 0:
            msg = "channel snapshot samples_seen must be non-negative"
            raise ValueError(msg)

        feature_dim = _validate_optional_positive_int(
            "feature_dim",
            snapshot.feature_dim,
        )
        feature_window, restored_feature_dim = _restore_vector_window(
            snapshot.feature_window,
            name="feature_window",
            expected_dim=feature_dim,
            max_len=self.config.novelty_window,
            dtype=self.dtype,
        )
        if feature_dim is None:
            feature_dim = restored_feature_dim
        activation_window, activation_dim = _restore_vector_window(
            snapshot.activation_window,
            name="activation_window",
            expected_dim=None,
            max_len=self.config.novelty_window,
            dtype=self.dtype,
        )
        previous_activations = _restore_optional_vector(
            snapshot.previous_activations,
            name="previous_activations",
            expected_dim=activation_dim,
            dtype=self.dtype,
        )
        state_delta_window = _restore_float_window(
            snapshot.state_delta_window,
            name="state_delta_window",
            max_len=self.config.stability_window,
        )
        prediction_window = _restore_float_window(
            snapshot.prediction_window,
            name="prediction_window",
            max_len=self.config.stability_window,
        )
        prediction_error_window = _restore_float_window(
            snapshot.prediction_error_window,
            name="prediction_error_window",
            max_len=self.config.drift_window,
        )

        self._samples_seen = snapshot.samples_seen
        self._feature_dim = feature_dim
        self._feature_window = feature_window
        self._activation_window = activation_window
        self._state_delta_window = state_delta_window
        self._prediction_window = prediction_window
        self._prediction_error_window = prediction_error_window
        self._previous_activations = previous_activations

    def update(
        self,
        input: object,
        state: ReservoirState,
        features: object,
        prediction: object | None = None,
        target: object | None = None,
    ) -> AdaptiveChannels:
        """Update numeric channel histories and return bounded channels.

        The raw ``input`` value is accepted to preserve the channel-calculator
        contract, but it is intentionally not stored by this calculator.
        Missing predictions are treated as unavailable bootstrapping signals.
        """

        del input
        if not isinstance(state, ReservoirState):
            msg = "state must be a ReservoirState"
            raise TypeError(msg)
        feature_vector = self._validate_features(features)
        activations = self._validate_activations(state)
        prediction_value = None if prediction is None else _validate_prediction(prediction)
        target_value = None if target is None else validate_target(target)
        novelty = self._calculate_novelty(
            features=feature_vector,
            activations=activations,
        )
        saturation = self._calculate_saturation(activations)
        state_delta = self._calculate_state_delta(activations)
        state_delta_values = _bounded_with_candidate(
            self._state_delta_window,
            state_delta,
            max_len=self.config.stability_window,
        )
        prediction_values = (
            _bounded_with_candidate(
                self._prediction_window,
                prediction_value,
                max_len=self.config.stability_window,
            )
            if prediction_value is not None
            else []
        )
        stability = self._calculate_stability(
            state_delta_values=state_delta_values,
            prediction_values=prediction_values,
        )
        prediction_error = (
            _safe_abs_difference(target_value, prediction_value)
            if target_value is not None and prediction_value is not None
            else None
        )
        prediction_error_values = (
            _bounded_with_candidate(
                self._prediction_error_window,
                prediction_error,
                max_len=self.config.drift_window,
            )
            if prediction_error is not None
            else []
        )
        drift_pressure = self._calculate_drift_pressure(
            prediction_error_values=prediction_error_values,
            novelty=novelty,
            stability=stability,
            supervised_available=prediction_error is not None,
        )
        confidence = _prediction_confidence(prediction_value)

        if self._feature_dim is None:
            self._feature_dim = int(feature_vector.size)
        self._append_feature(feature_vector)
        self._append_activation(activations)
        _append_bounded(
            self._state_delta_window,
            state_delta,
            max_len=self.config.stability_window,
        )
        if prediction_value is not None:
            _append_bounded(
                self._prediction_window,
                prediction_value,
                max_len=self.config.stability_window,
            )
        if prediction_error is not None:
            _append_bounded(
                self._prediction_error_window,
                prediction_error,
                max_len=self.config.drift_window,
            )
        self._samples_seen += 1

        return _safe_channels(
            novelty=novelty,
            stability=stability,
            drift_pressure=drift_pressure,
            confidence=confidence,
            saturation=saturation,
        )

    def _validate_features(self, features: object) -> FloatArray:
        return validate_features(
            features,
            expected_dim=self._feature_dim,
            dtype=self.dtype,
        )

    def _validate_activations(self, state: ReservoirState) -> FloatArray:
        activations = validate_features(state.activations, dtype=self.dtype)
        if self._activation_window and activations.size != self._activation_window[0].size:
            msg = "state activations shape must remain stable"
            raise ValueError(msg)
        return activations

    def _calculate_novelty(
        self,
        *,
        features: FloatArray,
        activations: FloatArray,
    ) -> float:
        feature_score = _distance_to_recent_mean_score(
            features,
            self._feature_window,
            epsilon=self.config.epsilon,
        )
        state_score = _distance_to_recent_mean_score(
            activations,
            self._activation_window,
            epsilon=self.config.epsilon,
        )
        return _clip01(max(feature_score, state_score))

    def _calculate_stability(
        self,
        *,
        state_delta_values: list[float],
        prediction_values: list[float],
    ) -> float:
        components = [
            _volatility_instability(
                state_delta_values,
                epsilon=self.config.epsilon,
            )
        ]
        if len(prediction_values) >= 2:
            components.append(
                _volatility_instability(
                    prediction_values,
                    epsilon=self.config.epsilon,
                )
            )
        instability = float(np.mean(components)) if components else 0.0
        return _clip01(1.0 - instability)

    def _calculate_drift_pressure(
        self,
        *,
        prediction_error_values: list[float],
        novelty: float,
        stability: float,
        supervised_available: bool,
    ) -> float:
        if supervised_available:
            return _error_trend_pressure(
                prediction_error_values,
                epsilon=self.config.epsilon,
            )
        return _unsupervised_drift_proxy(novelty=novelty, stability=stability)

    def _calculate_saturation(self, activations: FloatArray) -> float:
        return _saturation_fraction(
            activations,
            threshold=self.config.saturation_threshold,
        )

    def _append_feature(self, features: FloatArray) -> None:
        self._feature_window.append(features)
        overflow = len(self._feature_window) - self.config.novelty_window
        if overflow > 0:
            del self._feature_window[:overflow]

    def _append_activation(self, activations: FloatArray) -> None:
        stored = np.array(activations, dtype=self.dtype, copy=True)
        stored.setflags(write=False)
        self._activation_window.append(stored)
        overflow = len(self._activation_window) - self.config.novelty_window
        if overflow > 0:
            del self._activation_window[:overflow]

    def _calculate_state_delta(self, activations: FloatArray) -> float:
        if self._previous_activations is None:
            state_delta = 0.0
        else:
            if activations.size != self._previous_activations.size:
                msg = "state activations shape must remain stable"
                raise ValueError(msg)
            difference = activations.astype(np.float64) - self._previous_activations.astype(
                np.float64
            )
            state_delta = _rms_value(difference)
        self._previous_activations = np.array(activations, dtype=self.dtype, copy=True)
        self._previous_activations.setflags(write=False)
        return state_delta


def _validate_dtype(value: str) -> str:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        msg = "dtype must be a valid floating dtype"
        raise ValueError(msg) from exc
    if dtype.name not in {"float32", "float64"}:
        msg = "dtype must be one of: float32, float64"
        raise ValueError(msg)
    return dtype.name


def _validate_optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        msg = f"{name} must be a positive integer or None"
        raise ValueError(msg)
    return value


def _validate_prediction(value: object) -> float:
    if isinstance(value, (str, bytes, bool)):
        msg = "prediction must be numeric"
        raise ValueError(msg)
    try:
        prediction = float(value)
    except (TypeError, ValueError) as exc:
        msg = "prediction must be numeric"
        raise ValueError(msg) from exc
    if not math.isfinite(prediction):
        msg = "prediction must be finite"
        raise ValueError(msg)
    return prediction


def _prediction_confidence(prediction: float | None) -> float:
    if prediction is None:
        return 0.0
    return _clip01(abs(prediction))


def _saturation_fraction(activations: FloatArray, *, threshold: float) -> float:
    values = np.asarray(activations, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return _clip01(float(np.mean(np.abs(values) > threshold)))


def _distance_to_recent_mean_score(
    current: FloatArray,
    history: list[FloatArray],
    *,
    epsilon: float,
) -> float:
    if not history:
        return 0.0
    matrix = np.vstack(history).astype(np.float64)
    recent_mean = np.mean(matrix, axis=0)
    distance = _rms_distance(current, recent_mean)
    if distance <= epsilon:
        return 0.0
    history_distances = np.asarray(
        [_rms_distance(row, recent_mean) for row in matrix],
        dtype=np.float64,
    )
    baseline = float(np.mean(history_distances))
    denominator = distance + _NOVELTY_BASELINE_MULTIPLIER * baseline + epsilon
    return _clip01(distance / denominator)


def _volatility_instability(values: list[float], *, epsilon: float) -> float:
    if len(values) < 2:
        return 0.0
    vector = np.asarray(values, dtype=np.float64)
    scale = _max_abs(vector)
    if scale == 0.0:
        return 0.0
    normalized = vector / scale
    mean = float(np.mean(normalized))
    centered = normalized - mean
    volatility = _rms_value(centered)
    normalized_epsilon = epsilon / scale
    if volatility <= normalized_epsilon:
        return 0.0
    magnitude = float(np.mean(np.abs(normalized)))
    return _clip01(volatility / (volatility + magnitude + normalized_epsilon))


def _error_trend_pressure(values: list[float], *, epsilon: float) -> float:
    if len(values) < 2:
        return 0.0
    vector = np.asarray(values, dtype=np.float64)
    scale = _max_abs(vector)
    if scale == 0.0:
        return 0.0
    normalized = vector / scale
    midpoint = len(normalized) // 2
    older = normalized[:midpoint]
    newer = normalized[midpoint:]
    older_mean = float(np.mean(older))
    newer_mean = float(np.mean(newer))
    increase = max(0.0, newer_mean - older_mean)
    normalized_epsilon = epsilon / scale
    denominator = abs(older_mean) + abs(newer_mean) + normalized_epsilon
    return _clip01(increase / denominator)


def _unsupervised_drift_proxy(*, novelty: float, stability: float) -> float:
    return _clip01(0.5 * _clip01(novelty) + 0.5 * (1.0 - _clip01(stability)))


def _safe_abs_difference(left: float, right: float) -> float:
    left_value = _finite_float(left)
    right_value = _finite_float(right)
    scale = max(abs(left_value), abs(right_value))
    if scale == 0.0:
        return 0.0
    normalized_difference = abs((left_value / scale) - (right_value / scale))
    if normalized_difference > _MAX_FLOAT64 / scale:
        return _MAX_FLOAT64
    return scale * normalized_difference


def _vector_window_to_tuple(values: list[FloatArray]) -> tuple[tuple[float, ...], ...]:
    return tuple(_vector_to_tuple(value) for value in values)


def _optional_vector_to_tuple(value: FloatArray | None) -> tuple[float, ...] | None:
    return None if value is None else _vector_to_tuple(value)


def _vector_to_tuple(value: FloatArray) -> tuple[float, ...]:
    return tuple(float(item) for item in np.asarray(value, dtype=np.float64))


def _restore_vector_window(
    values: tuple[tuple[float, ...], ...],
    *,
    name: str,
    expected_dim: int | None,
    max_len: int,
    dtype: str,
) -> tuple[list[FloatArray], int | None]:
    if not isinstance(values, tuple):
        msg = f"{name} must be a tuple"
        raise ValueError(msg)
    if len(values) > max_len:
        msg = f"{name} length exceeds configured window"
        raise ValueError(msg)
    restored: list[FloatArray] = []
    dimension = expected_dim
    for index, vector in enumerate(values):
        array, dimension = _restore_vector(
            vector,
            name=f"{name}[{index}]",
            expected_dim=dimension,
            dtype=dtype,
        )
        restored.append(array)
    return restored, dimension


def _restore_optional_vector(
    value: tuple[float, ...] | None,
    *,
    name: str,
    expected_dim: int | None,
    dtype: str,
) -> FloatArray | None:
    if value is None:
        return None
    array, _ = _restore_vector(
        value,
        name=name,
        expected_dim=expected_dim,
        dtype=dtype,
    )
    return array


def _restore_vector(
    value: tuple[float, ...],
    *,
    name: str,
    expected_dim: int | None,
    dtype: str,
) -> tuple[FloatArray, int]:
    if not isinstance(value, tuple):
        msg = f"{name} must be a tuple"
        raise ValueError(msg)
    if not value:
        msg = f"{name} must not be empty"
        raise ValueError(msg)
    if expected_dim is not None and len(value) != expected_dim:
        msg = f"{name} dimension must match {expected_dim}"
        raise ValueError(msg)
    array = np.asarray([_finite_float(item) for item in value], dtype=dtype)
    array.setflags(write=False)
    return array, int(array.size)


def _restore_float_window(
    values: tuple[float, ...],
    *,
    name: str,
    max_len: int,
) -> list[float]:
    if not isinstance(values, tuple):
        msg = f"{name} must be a tuple"
        raise ValueError(msg)
    if len(values) > max_len:
        msg = f"{name} length exceeds configured window"
        raise ValueError(msg)
    return [_finite_float(value) for value in values]


def _max_abs(values: FloatArray) -> float:
    magnitudes = np.abs(np.asarray(values, dtype=np.float64))
    return float(np.max(magnitudes)) if magnitudes.size > 0 else 0.0


def _bounded_with_candidate(
    values: list[float],
    value: float,
    *,
    max_len: int,
) -> list[float]:
    result = [*values, _finite_float(value)]
    overflow = len(result) - max_len
    if overflow > 0:
        del result[:overflow]
    return result


def _rms_distance(left: FloatArray, right: FloatArray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return _rms_value(difference)


def _rms_value(values: FloatArray) -> float:
    magnitudes = np.abs(np.asarray(values, dtype=np.float64))
    scale = float(np.max(magnitudes)) if magnitudes.size > 0 else 0.0
    if scale == 0.0:
        return 0.0
    normalized = magnitudes / scale
    return float(scale * np.sqrt(np.mean(normalized * normalized)))


def _append_bounded(values: list[float], value: float, *, max_len: int) -> None:
    values.append(_finite_float(value))
    overflow = len(values) - max_len
    if overflow > 0:
        del values[:overflow]


def _finite_float(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        msg = "history values must be finite"
        raise ValueError(msg)
    return result


def _clip01(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        msg = "channel value must be finite"
        raise ValueError(msg)
    return min(1.0, max(0.0, result))


def _safe_channels(
    *,
    novelty: float,
    stability: float,
    drift_pressure: float,
    confidence: float,
    saturation: float,
) -> AdaptiveChannels:
    return AdaptiveChannels(
        novelty=_clip01(novelty),
        stability=_clip01(stability),
        drift_pressure=_clip01(drift_pressure),
        confidence=_clip01(confidence),
        saturation=_clip01(saturation),
    )


__all__ = [
    "AdaptiveChannelCalculator",
    "ChannelCalculatorSnapshot",
    "CHANNEL_CALCULATOR_SNAPSHOT_SCHEMA_VERSION",
]
