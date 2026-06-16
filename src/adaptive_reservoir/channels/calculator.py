"""Stateful base calculator for adaptive state channels."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from adaptive_reservoir.core.config import ChannelConfig
from adaptive_reservoir.core.result import AdaptiveChannels
from adaptive_reservoir.core.state import ReservoirState
from adaptive_reservoir.readout.base import FloatArray, validate_features, validate_target


class AdaptiveChannelCalculator:
    """Stateful base for bounded numeric adaptive channel signals."""

    def __init__(self, *, config: ChannelConfig, dtype: str = "float64") -> None:
        if not isinstance(config, ChannelConfig):
            msg = "config must be a ChannelConfig"
            raise TypeError(msg)
        self.config = config
        self.dtype = _validate_dtype(dtype)
        self._samples_seen = 0
        self._feature_dim: int | None = None
        self._feature_window: list[FloatArray] = []
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
        self._state_delta_window.clear()
        self._prediction_window.clear()
        self._prediction_error_window.clear()
        self._previous_activations = None

    def update(
        self,
        input: object,
        state: ReservoirState,
        features: object,
        prediction: object,
        target: object | None,
    ) -> AdaptiveChannels:
        """Update numeric channel histories and return safe default channels.

        The raw ``input`` value is accepted to preserve the channel-calculator
        contract, but it is intentionally not stored by this base calculator.
        """

        del input
        if not isinstance(state, ReservoirState):
            msg = "state must be a ReservoirState"
            raise TypeError(msg)
        feature_vector = self._validate_features(features)
        prediction_value = _validate_prediction(prediction)
        target_value = None if target is None else validate_target(target)
        state_delta = self._calculate_state_delta(state)

        self._append_feature(feature_vector)
        _append_bounded(
            self._state_delta_window,
            state_delta,
            max_len=self.config.stability_window,
        )
        _append_bounded(
            self._prediction_window,
            prediction_value,
            max_len=self.config.stability_window,
        )
        if target_value is not None:
            _append_bounded(
                self._prediction_error_window,
                abs(target_value - prediction_value),
                max_len=self.config.drift_window,
            )
        self._samples_seen += 1

        return _safe_channels(
            novelty=0.0,
            stability=1.0,
            drift_pressure=0.0,
            confidence=0.0,
            saturation=0.0,
        )

    def _validate_features(self, features: object) -> FloatArray:
        vector = validate_features(
            features,
            expected_dim=self._feature_dim,
            dtype=self.dtype,
        )
        if self._feature_dim is None:
            self._feature_dim = int(vector.size)
        return vector

    def _append_feature(self, features: FloatArray) -> None:
        self._feature_window.append(features)
        overflow = len(self._feature_window) - self.config.novelty_window
        if overflow > 0:
            del self._feature_window[:overflow]

    def _calculate_state_delta(self, state: ReservoirState) -> float:
        activations = validate_features(state.activations, dtype=self.dtype)
        if self._previous_activations is None:
            state_delta = 0.0
        else:
            if activations.size != self._previous_activations.size:
                msg = "state activations shape must remain stable"
                raise ValueError(msg)
            difference = activations - self._previous_activations
            state_delta = float(np.sqrt(np.mean(difference * difference)))
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


__all__ = ["AdaptiveChannelCalculator"]
