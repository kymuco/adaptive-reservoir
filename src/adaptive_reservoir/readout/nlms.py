"""Normalized least mean squares scalar readout."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from adaptive_reservoir.readout.base import (
    READOUT_SNAPSHOT_SCHEMA_VERSION,
    FloatArray,
    ReadoutSnapshot,
    validate_features,
    validate_snapshot_mapping,
    validate_target,
)

NLMS_READOUT_NAME = "nlms"


class NLMSReadout:
    """Cheap online linear scalar readout using normalized LMS updates."""

    def __init__(
        self,
        *,
        feature_dim: int,
        learning_rate: float = 0.1,
        epsilon: float = 1e-8,
        dtype: str = "float64",
    ) -> None:
        self.feature_dim = _validate_feature_dim(feature_dim)
        self.learning_rate = _validate_positive_finite("learning_rate", learning_rate)
        self.epsilon = _validate_positive_finite("epsilon", epsilon)
        self.dtype = _validate_floating_dtype(dtype)
        self._weights = np.zeros(self.feature_dim, dtype=self.dtype)
        self._weights.setflags(write=False)
        self._bias = 0.0
        self._samples_seen = 0

    @property
    def samples_seen(self) -> int:
        """Number of supervised updates applied to this readout."""

        return self._samples_seen

    @property
    def weights(self) -> FloatArray:
        """Read-only copy of the current linear weights."""

        weights = self._weights.copy()
        weights.setflags(write=False)
        return weights

    @property
    def bias(self) -> float:
        """Current scalar bias term."""

        return self._bias

    def predict(self, features: object) -> float:
        """Return ``weights dot features + bias`` for a validated feature vector."""

        vector = validate_features(
            features,
            expected_dim=self.feature_dim,
            dtype=self.dtype,
        )
        return float(np.dot(self._weights, vector) + self._bias)

    def update(self, features: object, target: object) -> None:
        """Apply one normalized least mean squares update."""

        vector = validate_features(
            features,
            expected_dim=self.feature_dim,
            dtype=self.dtype,
        )
        target_value = validate_target(target)
        prediction = float(np.dot(self._weights, vector) + self._bias)
        error = target_value - prediction
        normalizer = self.epsilon + float(np.dot(vector, vector))
        updated_weights = self._weights + (self.learning_rate * error / normalizer) * vector
        updated_weights = np.asarray(updated_weights, dtype=self.dtype)
        updated_weights.setflags(write=False)
        self._weights = updated_weights
        self._bias = float(self._bias + self.learning_rate * error)
        self._samples_seen += 1

    def snapshot(self) -> ReadoutSnapshot:
        """Return an immutable numeric snapshot of the NLMS readout state."""

        return ReadoutSnapshot(
            schema_version=READOUT_SNAPSHOT_SCHEMA_VERSION,
            name=NLMS_READOUT_NAME,
            state={
                "feature_dim": self.feature_dim,
                "dtype": self.dtype,
                "learning_rate": self.learning_rate,
                "epsilon": self.epsilon,
                "weights": tuple(float(value) for value in self._weights),
                "bias": self._bias,
                "samples_seen": self._samples_seen,
            },
        )

    def restore(self, snapshot: ReadoutSnapshot) -> None:
        """Restore NLMS readout state from a compatible snapshot."""

        if not isinstance(snapshot, ReadoutSnapshot):
            msg = "snapshot must be a ReadoutSnapshot"
            raise TypeError(msg)
        if snapshot.schema_version != READOUT_SNAPSHOT_SCHEMA_VERSION:
            msg = f"unsupported readout snapshot schema_version: {snapshot.schema_version}"
            raise ValueError(msg)
        if snapshot.name != NLMS_READOUT_NAME:
            msg = f"snapshot name must be {NLMS_READOUT_NAME!r}"
            raise ValueError(msg)
        state = validate_snapshot_mapping(snapshot.state)
        self._restore_state(state)

    def _restore_state(self, state: Mapping[str, object]) -> None:
        feature_dim = _required_int(state, "feature_dim")
        if feature_dim != self.feature_dim:
            msg = f"snapshot feature_dim must match {self.feature_dim}; got {feature_dim}"
            raise ValueError(msg)
        dtype = _required_str(state, "dtype")
        if np.dtype(dtype) != np.dtype(self.dtype):
            msg = f"snapshot dtype must match {self.dtype!r}; got {dtype!r}"
            raise ValueError(msg)
        learning_rate = _required_float(state, "learning_rate")
        if learning_rate != self.learning_rate:
            msg = "snapshot learning_rate must match current readout"
            raise ValueError(msg)
        epsilon = _required_float(state, "epsilon")
        if epsilon != self.epsilon:
            msg = "snapshot epsilon must match current readout"
            raise ValueError(msg)
        weights = validate_features(
            state.get("weights"),
            expected_dim=self.feature_dim,
            dtype=self.dtype,
        )
        bias = _required_float(state, "bias")
        samples_seen = _required_int(state, "samples_seen")
        if samples_seen < 0:
            msg = "snapshot samples_seen must be non-negative"
            raise ValueError(msg)

        restored_weights = weights.copy()
        restored_weights.setflags(write=False)
        self._weights = restored_weights
        self._bias = bias
        self._samples_seen = samples_seen


def _validate_feature_dim(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = "feature_dim must be a positive integer"
        raise ValueError(msg)
    return value


def _validate_positive_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        msg = f"{name} must be finite and positive"
        raise ValueError(msg)
    return result


def _validate_floating_dtype(value: str) -> str:
    dtype = np.dtype(value)
    if not np.issubdtype(dtype, np.floating):
        msg = "dtype must be a floating dtype"
        raise ValueError(msg)
    return dtype.name


def _required_int(state: Mapping[str, object], key: str) -> int:
    value = state.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"snapshot state.{key} must be an integer"
        raise ValueError(msg)
    return value


def _required_str(state: Mapping[str, object], key: str) -> str:
    value = state.get(key)
    if not isinstance(value, str):
        msg = f"snapshot state.{key} must be a string"
        raise ValueError(msg)
    return value


def _required_float(state: Mapping[str, object], key: str) -> float:
    value = state.get(key)
    if isinstance(value, bool):
        msg = f"snapshot state.{key} must be numeric"
        raise ValueError(msg)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"snapshot state.{key} must be numeric"
        raise ValueError(msg) from exc
    if not math.isfinite(result):
        msg = f"snapshot state.{key} must be finite"
        raise ValueError(msg)
    return result
