"""Experimental sparse online scalar readout.

This module is intentionally isolated from the stable readout factory and root
package exports. The implementation is experimental evidence-gathering code, not
a default public API readout.
"""

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

SPARSE_ONLINE_READOUT_NAME = "experimental_sparse_online"


class SparseOnlineReadout:
    """Online linear scalar readout with simple L1 soft-threshold shrinkage."""

    def __init__(
        self,
        *,
        feature_dim: int,
        learning_rate: float = 0.05,
        l1_strength: float = 1e-4,
        epsilon: float = 1e-8,
        dtype: str = "float64",
    ) -> None:
        self.feature_dim = _validate_positive_int("feature_dim", feature_dim)
        self.learning_rate = _validate_positive_finite("learning_rate", learning_rate)
        self.l1_strength = _validate_non_negative_finite("l1_strength", l1_strength)
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
        """Apply one normalized online update followed by L1 weight shrinkage."""

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
        shrunk_weights = _soft_threshold(updated_weights, threshold=self.l1_strength)
        shrunk_weights = np.asarray(shrunk_weights, dtype=self.dtype)
        shrunk_weights.setflags(write=False)
        self._weights = shrunk_weights
        self._bias = float(self._bias + self.learning_rate * error)
        self._samples_seen += 1

    def snapshot(self) -> ReadoutSnapshot:
        """Return an immutable numeric snapshot of the sparse online readout state."""

        return ReadoutSnapshot(
            schema_version=READOUT_SNAPSHOT_SCHEMA_VERSION,
            name=SPARSE_ONLINE_READOUT_NAME,
            state={
                "feature_dim": self.feature_dim,
                "dtype": self.dtype,
                "learning_rate": self.learning_rate,
                "l1_strength": self.l1_strength,
                "epsilon": self.epsilon,
                "weights": tuple(float(value) for value in self._weights),
                "bias": self._bias,
                "samples_seen": self._samples_seen,
            },
        )

    def restore(self, snapshot: ReadoutSnapshot) -> None:
        """Restore sparse online readout state from a compatible snapshot."""

        if not isinstance(snapshot, ReadoutSnapshot):
            msg = "snapshot must be a ReadoutSnapshot"
            raise TypeError(msg)
        if snapshot.schema_version != READOUT_SNAPSHOT_SCHEMA_VERSION:
            msg = f"unsupported readout snapshot schema_version: {snapshot.schema_version}"
            raise ValueError(msg)
        if snapshot.name != SPARSE_ONLINE_READOUT_NAME:
            msg = f"snapshot name must be {SPARSE_ONLINE_READOUT_NAME!r}"
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
        l1_strength = _required_float(state, "l1_strength")
        if l1_strength != self.l1_strength:
            msg = "snapshot l1_strength must match current readout"
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


def _soft_threshold(values: FloatArray, *, threshold: float) -> FloatArray:
    if threshold == 0.0:
        return np.asarray(values)
    magnitudes = np.maximum(np.abs(values) - threshold, 0.0)
    return np.sign(values) * magnitudes


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return value


def _validate_positive_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        msg = f"{name} must be finite and positive"
        raise ValueError(msg)
    return result


def _validate_non_negative_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        msg = f"{name} must be finite and non-negative"
        raise ValueError(msg)
    return result


def _validate_floating_dtype(value: str) -> str:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        msg = "dtype must be a valid floating dtype"
        raise ValueError(msg) from exc
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


__all__ = ["SPARSE_ONLINE_READOUT_NAME", "SparseOnlineReadout"]
