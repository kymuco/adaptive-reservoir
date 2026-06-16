"""Replay-buffer ridge regression scalar readout."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from adaptive_reservoir.readout.base import (
    READOUT_SNAPSHOT_SCHEMA_VERSION,
    FloatArray,
    ReadoutSnapshot,
    validate_features,
    validate_snapshot_mapping,
    validate_target,
)

REPLAY_RIDGE_READOUT_NAME = "replay_ridge"
_SUPPORTED_LINALG_DTYPE_NAMES = frozenset({"float32", "float64"})


class ReplayRidgeReadout:
    """Stable scalar readout using periodic ridge refits over a replay buffer."""

    def __init__(
        self,
        *,
        feature_dim: int,
        buffer_size: int = 256,
        refit_interval: int = 1,
        alpha: float = 1e-3,
        dtype: str = "float64",
    ) -> None:
        self.feature_dim = _validate_positive_int("feature_dim", feature_dim)
        self.buffer_size = _validate_positive_int("buffer_size", buffer_size)
        self.refit_interval = _validate_positive_int("refit_interval", refit_interval)
        self.alpha = _validate_positive_finite("alpha", alpha)
        self.dtype = _validate_floating_dtype(dtype)
        self._weights = np.zeros(self.feature_dim, dtype=self.dtype)
        self._weights.setflags(write=False)
        self._bias = 0.0
        self._samples_seen = 0
        self._solve_count = 0
        self._features_buffer: list[FloatArray] = []
        self._targets_buffer: list[float] = []

    @property
    def samples_seen(self) -> int:
        """Number of supervised updates observed by this readout."""

        return self._samples_seen

    @property
    def solve_count(self) -> int:
        """Number of successful ridge refits performed by this readout."""

        return self._solve_count

    @property
    def buffer_count(self) -> int:
        """Number of supervised samples currently stored in the replay buffer."""

        return len(self._targets_buffer)

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
        """Store one supervised sample and periodically refit ridge weights."""

        vector = validate_features(
            features,
            expected_dim=self.feature_dim,
            dtype=self.dtype,
        )
        target_value = validate_target(target)
        self._append_sample(vector, target_value)
        self._samples_seen += 1
        if self._samples_seen % self.refit_interval == 0:
            self._refit()

    def snapshot(self) -> ReadoutSnapshot:
        """Return an immutable numeric snapshot of this replay ridge readout."""

        return ReadoutSnapshot(
            schema_version=READOUT_SNAPSHOT_SCHEMA_VERSION,
            name=REPLAY_RIDGE_READOUT_NAME,
            state={
                "feature_dim": self.feature_dim,
                "buffer_size": self.buffer_size,
                "refit_interval": self.refit_interval,
                "alpha": self.alpha,
                "dtype": self.dtype,
                "weights": tuple(float(value) for value in self._weights),
                "bias": self._bias,
                "samples_seen": self._samples_seen,
                "solve_count": self._solve_count,
                "features_buffer": tuple(
                    tuple(float(value) for value in row)
                    for row in self._features_buffer
                ),
                "targets_buffer": tuple(self._targets_buffer),
            },
        )

    def restore(self, snapshot: ReadoutSnapshot) -> None:
        """Restore readout state from a compatible replay ridge snapshot."""

        if not isinstance(snapshot, ReadoutSnapshot):
            msg = "snapshot must be a ReadoutSnapshot"
            raise TypeError(msg)
        if snapshot.schema_version != READOUT_SNAPSHOT_SCHEMA_VERSION:
            msg = f"unsupported readout snapshot schema_version: {snapshot.schema_version}"
            raise ValueError(msg)
        if snapshot.name != REPLAY_RIDGE_READOUT_NAME:
            msg = f"snapshot name must be {REPLAY_RIDGE_READOUT_NAME!r}"
            raise ValueError(msg)
        state = validate_snapshot_mapping(snapshot.state)
        self._restore_state(state)

    def _append_sample(self, features: FloatArray, target: float) -> None:
        if len(self._features_buffer) >= self.buffer_size:
            self._features_buffer.pop(0)
            self._targets_buffer.pop(0)
        self._features_buffer.append(features)
        self._targets_buffer.append(target)

    def _refit(self) -> None:
        if not self._features_buffer:
            return
        features = np.vstack(self._features_buffer).astype(self.dtype, copy=False)
        targets = np.asarray(self._targets_buffer, dtype=self.dtype)
        bias_column = np.ones((features.shape[0], 1), dtype=self.dtype)
        design = np.hstack((features, bias_column))
        penalty = np.eye(self.feature_dim + 1, dtype=self.dtype) * self.alpha
        penalty[-1, -1] = 0.0
        normal_matrix = design.T @ design + penalty
        rhs = design.T @ targets
        coefficients = np.linalg.solve(normal_matrix, rhs)
        if not np.all(np.isfinite(coefficients)):
            msg = "ridge refit produced non-finite coefficients"
            raise ValueError(msg)
        weights = np.asarray(coefficients[:-1], dtype=self.dtype)
        weights.setflags(write=False)
        self._weights = weights
        self._bias = float(coefficients[-1])
        self._solve_count += 1

    def _restore_state(self, state: Mapping[str, object]) -> None:
        feature_dim = _required_int(state, "feature_dim")
        if feature_dim != self.feature_dim:
            msg = f"snapshot feature_dim must match {self.feature_dim}; got {feature_dim}"
            raise ValueError(msg)
        buffer_size = _required_int(state, "buffer_size")
        if buffer_size != self.buffer_size:
            msg = f"snapshot buffer_size must match {self.buffer_size}; got {buffer_size}"
            raise ValueError(msg)
        refit_interval = _required_int(state, "refit_interval")
        if refit_interval != self.refit_interval:
            msg = f"snapshot refit_interval must match {self.refit_interval}; got {refit_interval}"
            raise ValueError(msg)
        alpha = _required_float(state, "alpha")
        if alpha != self.alpha:
            msg = "snapshot alpha must match current readout"
            raise ValueError(msg)
        dtype = _required_str(state, "dtype")
        if np.dtype(dtype) != np.dtype(self.dtype):
            msg = f"snapshot dtype must match {self.dtype!r}; got {dtype!r}"
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
        solve_count = _optional_non_negative_int(state, "solve_count", default=0)
        features_buffer = _validate_features_buffer(
            state.get("features_buffer"),
            feature_dim=self.feature_dim,
            buffer_size=self.buffer_size,
            dtype=self.dtype,
        )
        targets_buffer = _validate_targets_buffer(
            state.get("targets_buffer"),
            expected_len=len(features_buffer),
        )
        if samples_seen < len(targets_buffer):
            msg = "snapshot samples_seen must be at least the replay buffer length"
            raise ValueError(msg)

        restored_weights = weights.copy()
        restored_weights.setflags(write=False)
        self._weights = restored_weights
        self._bias = bias
        self._samples_seen = samples_seen
        self._solve_count = solve_count
        self._features_buffer = features_buffer
        self._targets_buffer = targets_buffer


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


def _validate_floating_dtype(value: str) -> str:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        msg = "dtype must be a valid floating dtype"
        raise ValueError(msg) from exc
    if not np.issubdtype(dtype, np.floating):
        msg = "dtype must be a floating dtype"
        raise ValueError(msg)
    if dtype.name not in _SUPPORTED_LINALG_DTYPE_NAMES:
        msg = "dtype must be one of: float32, float64"
        raise ValueError(msg)
    return dtype.name


def _required_int(state: Mapping[str, object], key: str) -> int:
    value = state.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"snapshot state.{key} must be an integer"
        raise ValueError(msg)
    return value


def _optional_non_negative_int(
    state: Mapping[str, object],
    key: str,
    *,
    default: int,
) -> int:
    value = state.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"snapshot state.{key} must be a non-negative integer"
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


def _validate_features_buffer(
    value: object,
    *,
    feature_dim: int,
    buffer_size: int,
    dtype: str,
) -> list[FloatArray]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = "snapshot state.features_buffer must be a sequence"
        raise ValueError(msg)
    if len(value) > buffer_size:
        msg = "snapshot state.features_buffer must not exceed buffer_size"
        raise ValueError(msg)
    result: list[FloatArray] = []
    for row in value:
        features = validate_features(row, expected_dim=feature_dim, dtype=dtype)
        result.append(features)
    return result


def _validate_targets_buffer(value: object, *, expected_len: int) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = "snapshot state.targets_buffer must be a sequence"
        raise ValueError(msg)
    if len(value) != expected_len:
        msg = "snapshot state.targets_buffer length must match features_buffer length"
        raise ValueError(msg)
    return [_coerce_finite_float(item, "targets_buffer") for item in value]


def _coerce_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        msg = f"snapshot state.{name} values must be numeric"
        raise ValueError(msg)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"snapshot state.{name} values must be numeric"
        raise ValueError(msg) from exc
    if not math.isfinite(result):
        msg = f"snapshot state.{name} values must be finite"
        raise ValueError(msg)
    return result
