"""Experimental recursive least squares scalar readout."""

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

RLS_READOUT_NAME = "experimental_rls"
_SUPPORTED_DTYPE_NAME = "float64"
_COVARIANCE_EIGENVALUE_TOLERANCE = 1e-10


class RLSReadout:
    """Experimental scalar readout using recursive least squares updates."""

    def __init__(
        self,
        *,
        feature_dim: int,
        forgetting_factor: float = 0.99,
        covariance_scale: float = 1_000.0,
        jitter: float = 1e-8,
        dtype: str = "float64",
    ) -> None:
        self.feature_dim = _validate_positive_int("feature_dim", feature_dim)
        self.forgetting_factor = _validate_forgetting_factor(forgetting_factor)
        self.covariance_scale = _validate_positive_finite(
            "covariance_scale",
            covariance_scale,
        )
        self.jitter = _validate_positive_finite("jitter", jitter)
        self.dtype = _validate_float64_dtype(dtype)
        self._theta = np.zeros(self.feature_dim + 1, dtype=self.dtype)
        self._theta.setflags(write=False)
        self._covariance = np.eye(self.feature_dim + 1, dtype=self.dtype)
        self._covariance *= self.covariance_scale
        self._covariance.setflags(write=False)
        self._samples_seen = 0

    @property
    def samples_seen(self) -> int:
        """Number of supervised updates applied to this readout."""

        return self._samples_seen

    @property
    def weights(self) -> FloatArray:
        """Read-only copy of the current linear weights."""

        weights = self._theta[:-1].copy()
        weights.setflags(write=False)
        return weights

    @property
    def bias(self) -> float:
        """Current scalar bias term."""

        return float(self._theta[-1])

    @property
    def covariance(self) -> FloatArray:
        """Read-only copy of the current RLS covariance matrix."""

        covariance = self._covariance.copy()
        covariance.setflags(write=False)
        return covariance

    def predict(self, features: object) -> float:
        """Return the current scalar prediction for a validated feature vector."""

        vector = validate_features(
            features,
            expected_dim=self.feature_dim,
            dtype=self.dtype,
        )
        augmented = _augment_features(vector)
        return float(np.dot(self._theta, augmented))

    def update(self, features: object, target: object) -> None:
        """Apply one recursive least squares update atomically."""

        vector = validate_features(
            features,
            expected_dim=self.feature_dim,
            dtype=self.dtype,
        )
        target_value = validate_target(target)
        augmented = _augment_features(vector)
        prediction = float(np.dot(self._theta, augmented))
        error = target_value - prediction
        covariance_times_features = self._covariance @ augmented
        denominator = (
            self.forgetting_factor
            + float(np.dot(augmented, covariance_times_features))
            + self.jitter
        )
        if not math.isfinite(denominator) or denominator <= 0.0:
            msg = "RLS update denominator must be finite and positive"
            raise ValueError(msg)
        gain = covariance_times_features / denominator
        new_theta = self._theta + gain * error
        covariance_update = np.outer(gain, augmented @ self._covariance)
        new_covariance = (self._covariance - covariance_update) / self.forgetting_factor
        new_covariance = 0.5 * (new_covariance + new_covariance.T)
        _validate_vector("theta", new_theta, expected_dim=self.feature_dim + 1)
        _validate_matrix(
            "covariance",
            new_covariance,
            expected_dim=self.feature_dim + 1,
        )

        new_theta = np.asarray(new_theta, dtype=self.dtype)
        new_covariance = np.asarray(new_covariance, dtype=self.dtype)
        new_theta.setflags(write=False)
        new_covariance.setflags(write=False)
        self._theta = new_theta
        self._covariance = new_covariance
        self._samples_seen += 1

    def snapshot(self) -> ReadoutSnapshot:
        """Return an immutable numeric snapshot of this experimental RLS readout."""

        return ReadoutSnapshot(
            schema_version=READOUT_SNAPSHOT_SCHEMA_VERSION,
            name=RLS_READOUT_NAME,
            state={
                "feature_dim": self.feature_dim,
                "forgetting_factor": self.forgetting_factor,
                "covariance_scale": self.covariance_scale,
                "jitter": self.jitter,
                "dtype": self.dtype,
                "weights": tuple(float(value) for value in self._theta[:-1]),
                "bias": float(self._theta[-1]),
                "covariance": tuple(
                    tuple(float(value) for value in row)
                    for row in self._covariance
                ),
                "samples_seen": self._samples_seen,
            },
        )

    def restore(self, snapshot: ReadoutSnapshot) -> None:
        """Restore experimental RLS state from a compatible snapshot."""

        if not isinstance(snapshot, ReadoutSnapshot):
            msg = "snapshot must be a ReadoutSnapshot"
            raise TypeError(msg)
        if snapshot.schema_version != READOUT_SNAPSHOT_SCHEMA_VERSION:
            msg = f"unsupported readout snapshot schema_version: {snapshot.schema_version}"
            raise ValueError(msg)
        if snapshot.name != RLS_READOUT_NAME:
            msg = f"snapshot name must be {RLS_READOUT_NAME!r}"
            raise ValueError(msg)
        state = validate_snapshot_mapping(snapshot.state)
        self._restore_state(state)

    def _restore_state(self, state: Mapping[str, object]) -> None:
        feature_dim = _required_int(state, "feature_dim")
        if feature_dim != self.feature_dim:
            msg = f"snapshot feature_dim must match {self.feature_dim}; got {feature_dim}"
            raise ValueError(msg)
        forgetting_factor = _required_float(state, "forgetting_factor")
        if forgetting_factor != self.forgetting_factor:
            msg = "snapshot forgetting_factor must match current readout"
            raise ValueError(msg)
        covariance_scale = _required_float(state, "covariance_scale")
        if covariance_scale != self.covariance_scale:
            msg = "snapshot covariance_scale must match current readout"
            raise ValueError(msg)
        jitter = _required_float(state, "jitter")
        if jitter != self.jitter:
            msg = "snapshot jitter must match current readout"
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
        covariance = _validate_covariance(
            state.get("covariance"),
            expected_dim=self.feature_dim + 1,
            dtype=self.dtype,
        )
        samples_seen = _required_int(state, "samples_seen")
        if samples_seen < 0:
            msg = "snapshot samples_seen must be non-negative"
            raise ValueError(msg)

        theta = np.concatenate((weights, np.asarray([bias], dtype=self.dtype)))
        theta.setflags(write=False)
        covariance.setflags(write=False)
        self._theta = theta
        self._covariance = covariance
        self._samples_seen = samples_seen


def _augment_features(features: FloatArray) -> FloatArray:
    augmented = np.empty(features.size + 1, dtype=features.dtype)
    augmented[:-1] = features
    augmented[-1] = 1.0
    augmented.setflags(write=False)
    return augmented


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return value


def _validate_forgetting_factor(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0 or result > 1.0:
        msg = "forgetting_factor must be finite and in the interval (0, 1]"
        raise ValueError(msg)
    return result


def _validate_positive_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        msg = f"{name} must be finite and positive"
        raise ValueError(msg)
    return result


def _validate_float64_dtype(value: str) -> str:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        msg = "dtype must be a valid numpy dtype"
        raise ValueError(msg) from exc
    if dtype.name != _SUPPORTED_DTYPE_NAME:
        msg = "dtype must be float64 for experimental RLS"
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


def _validate_covariance(
    value: object,
    *,
    expected_dim: int,
    dtype: str,
) -> FloatArray:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = "snapshot state.covariance must be a matrix sequence"
        raise ValueError(msg)
    try:
        matrix = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        msg = "snapshot state.covariance must contain only numeric values"
        raise ValueError(msg) from exc
    _validate_matrix("covariance", matrix, expected_dim=expected_dim)
    matrix = 0.5 * (matrix + matrix.T)
    _validate_positive_semidefinite_covariance(matrix)
    return np.asarray(matrix, dtype=dtype)


def _validate_vector(name: str, value: object, *, expected_dim: int) -> None:
    vector = np.asarray(value, dtype="float64")
    if vector.ndim != 1 or vector.size != expected_dim:
        msg = f"{name} must have shape ({expected_dim},)"
        raise ValueError(msg)
    if not np.all(np.isfinite(vector)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)


def _validate_matrix(name: str, value: object, *, expected_dim: int) -> None:
    matrix = np.asarray(value, dtype="float64")
    expected_shape = (expected_dim, expected_dim)
    if matrix.shape != expected_shape:
        msg = f"{name} must have shape {expected_shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(matrix)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)


def _validate_positive_semidefinite_covariance(matrix: FloatArray) -> None:
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.min(eigenvalues) < -_COVARIANCE_EIGENVALUE_TOLERANCE:
        msg = "covariance must be positive semidefinite"
        raise ValueError(msg)
