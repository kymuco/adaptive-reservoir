"""Experimental online Oja-style feature compressor.

This module is intentionally isolated from the stable reservoir runtime, readout
factory, and root package exports. It provides an unsupervised experimental
projection layer, not a supervised readout.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from adaptive_reservoir.core.serialization import json_friendly
from adaptive_reservoir.readout.base import FloatArray, validate_features

OJA_COMPRESSOR_NAME = "experimental_oja_compressor"
OJA_COMPRESSOR_SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class OjaCompressorSnapshot:
    """Immutable numeric snapshot for an experimental Oja compressor."""

    schema_version: int
    name: str
    state: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly Oja compressor snapshot dictionary."""

        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "state": json_friendly(self.state),
        }

    @classmethod
    def from_dict(cls, data: object) -> OjaCompressorSnapshot:
        """Create an Oja compressor snapshot from a JSON-friendly mapping."""

        if not isinstance(data, Mapping):
            msg = "oja snapshot must be a mapping"
            raise ValueError(msg)
        state = validate_snapshot_mapping(data.get("state"))
        return cls(
            schema_version=_required_int(data, "schema_version"),
            name=_required_str(data, "name"),
            state=state,
        )

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            msg = "schema_version must be positive"
            raise ValueError(msg)
        if not self.name:
            msg = "name must not be empty"
            raise ValueError(msg)
        state = validate_snapshot_mapping(self.state)
        object.__setattr__(self, "state", _freeze_snapshot_mapping(state))


class OjaCompressor:
    """Unsupervised online feature compressor using an Oja-style basis update."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        learning_rate: float = 0.01,
        seed: int | None = None,
        dtype: str = "float64",
    ) -> None:
        self.input_dim = _validate_positive_int("input_dim", input_dim)
        self.output_dim = _validate_positive_int("output_dim", output_dim)
        if self.output_dim >= self.input_dim:
            msg = "output_dim must be smaller than input_dim"
            raise ValueError(msg)
        self.learning_rate = _validate_positive_finite("learning_rate", learning_rate)
        self.seed = _validate_seed(seed)
        self.dtype = _validate_floating_dtype(dtype)
        self._components = _initial_components(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            seed=self.seed,
            dtype=self.dtype,
        )
        self._samples_seen = 0

    @property
    def samples_seen(self) -> int:
        """Number of unsupervised updates applied to this compressor."""

        return self._samples_seen

    @property
    def components(self) -> FloatArray:
        """Read-only copy of the current projection components matrix."""

        components = self._components.copy()
        components.setflags(write=False)
        return components

    def transform(self, features: object) -> FloatArray:
        """Project a validated feature vector without mutating compressor state."""

        vector = validate_features(
            features,
            expected_dim=self.input_dim,
            dtype=self.dtype,
        )
        projection = np.asarray(self._components @ vector, dtype=self.dtype)
        projection.setflags(write=False)
        return projection

    def update(self, features: object) -> None:
        """Apply one unsupervised Oja-style online basis update."""

        vector = validate_features(
            features,
            expected_dim=self.input_dim,
            dtype=self.dtype,
        )
        projection = self._components @ vector
        reconstruction = projection @ self._components
        residual = vector - reconstruction
        updated = self._components + self.learning_rate * np.outer(projection, residual)
        self._components = _orthonormalize_rows(updated, dtype=self.dtype)
        self._samples_seen += 1

    def step(self, features: object) -> FloatArray:
        """Return the current projection, then update the basis from features."""

        projection = self.transform(features)
        self.update(features)
        return projection

    def snapshot(self) -> OjaCompressorSnapshot:
        """Return an immutable numeric snapshot of the compressor state."""

        return OjaCompressorSnapshot(
            schema_version=OJA_COMPRESSOR_SNAPSHOT_SCHEMA_VERSION,
            name=OJA_COMPRESSOR_NAME,
            state={
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "learning_rate": self.learning_rate,
                "seed": self.seed,
                "dtype": self.dtype,
                "components": tuple(
                    tuple(float(value) for value in row) for row in self._components
                ),
                "samples_seen": self._samples_seen,
            },
        )

    def restore(self, snapshot: OjaCompressorSnapshot) -> None:
        """Restore compressor state from a compatible Oja snapshot."""

        if not isinstance(snapshot, OjaCompressorSnapshot):
            msg = "snapshot must be an OjaCompressorSnapshot"
            raise TypeError(msg)
        if snapshot.schema_version != OJA_COMPRESSOR_SNAPSHOT_SCHEMA_VERSION:
            msg = (
                "unsupported Oja compressor snapshot schema_version: "
                f"{snapshot.schema_version}"
            )
            raise ValueError(msg)
        if snapshot.name != OJA_COMPRESSOR_NAME:
            msg = f"snapshot name must be {OJA_COMPRESSOR_NAME!r}"
            raise ValueError(msg)
        state = validate_snapshot_mapping(snapshot.state)
        self._restore_state(state)

    def _restore_state(self, state: Mapping[str, object]) -> None:
        input_dim = _required_int(state, "input_dim")
        if input_dim != self.input_dim:
            msg = f"snapshot input_dim must match {self.input_dim}; got {input_dim}"
            raise ValueError(msg)
        output_dim = _required_int(state, "output_dim")
        if output_dim != self.output_dim:
            msg = f"snapshot output_dim must match {self.output_dim}; got {output_dim}"
            raise ValueError(msg)
        learning_rate = _required_float(state, "learning_rate")
        if learning_rate != self.learning_rate:
            msg = "snapshot learning_rate must match current compressor"
            raise ValueError(msg)
        dtype = _required_str(state, "dtype")
        if np.dtype(dtype) != np.dtype(self.dtype):
            msg = f"snapshot dtype must match {self.dtype!r}; got {dtype!r}"
            raise ValueError(msg)
        components = _validate_components(
            state.get("components"),
            output_dim=self.output_dim,
            input_dim=self.input_dim,
            dtype=self.dtype,
        )
        samples_seen = _required_int(state, "samples_seen")
        if samples_seen < 0:
            msg = "snapshot samples_seen must be non-negative"
            raise ValueError(msg)

        components.setflags(write=False)
        self._components = components
        self._samples_seen = samples_seen


def validate_snapshot_mapping(snapshot: object) -> Mapping[str, object]:
    """Validate that an Oja compressor snapshot payload is mapping-shaped."""

    if not isinstance(snapshot, Mapping):
        msg = "snapshot must be a mapping"
        raise ValueError(msg)
    return snapshot


def _initial_components(
    *,
    input_dim: int,
    output_dim: int,
    seed: int | None,
    dtype: str,
) -> FloatArray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(output_dim, input_dim))
    return _orthonormalize_rows(raw, dtype=dtype)


def _orthonormalize_rows(values: object, *, dtype: str) -> FloatArray:
    matrix = np.asarray(values, dtype="float64")
    if matrix.ndim != 2:
        msg = "components must be a 2D matrix"
        raise ValueError(msg)
    if not np.all(np.isfinite(matrix)):
        msg = "components must contain only finite values"
        raise ValueError(msg)
    q, _ = np.linalg.qr(matrix.T, mode="reduced")
    components = q.T
    components = _canonicalize_row_signs(components)
    components = np.asarray(components, dtype=dtype)
    components.setflags(write=False)
    return components


def _canonicalize_row_signs(matrix: FloatArray) -> FloatArray:
    canonical = np.asarray(matrix, dtype="float64").copy()
    for row in canonical:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return canonical


def _validate_components(
    value: object,
    *,
    output_dim: int,
    input_dim: int,
    dtype: str,
) -> FloatArray:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = "snapshot state.components must be a matrix sequence"
        raise ValueError(msg)
    try:
        matrix = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        msg = "snapshot state.components must contain only numeric values"
        raise ValueError(msg) from exc
    expected_shape = (output_dim, input_dim)
    if matrix.shape != expected_shape:
        msg = f"snapshot state.components must have shape {expected_shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(matrix)):
        msg = "snapshot state.components must contain only finite values"
        raise ValueError(msg)
    restored = matrix.copy()
    restored.setflags(write=False)
    return restored


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


def _validate_seed(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        msg = "seed must be None or an integer"
        raise ValueError(msg)
    return value


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


def _freeze_snapshot_mapping(snapshot: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {str(key): _freeze_snapshot_value(value) for key, value in snapshot.items()}
    )


def _freeze_snapshot_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return tuple(_freeze_snapshot_value(item) for item in value.tolist())
    if isinstance(value, Mapping):
        return _freeze_snapshot_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_snapshot_value(item) for item in value)
    msg = f"unsupported Oja compressor snapshot value: {type(value).__name__}"
    raise TypeError(msg)


__all__ = [
    "OJA_COMPRESSOR_NAME",
    "OJA_COMPRESSOR_SNAPSHOT_SCHEMA_VERSION",
    "OjaCompressor",
    "OjaCompressorSnapshot",
]
