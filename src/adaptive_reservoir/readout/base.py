"""Base contracts and validation helpers for scalar readouts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from adaptive_reservoir.core.serialization import (
    json_friendly,
    require_int,
    require_mapping,
    require_str,
)

FloatArray = NDArray[np.floating]
READOUT_SNAPSHOT_SCHEMA_VERSION = 1


@runtime_checkable
class ReadoutProtocol(Protocol):
    """Protocol for scalar online readouts.

    Runtime checks verify method presence only. They do not fully validate method
    signatures or readout state semantics.
    """

    def predict(self, features: FloatArray) -> float:
        """Return a scalar prediction for the given feature vector."""

        ...

    def update(self, features: FloatArray, target: float) -> None:
        """Update readout parameters from a supervised target."""

        ...

    def snapshot(self) -> Mapping[str, object]:
        """Return a numeric readout snapshot."""

        ...

    def restore(self, snapshot: Mapping[str, object]) -> None:
        """Restore readout state from a numeric snapshot."""

        ...


@dataclass(frozen=True, slots=True)
class ReadoutSnapshot:
    """Immutable base container for readout state snapshots."""

    schema_version: int
    name: str
    state: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly readout snapshot dictionary."""

        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "state": json_friendly(self.state),
        }

    @classmethod
    def from_dict(cls, data: object) -> ReadoutSnapshot:
        """Create a readout snapshot from a JSON-friendly mapping."""

        mapping = require_mapping(data, "readout")
        return cls(
            schema_version=require_int(mapping, "schema_version"),
            name=require_str(mapping, "name"),
            state=require_mapping(mapping.get("state"), "state"),
        )

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        _validate_name(self.name)
        state = validate_snapshot_mapping(self.state)
        frozen_state = _freeze_snapshot_mapping(state)
        object.__setattr__(self, "state", frozen_state)


def validate_features(
    features: object,
    *,
    expected_dim: int | None = None,
    dtype: str = "float64",
) -> FloatArray:
    """Validate and return a read-only 1D floating feature vector."""

    if expected_dim is not None:
        _validate_expected_dim(expected_dim)
    if isinstance(features, (str, bytes)):
        msg = "features must be a 1D numeric vector"
        raise ValueError(msg)
    try:
        array = np.asarray(features, dtype=dtype)
    except (TypeError, ValueError) as exc:
        msg = "features must contain only numeric values"
        raise ValueError(msg) from exc
    if array.ndim != 1:
        msg = "features must be a 1D numeric vector"
        raise ValueError(msg)
    if array.size == 0:
        msg = "features must not be empty"
        raise ValueError(msg)
    if not np.issubdtype(array.dtype, np.floating):
        msg = "features must have a floating dtype"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = "features must contain only finite values"
        raise ValueError(msg)
    if expected_dim is not None and array.size != expected_dim:
        msg = f"expected feature_dim={expected_dim}, got {array.size}"
        raise ValueError(msg)

    readonly = array.astype(array.dtype, copy=True)
    readonly.setflags(write=False)
    return readonly


def validate_target(target: object) -> float:
    """Validate and return a finite scalar target value."""

    if isinstance(target, (str, bytes)):
        msg = "target must be numeric"
        raise ValueError(msg)
    try:
        value = float(target)
    except (TypeError, ValueError) as exc:
        msg = "target must be numeric"
        raise ValueError(msg) from exc
    if not math.isfinite(value):
        msg = "target must be finite"
        raise ValueError(msg)
    return value


def validate_snapshot_mapping(snapshot: object) -> Mapping[str, object]:
    """Validate that a readout snapshot payload is mapping-shaped."""

    if not isinstance(snapshot, Mapping):
        msg = "snapshot must be a mapping"
        raise ValueError(msg)
    return snapshot


def _freeze_snapshot_mapping(snapshot: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {str(key): _freeze_snapshot_value(value) for key, value in snapshot.items()}
    )


def _freeze_snapshot_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = "readout snapshot float values must be finite"
            raise ValueError(msg)
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        if not math.isfinite(result):
            msg = "readout snapshot float values must be finite"
            raise ValueError(msg)
        return result
    if isinstance(value, np.ndarray):
        return tuple(_freeze_snapshot_value(item) for item in value.tolist())
    if isinstance(value, Mapping):
        return _freeze_snapshot_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_snapshot_value(item) for item in value)
    msg = f"unsupported readout snapshot value: {type(value).__name__}"
    raise TypeError(msg)


def _validate_schema_version(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = "schema_version must be a positive integer"
        raise ValueError(msg)


def _validate_name(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        msg = "name must be a non-empty string"
        raise ValueError(msg)


def _validate_expected_dim(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = "expected_dim must be a positive integer"
        raise ValueError(msg)
