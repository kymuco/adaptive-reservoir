"""Base contracts and validation helpers for scalar readouts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

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

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        _validate_name(self.name)
        state = validate_snapshot_mapping(self.state)
        object.__setattr__(self, "state", MappingProxyType(dict(state)))


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
