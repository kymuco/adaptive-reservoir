"""Snapshot and restore helpers for mathematical reservoir runtime state."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from adaptive_reservoir.core.protocols import FloatArray
from adaptive_reservoir.core.state import ReservoirState

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_API_STAGE = "snapshot_restore_v1"


def snapshot_state(state: ReservoirState) -> dict[str, object]:
    """Return a JSON-friendly snapshot of mathematical reservoir state."""

    return {
        "n_cells": int(state.activations.shape[0]),
        "dtype": state.activations.dtype.name,
        "samples_seen": state.samples_seen,
        "activations": _array_to_list(state.activations),
        "fast_trace": _array_to_list(state.fast_trace),
        "mid_trace": _array_to_list(state.mid_trace),
        "slow_trace": _array_to_list(state.slow_trace),
    }


def restore_state(
    snapshot: object,
    *,
    expected_n_cells: int,
    dtype: str,
) -> ReservoirState:
    """Restore ``ReservoirState`` from a validated state snapshot mapping."""

    if not isinstance(snapshot, Mapping):
        msg = "snapshot state must be a mapping"
        raise ValueError(msg)
    n_cells = _required_int(snapshot, "n_cells")
    if n_cells != expected_n_cells:
        msg = (
            "snapshot n_cells must match "
            f"config.n_cells={expected_n_cells}; got {n_cells}"
        )
        raise ValueError(msg)
    snapshot_dtype = _required_str(snapshot, "dtype")
    if snapshot_dtype != dtype:
        msg = (
            "snapshot dtype must match "
            f"config.dtype={dtype!r}; got {snapshot_dtype!r}"
        )
        raise ValueError(msg)
    samples_seen = _required_int(snapshot, "samples_seen")
    if samples_seen < 0:
        msg = "snapshot samples_seen must be non-negative"
        raise ValueError(msg)

    return ReservoirState(
        activations=_array_from_snapshot(
            snapshot.get("activations"),
            name="activations",
            dtype=dtype,
            n_cells=n_cells,
        ),
        fast_trace=_array_from_snapshot(
            snapshot.get("fast_trace"),
            name="fast_trace",
            dtype=dtype,
            n_cells=n_cells,
        ),
        mid_trace=_array_from_snapshot(
            snapshot.get("mid_trace"),
            name="mid_trace",
            dtype=dtype,
            n_cells=n_cells,
        ),
        slow_trace=_array_from_snapshot(
            snapshot.get("slow_trace"),
            name="slow_trace",
            dtype=dtype,
            n_cells=n_cells,
        ),
        samples_seen=samples_seen,
    )


def validate_runtime_snapshot(snapshot: object) -> Mapping[str, object]:
    """Validate top-level runtime snapshot metadata and placeholders."""

    if not isinstance(snapshot, Mapping):
        msg = "snapshot must be a mapping"
        raise ValueError(msg)
    schema_version = snapshot.get("schema_version")
    if schema_version != SNAPSHOT_SCHEMA_VERSION:
        msg = f"unsupported snapshot schema_version: {schema_version!r}"
        raise ValueError(msg)
    if "state" not in snapshot:
        msg = "snapshot state is required"
        raise ValueError(msg)
    readout_state = snapshot.get("readout_state")
    if readout_state is not None:
        msg = "readout_state restore is not supported yet"
        raise ValueError(msg)
    metrics_buffers = snapshot.get("metrics_buffers")
    if not isinstance(metrics_buffers, Mapping):
        msg = "metrics_buffers must be a mapping"
        raise ValueError(msg)
    if metrics_buffers:
        msg = "metrics_buffers restore is not supported yet when non-empty"
        raise ValueError(msg)
    return snapshot


def _array_to_list(values: FloatArray) -> list[float]:
    return [float(value) for value in values]


def _array_from_snapshot(
    values: object,
    *,
    name: str,
    dtype: str,
    n_cells: int,
) -> FloatArray:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        msg = f"snapshot state.{name} must be a sequence"
        raise ValueError(msg)
    array = np.asarray(
        tuple(_finite_float(value, name=name) for value in values),
        dtype=dtype,
    )
    if array.shape != (n_cells,):
        msg = f"snapshot state.{name} must have length {n_cells}; got {array.size}"
        raise ValueError(msg)
    return array


def _required_int(snapshot: Mapping[str, object], key: str) -> int:
    value = snapshot.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"snapshot state.{key} must be an integer"
        raise ValueError(msg)
    return value


def _required_str(snapshot: Mapping[str, object], key: str) -> str:
    value = snapshot.get(key)
    if not isinstance(value, str):
        msg = f"snapshot state.{key} must be a string"
        raise ValueError(msg)
    return value


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"snapshot state.{name} must contain only numeric values"
        raise ValueError(msg) from exc
    if not math.isfinite(result):
        msg = f"snapshot state.{name} must contain only finite values"
        raise ValueError(msg)
    return result
