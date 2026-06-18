"""Generic helpers for JSON-friendly dict serialization."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TypeAlias

import numpy as np

JsonValue: TypeAlias = object


def require_mapping(data: object, name: str) -> Mapping[str, object]:
    """Return a mapping or raise a clear ValueError."""

    if not isinstance(data, Mapping):
        msg = f"{name} must be a mapping"
        raise ValueError(msg)
    return data


def require_int(data: Mapping[str, object], key: str) -> int:
    """Return a required integer field."""

    if key not in data:
        msg = f"missing required field: {key}"
        raise ValueError(msg)
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key} must be an integer"
        raise ValueError(msg)
    return value


def optional_int(data: Mapping[str, object], key: str, default: int) -> int:
    """Return an optional integer field."""

    if key not in data:
        return default
    return require_int(data, key)


def require_float(data: Mapping[str, object], key: str) -> float:
    """Return a required finite float-compatible field."""

    if key not in data:
        msg = f"missing required field: {key}"
        raise ValueError(msg)
    return finite_float(data[key], key)


def optional_float(data: Mapping[str, object], key: str, default: float) -> float:
    """Return an optional finite float-compatible field."""

    if key not in data:
        return default
    return require_float(data, key)


def require_str(data: Mapping[str, object], key: str) -> str:
    """Return a required string field."""

    if key not in data:
        msg = f"missing required field: {key}"
        raise ValueError(msg)
    value = data[key]
    if not isinstance(value, str):
        msg = f"{key} must be a string"
        raise ValueError(msg)
    return value


def optional_str(data: Mapping[str, object], key: str, default: str) -> str:
    """Return an optional string field."""

    if key not in data:
        return default
    return require_str(data, key)


def optional_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object] | None:
    """Return an optional nested mapping."""

    if key not in data:
        return None
    return require_mapping(data[key], key)


def require_sequence(data: Mapping[str, object], key: str) -> Sequence[object]:
    """Return a required non-string sequence field."""

    if key not in data:
        msg = f"missing required field: {key}"
        raise ValueError(msg)
    value = data[key]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{key} must be a sequence"
        raise ValueError(msg)
    return value


def optional_int_or_none(
    data: Mapping[str, object],
    key: str,
    default: int | None,
) -> int | None:
    """Return an optional integer-or-null field."""

    if key not in data:
        return default
    value = data[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key} must be an integer or null"
        raise ValueError(msg)
    return value


def finite_float(value: object, name: str) -> float:
    """Return a finite float and reject bools/non-numeric values."""

    if isinstance(value, bool):
        msg = f"{name} must be numeric"
        raise ValueError(msg)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be numeric"
        raise ValueError(msg) from exc
    if not math.isfinite(result):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return result


def numeric_sequence_to_tuple(value: object, name: str) -> tuple[float, ...]:
    """Return a tuple of finite floats from a non-string sequence."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{name} must be a sequence"
        raise ValueError(msg)
    return tuple(finite_float(item, f"{name} values") for item in value)


def nested_numeric_sequence_to_tuple(
    value: object,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    """Return a tuple of numeric tuples from a nested sequence."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{name} must be a sequence"
        raise ValueError(msg)
    return tuple(numeric_sequence_to_tuple(row, f"{name} rows") for row in value)


def optional_numeric_sequence_to_tuple(
    value: object,
    name: str,
) -> tuple[float, ...] | None:
    """Return None or a tuple of finite floats."""

    if value is None:
        return None
    return numeric_sequence_to_tuple(value, name)


def json_friendly(value: object) -> JsonValue:
    """Recursively convert supported numeric containers to JSON-friendly values."""

    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = "float values must be finite"
            raise ValueError(msg)
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        if not math.isfinite(result):
            msg = "float values must be finite"
            raise ValueError(msg)
        return result
    if isinstance(value, np.ndarray):
        return [json_friendly(item) for item in value.tolist()]
    if isinstance(value, MappingProxyType) or isinstance(value, Mapping):
        return {str(key): json_friendly(item) for key, item in value.items()}
    if isinstance(value, tuple) or isinstance(value, list):
        return [json_friendly(item) for item in value]
    msg = f"unsupported JSON-friendly value: {type(value).__name__}"
    raise TypeError(msg)


__all__ = [
    "JsonValue",
    "finite_float",
    "json_friendly",
    "nested_numeric_sequence_to_tuple",
    "numeric_sequence_to_tuple",
    "optional_float",
    "optional_int",
    "optional_int_or_none",
    "optional_mapping",
    "optional_numeric_sequence_to_tuple",
    "optional_str",
    "require_float",
    "require_int",
    "require_mapping",
    "require_sequence",
    "require_str",
]
