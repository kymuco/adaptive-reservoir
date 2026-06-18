"""Aggregate runtime metrics for the public AdaptiveReservoir facade."""

from __future__ import annotations

import math
from dataclasses import dataclass

from adaptive_reservoir.core.serialization import (
    optional_int,
    require_float,
    require_int,
    require_mapping,
)

ADAPTIVE_RESERVOIR_METRICS_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class AdaptiveReservoirMetricsSnapshot:
    """Immutable aggregate runtime metrics snapshot."""

    samples_seen: int
    us_per_sample_avg: float = 0.0
    readout_update_count: int = 0
    readout_solve_count: int = 0
    saturation_rate_avg: float = 0.0
    schema_version: int = ADAPTIVE_RESERVOIR_METRICS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly metrics snapshot dictionary."""

        return {
            "schema_version": self.schema_version,
            "samples_seen": self.samples_seen,
            "us_per_sample_avg": self.us_per_sample_avg,
            "readout_update_count": self.readout_update_count,
            "readout_solve_count": self.readout_solve_count,
            "saturation_rate_avg": self.saturation_rate_avg,
        }

    @classmethod
    def from_dict(cls, data: object) -> AdaptiveReservoirMetricsSnapshot:
        """Create metrics snapshot from a JSON-friendly mapping."""

        mapping = require_mapping(data, "metrics")
        schema_version = optional_int(
            mapping,
            "schema_version",
            ADAPTIVE_RESERVOIR_METRICS_SCHEMA_VERSION,
        )
        if schema_version != ADAPTIVE_RESERVOIR_METRICS_SCHEMA_VERSION:
            msg = f"unsupported metrics schema_version: {schema_version}"
            raise ValueError(msg)
        return cls(
            schema_version=schema_version,
            samples_seen=require_int(mapping, "samples_seen"),
            us_per_sample_avg=require_float(mapping, "us_per_sample_avg"),
            readout_update_count=require_int(mapping, "readout_update_count"),
            readout_solve_count=require_int(mapping, "readout_solve_count"),
            saturation_rate_avg=require_float(mapping, "saturation_rate_avg"),
        )

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        _validate_non_negative_int("samples_seen", self.samples_seen)
        _validate_non_negative_finite("us_per_sample_avg", self.us_per_sample_avg)
        _validate_non_negative_int("readout_update_count", self.readout_update_count)
        _validate_non_negative_int("readout_solve_count", self.readout_solve_count)
        _validate_channel_average("saturation_rate_avg", self.saturation_rate_avg)
        if self.readout_update_count > self.samples_seen:
            msg = "readout_update_count must be less than or equal to samples_seen"
            raise ValueError(msg)


def _validate_schema_version(value: int) -> None:
    if value != ADAPTIVE_RESERVOIR_METRICS_SCHEMA_VERSION:
        msg = f"unsupported metrics schema_version: {value}"
        raise ValueError(msg)


def _validate_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{name} must be a non-negative integer"
        raise ValueError(msg)


def _validate_non_negative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        msg = f"{name} must be finite and non-negative"
        raise ValueError(msg)


def _validate_channel_average(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        msg = f"{name} must be in the range [0.0, 1.0]"
        raise ValueError(msg)


__all__ = [
    "ADAPTIVE_RESERVOIR_METRICS_SCHEMA_VERSION",
    "AdaptiveReservoirMetricsSnapshot",
]
