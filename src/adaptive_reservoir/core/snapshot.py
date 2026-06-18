"""Immutable runtime snapshots for adaptive-reservoir."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_reservoir.channels import ChannelCalculatorSnapshot
from adaptive_reservoir.core.metrics import AdaptiveReservoirMetricsSnapshot
from adaptive_reservoir.core.serialization import require_int, require_mapping
from adaptive_reservoir.core.state import ReservoirState
from adaptive_reservoir.readout.base import ReadoutSnapshot

SNAPSHOT_SCHEMA_VERSION = 4


@dataclass(frozen=True, slots=True, kw_only=True)
class ReservoirSnapshot:
    """Immutable numeric checkpoint for reservoir, readout, channels, and metrics."""

    state: ReservoirState
    readout: ReadoutSnapshot
    channels: ChannelCalculatorSnapshot
    metrics: AdaptiveReservoirMetricsSnapshot
    schema_version: int = SNAPSHOT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly reservoir snapshot dictionary."""

        return {
            "schema_version": self.schema_version,
            "state": self.state.to_dict(),
            "readout": self.readout.to_dict(),
            "channels": self.channels.to_dict(),
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: object) -> ReservoirSnapshot:
        """Create a reservoir snapshot from a JSON-friendly mapping."""

        mapping = require_mapping(data, "snapshot")
        schema_version = require_int(mapping, "schema_version")
        if schema_version != SNAPSHOT_SCHEMA_VERSION:
            msg = f"unsupported snapshot schema_version: {schema_version}"
            raise ValueError(msg)
        return cls(
            schema_version=schema_version,
            state=ReservoirState.from_dict(require_mapping(mapping.get("state"), "state")),
            readout=ReadoutSnapshot.from_dict(
                require_mapping(mapping.get("readout"), "readout"),
            ),
            channels=ChannelCalculatorSnapshot.from_dict(
                require_mapping(mapping.get("channels"), "channels"),
            ),
            metrics=AdaptiveReservoirMetricsSnapshot.from_dict(
                require_mapping(mapping.get("metrics"), "metrics"),
            ),
        )


def clone_reservoir_state(state: ReservoirState) -> ReservoirState:
    """Return an independent copy of a reservoir state."""

    return ReservoirState(
        activations=np.array(state.activations, copy=True),
        fast_trace=np.array(state.fast_trace, copy=True),
        mid_trace=np.array(state.mid_trace, copy=True),
        slow_trace=np.array(state.slow_trace, copy=True),
        samples_seen=state.samples_seen,
    )
