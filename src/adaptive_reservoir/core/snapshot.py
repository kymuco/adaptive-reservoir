"""Immutable runtime snapshots for adaptive-reservoir."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_reservoir.channels import ChannelCalculatorSnapshot
from adaptive_reservoir.core.metrics import AdaptiveReservoirMetricsSnapshot
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


def clone_reservoir_state(state: ReservoirState) -> ReservoirState:
    """Return an independent copy of a reservoir state."""

    return ReservoirState(
        activations=np.array(state.activations, copy=True),
        fast_trace=np.array(state.fast_trace, copy=True),
        mid_trace=np.array(state.mid_trace, copy=True),
        slow_trace=np.array(state.slow_trace, copy=True),
        samples_seen=state.samples_seen,
    )
