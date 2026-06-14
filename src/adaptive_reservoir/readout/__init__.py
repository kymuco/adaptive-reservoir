"""Readout contracts and implementations."""

from adaptive_reservoir.readout.base import (
    READOUT_SNAPSHOT_SCHEMA_VERSION,
    ReadoutProtocol,
    ReadoutSnapshot,
    validate_features,
    validate_snapshot_mapping,
    validate_target,
)
from adaptive_reservoir.readout.factory import create_readout
from adaptive_reservoir.readout.nlms import NLMSReadout
from adaptive_reservoir.readout.replay_ridge import ReplayRidgeReadout
from adaptive_reservoir.readout.sliding_ridge import SlidingWindowRidgeReadout

__all__ = [
    "NLMSReadout",
    "READOUT_SNAPSHOT_SCHEMA_VERSION",
    "ReadoutProtocol",
    "ReadoutSnapshot",
    "ReplayRidgeReadout",
    "SlidingWindowRidgeReadout",
    "create_readout",
    "validate_features",
    "validate_snapshot_mapping",
    "validate_target",
]
