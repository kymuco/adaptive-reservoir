"""Structural protocols for adaptive-reservoir extension points."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from adaptive_reservoir.core.config import ReservoirConfig
from adaptive_reservoir.core.result import AdaptiveChannels
from adaptive_reservoir.core.state import ReservoirState

FloatArray = NDArray[np.floating]


@runtime_checkable
class ReadoutProtocol(Protocol):
    """Protocol for scalar online readouts.

    Runtime checks verify method presence only. They do not fully validate method
    signatures.
    """

    def predict(self, features: FloatArray) -> float:
        """Return a scalar prediction for the given feature vector."""

        ...

    def update(self, features: FloatArray, target: float) -> None:
        """Update readout parameters from a supervised target."""

        ...

    def snapshot(self) -> Mapping[str, object]:
        """Return a serializable readout snapshot."""

        ...

    def restore(self, snapshot: Mapping[str, object]) -> None:
        """Restore readout state from a snapshot."""

        ...


@runtime_checkable
class TopologyBuilderProtocol(Protocol):
    """Protocol for recurrent topology builders."""

    def build(self, config: ReservoirConfig) -> FloatArray:
        """Build a recurrent topology matrix for the given config."""

        ...


@runtime_checkable
class FeatureExtractorProtocol(Protocol):
    """Protocol for reservoir feature extractors."""

    def extract(self, state: ReservoirState, x: FloatArray) -> FloatArray:
        """Extract a feature vector from reservoir state and current input."""

        ...


@runtime_checkable
class ChannelCalculatorProtocol(Protocol):
    """Protocol for adaptive channel calculators."""

    def update(
        self,
        x: FloatArray,
        state: ReservoirState,
        features: FloatArray,
        prediction: float | None,
        target: float | None = None,
    ) -> AdaptiveChannels:
        """Update internal channel state and return normalized adaptive channels."""

        ...
