"""Structural protocols for adaptive-reservoir extension points."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from adaptive_reservoir.core.config import ReservoirConfig
from adaptive_reservoir.core.result import AdaptiveChannels
from adaptive_reservoir.core.state import ReservoirState
from adaptive_reservoir.readout.base import ReadoutProtocol

if TYPE_CHECKING:
    from adaptive_reservoir.topology.edges import EdgeList

FloatArray = NDArray[np.floating]


@runtime_checkable
class TopologyBuilderProtocol(Protocol):
    """Protocol for recurrent topology builders."""

    def build(self, config: ReservoirConfig) -> EdgeList:
        """Build recurrent topology edges for the given config."""

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
