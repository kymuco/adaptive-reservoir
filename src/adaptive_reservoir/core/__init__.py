"""Core public API objects."""

from adaptive_reservoir.core.adaptive_reservoir import AdaptiveReservoir
from adaptive_reservoir.core.config import (
    ChannelConfig,
    ReadoutConfig,
    ReservoirConfig,
    TraceConfig,
)
from adaptive_reservoir.core.protocols import (
    ChannelCalculatorProtocol,
    FeatureExtractorProtocol,
    ReadoutProtocol,
    TopologyBuilderProtocol,
)
from adaptive_reservoir.core.reservoir import ReservoirCore
from adaptive_reservoir.core.result import AdaptiveChannels, AdaptiveStepResult, StepMetrics
from adaptive_reservoir.core.state import ReservoirState
from adaptive_reservoir.features import extract_features

__all__ = [
    "AdaptiveChannels",
    "AdaptiveReservoir",
    "AdaptiveStepResult",
    "ChannelCalculatorProtocol",
    "ChannelConfig",
    "FeatureExtractorProtocol",
    "ReadoutConfig",
    "ReadoutProtocol",
    "ReservoirConfig",
    "ReservoirCore",
    "ReservoirState",
    "StepMetrics",
    "TopologyBuilderProtocol",
    "TraceConfig",
    "extract_features",
]
