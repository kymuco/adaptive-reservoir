"""CPU-friendly temporal adaptation layer for software agents and streaming systems."""

from adaptive_reservoir.core.adaptive_reservoir import AdaptiveReservoir
from adaptive_reservoir.core.config import (
    ChannelConfig,
    ReadoutConfig,
    ReservoirConfig,
    TraceConfig,
)
from adaptive_reservoir.core.result import AdaptiveChannels, AdaptiveStepResult, StepMetrics
from adaptive_reservoir.core.state import ReservoirState

__version__ = "0.0.0"

__all__ = [
    "AdaptiveChannels",
    "AdaptiveReservoir",
    "AdaptiveStepResult",
    "ChannelConfig",
    "ReadoutConfig",
    "ReservoirConfig",
    "ReservoirState",
    "StepMetrics",
    "TraceConfig",
    "__version__",
]
