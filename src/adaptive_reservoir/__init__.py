"""CPU-friendly temporal adaptation layer for software agents and streaming systems."""

from adaptive_reservoir.core.adaptive_reservoir import AdaptiveReservoir
from adaptive_reservoir.core.config import ReservoirConfig
from adaptive_reservoir.core.result import AdaptiveChannels, AdaptiveStepResult, StepMetrics

__version__ = "0.0.0"

__all__ = [
    "AdaptiveChannels",
    "AdaptiveReservoir",
    "AdaptiveStepResult",
    "ReservoirConfig",
    "StepMetrics",
    "__version__",
]
