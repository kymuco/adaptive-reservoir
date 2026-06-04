"""CPU-friendly temporal adaptation layer for software agents and streaming systems."""

from adaptive_reservoir.core import AdaptiveChannels
from adaptive_reservoir.core import AdaptiveReservoir
from adaptive_reservoir.core import AdaptiveStepResult
from adaptive_reservoir.core import ReservoirConfig
from adaptive_reservoir.core import StepMetrics

__version__ = "0.0.0"

__all__ = [
    "AdaptiveChannels",
    "AdaptiveReservoir",
    "AdaptiveStepResult",
    "ReservoirConfig",
    "StepMetrics",
    "__version__",
]
