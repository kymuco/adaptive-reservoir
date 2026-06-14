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
    TopologyBuilderProtocol,
)
from adaptive_reservoir.core.reservoir import ReservoirCore
from adaptive_reservoir.core.result import AdaptiveChannels, AdaptiveStepResult, StepMetrics
from adaptive_reservoir.core.snapshot import ReservoirSnapshot
from adaptive_reservoir.core.state import ReservoirState
from adaptive_reservoir.diagnostics import (
    StateDiagnostics,
    TraceNorms,
    calculate_state_diagnostics,
    rms_norm,
)
from adaptive_reservoir.features import extract_features
from adaptive_reservoir.readout import ReadoutProtocol, ReadoutSnapshot

__all__ = [
    "AdaptiveChannels",
    "AdaptiveReservoir",
    "AdaptiveStepResult",
    "ChannelCalculatorProtocol",
    "ChannelConfig",
    "FeatureExtractorProtocol",
    "ReadoutConfig",
    "ReadoutProtocol",
    "ReadoutSnapshot",
    "ReservoirConfig",
    "ReservoirCore",
    "ReservoirSnapshot",
    "ReservoirState",
    "StateDiagnostics",
    "StepMetrics",
    "TopologyBuilderProtocol",
    "TraceConfig",
    "TraceNorms",
    "calculate_state_diagnostics",
    "extract_features",
    "rms_norm",
]
