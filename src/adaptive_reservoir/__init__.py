"""CPU-friendly temporal adaptation layer for software agents and streaming systems."""

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
from adaptive_reservoir.core.snapshot import (
    SNAPSHOT_API_STAGE,
    SNAPSHOT_SCHEMA_VERSION,
    restore_state,
    snapshot_state,
    validate_runtime_snapshot,
)
from adaptive_reservoir.core.state import ReservoirState
from adaptive_reservoir.diagnostics import (
    StateDiagnostics,
    TraceNorms,
    calculate_state_diagnostics,
    rms_norm,
)
from adaptive_reservoir.features import extract_features

__version__ = "0.0.0"

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
    "SNAPSHOT_API_STAGE",
    "SNAPSHOT_SCHEMA_VERSION",
    "StateDiagnostics",
    "StepMetrics",
    "TopologyBuilderProtocol",
    "TraceConfig",
    "TraceNorms",
    "__version__",
    "calculate_state_diagnostics",
    "extract_features",
    "restore_state",
    "rms_norm",
    "snapshot_state",
    "validate_runtime_snapshot",
]
