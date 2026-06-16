"""Core public API objects."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AdaptiveChannels",
    "AdaptiveReservoir",
    "AdaptiveReservoirMetricsSnapshot",
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

_EXPORTS = {
    "AdaptiveChannels": "adaptive_reservoir.core.result",
    "AdaptiveReservoir": "adaptive_reservoir.core.adaptive_reservoir",
    "AdaptiveReservoirMetricsSnapshot": "adaptive_reservoir.core.metrics",
    "AdaptiveStepResult": "adaptive_reservoir.core.result",
    "ChannelCalculatorProtocol": "adaptive_reservoir.core.protocols",
    "ChannelConfig": "adaptive_reservoir.core.config",
    "FeatureExtractorProtocol": "adaptive_reservoir.core.protocols",
    "ReadoutConfig": "adaptive_reservoir.core.config",
    "ReadoutProtocol": "adaptive_reservoir.readout",
    "ReadoutSnapshot": "adaptive_reservoir.readout",
    "ReservoirConfig": "adaptive_reservoir.core.config",
    "ReservoirCore": "adaptive_reservoir.core.reservoir",
    "ReservoirSnapshot": "adaptive_reservoir.core.snapshot",
    "ReservoirState": "adaptive_reservoir.core.state",
    "StateDiagnostics": "adaptive_reservoir.diagnostics",
    "StepMetrics": "adaptive_reservoir.core.result",
    "TopologyBuilderProtocol": "adaptive_reservoir.core.protocols",
    "TraceConfig": "adaptive_reservoir.core.config",
    "TraceNorms": "adaptive_reservoir.diagnostics",
    "calculate_state_diagnostics": "adaptive_reservoir.diagnostics",
    "extract_features": "adaptive_reservoir.features",
    "rms_norm": "adaptive_reservoir.diagnostics",
}


def __getattr__(name: str) -> Any:
    """Load core exports lazily to avoid package-level import cycles."""

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
