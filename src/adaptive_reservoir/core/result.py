"""Result objects returned by the public adaptive-reservoir API."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from adaptive_reservoir.core.state import ReservoirState

if TYPE_CHECKING:
    from adaptive_reservoir.diagnostics import TraceNorms


@dataclass(frozen=True, slots=True)
class AdaptiveChannels:
    """Normalized adaptive state channels.

    All values must stay in the inclusive range [0.0, 1.0]. Channel objects are
    strict by design: invalid channel calculators should fail loudly instead of
    being silently clamped.
    """

    novelty: float = 0.0
    stability: float = 1.0
    drift_pressure: float = 0.0
    confidence: float = 0.0
    saturation: float = 0.0

    def __post_init__(self) -> None:
        _validate_channel("novelty", self.novelty)
        _validate_channel("stability", self.stability)
        _validate_channel("drift_pressure", self.drift_pressure)
        _validate_channel("confidence", self.confidence)
        _validate_channel("saturation", self.saturation)


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Runtime metrics for a single adaptive step."""

    samples_seen: int
    prediction_available: bool = False
    target_available: bool = False
    readout_updated: bool = False
    state_norm: float | None = None
    state_delta: float | None = None
    feature_norm: float | None = None
    saturation_rate: float | None = None
    trace_norms: TraceNorms | None = None
    prediction_error: float | None = None
    us_per_sample: float | None = None

    def __post_init__(self) -> None:
        if self.samples_seen < 0:
            msg = "samples_seen must be non-negative"
            raise ValueError(msg)
        _validate_optional_non_negative("state_norm", self.state_norm)
        _validate_optional_non_negative("state_delta", self.state_delta)
        _validate_optional_non_negative("feature_norm", self.feature_norm)
        _validate_optional_channel("saturation_rate", self.saturation_rate)
        _validate_optional_non_negative("prediction_error", self.prediction_error)
        _validate_optional_non_negative("us_per_sample", self.us_per_sample)


@dataclass(frozen=True, slots=True)
class AdaptiveStepResult:
    """Output returned by :meth:`adaptive_reservoir.AdaptiveReservoir.step`."""

    prediction: float | None
    features: tuple[float, ...]
    channels: AdaptiveChannels
    metrics: StepMetrics
    state: ReservoirState | None = None

    def __post_init__(self) -> None:
        if self.prediction is not None and not math.isfinite(self.prediction):
            msg = "prediction must be finite when provided"
            raise ValueError(msg)
        if not all(math.isfinite(feature) for feature in self.features):
            msg = "features must contain only finite values"
            raise ValueError(msg)


def _validate_channel(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        msg = f"{name} must be in the range [0.0, 1.0]"
        raise ValueError(msg)


def _validate_optional_channel(name: str, value: float | None) -> None:
    if value is None:
        return
    _validate_channel(name, value)


def _validate_optional_non_negative(name: str, value: float | None) -> None:
    if value is None:
        return
    if not math.isfinite(value) or value < 0.0:
        msg = f"{name} must be finite and non-negative"
        raise ValueError(msg)
