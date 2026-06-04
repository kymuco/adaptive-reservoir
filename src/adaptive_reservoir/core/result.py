"""Result objects returned by the public adaptive-reservoir API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AdaptiveChannels:
    """Normalized adaptive state channels.

    All values are expected to stay in the inclusive range [0.0, 1.0]. PR0.3
    returns neutral defaults; later milestones will implement channel logic.
    """

    novelty: float = 0.0
    stability: float = 1.0
    drift_pressure: float = 0.0
    confidence: float = 0.0
    saturation: float = 0.0


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Runtime metrics for a single adaptive step."""

    samples_seen: int
    prediction_available: bool = False
    target_available: bool = False
    us_per_sample: float | None = None


@dataclass(frozen=True, slots=True)
class AdaptiveStepResult:
    """Output returned by :meth:`adaptive_reservoir.AdaptiveReservoir.step`."""

    prediction: float | None
    features: tuple[float, ...]
    channels: AdaptiveChannels
    metrics: StepMetrics
    extra: dict[str, Any] | None = None
