"""Generic behavior adapter example without application-specific imports.

Run from a source checkout with:

    python examples/generic_behavior_adapter.py

The host application owns event semantics. This example converts generic
operational behavior events into numeric vectors before passing them into
``AdaptiveReservoir``.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))

from adaptive_reservoir import AdaptiveReservoir, ReadoutConfig, ReservoirConfig  # noqa: E402
from adaptive_reservoir.adapters import EventVectorizer, FloatArray  # noqa: E402


@dataclass(frozen=True, slots=True)
class BehaviorEvent:
    """Small generic event owned by a host application."""

    latency_ms: float
    error_count: int
    activity_score: float
    target_score: float


class GenericBehaviorVectorizer:
    """Convert generic behavior events into numeric reservoir inputs."""

    def __init__(
        self,
        *,
        latency_scale_ms: float = 1_000.0,
        error_count_scale: float = 10.0,
    ) -> None:
        if latency_scale_ms <= 0.0 or not math.isfinite(latency_scale_ms):
            msg = "latency_scale_ms must be finite and positive"
            raise ValueError(msg)
        if error_count_scale <= 0.0 or not math.isfinite(error_count_scale):
            msg = "error_count_scale must be finite and positive"
            raise ValueError(msg)
        self._latency_scale_ms = latency_scale_ms
        self._error_count_scale = error_count_scale

    def transform(self, event: BehaviorEvent) -> FloatArray:
        """Return a 1D finite numeric vector for one behavior event."""

        _validate_event(event)
        return np.asarray(
            (
                event.latency_ms / self._latency_scale_ms,
                float(event.error_count) / self._error_count_scale,
                event.activity_score,
            ),
            dtype=np.float64,
        )


def build_model(*, seed: int = 0, input_dim: int = 3) -> AdaptiveReservoir:
    """Build a small reservoir suitable for the generic behavior vectorizer."""

    return AdaptiveReservoir(
        ReservoirConfig(
            input_dim=input_dim,
            n_cells=16,
            topology="ring_shortcuts",
            feature_mode="state_slow_raw",
            seed=seed,
            readout=ReadoutConfig(name="nlms", update_interval=1),
        )
    )


def default_events() -> tuple[BehaviorEvent, ...]:
    """Return a tiny deterministic event stream for the example."""

    return (
        BehaviorEvent(
            latency_ms=120.0,
            error_count=0,
            activity_score=0.20,
            target_score=0.25,
        ),
        BehaviorEvent(
            latency_ms=180.0,
            error_count=1,
            activity_score=0.35,
            target_score=0.30,
        ),
        BehaviorEvent(
            latency_ms=90.0,
            error_count=0,
            activity_score=0.55,
            target_score=0.50,
        ),
        BehaviorEvent(
            latency_ms=240.0,
            error_count=2,
            activity_score=0.40,
            target_score=0.35,
        ),
    )


def run_example(events: Sequence[BehaviorEvent] | None = None) -> tuple[float, ...]:
    """Run the generic adapter example and return one prediction per event."""

    event_stream = default_events() if events is None else tuple(events)
    vectorizer: EventVectorizer[BehaviorEvent] = GenericBehaviorVectorizer()
    model = build_model(input_dim=3)
    predictions: list[float] = []
    for event in event_stream:
        vector = vectorizer.transform(event)
        result = model.step(vector, target=event.target_score)
        predictions.append(result.prediction)
    return tuple(predictions)


def main() -> int:
    """Run the example and print stable key-value output."""

    predictions = run_example()
    for index, prediction in enumerate(predictions):
        print(f"prediction_{index}: {prediction:.6f}")
    return 0


def _validate_event(event: BehaviorEvent) -> None:
    if not isinstance(event, BehaviorEvent):
        msg = "event must be a BehaviorEvent"
        raise TypeError(msg)
    if event.error_count < 0:
        msg = "event.error_count must be non-negative"
        raise ValueError(msg)
    for name, value in (
        ("latency_ms", event.latency_ms),
        ("activity_score", event.activity_score),
        ("target_score", event.target_score),
    ):
        if not math.isfinite(value):
            msg = f"event.{name} must be finite"
            raise ValueError(msg)


__all__ = [
    "BehaviorEvent",
    "GenericBehaviorVectorizer",
    "build_model",
    "default_events",
    "main",
    "run_example",
]

if __name__ == "__main__":
    raise SystemExit(main())
