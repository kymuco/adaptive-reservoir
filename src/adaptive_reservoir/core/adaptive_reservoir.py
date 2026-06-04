"""Public AdaptiveReservoir facade."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

from adaptive_reservoir.core.config import ReservoirConfig
from adaptive_reservoir.core.result import AdaptiveChannels, AdaptiveStepResult, StepMetrics


class AdaptiveReservoir:
    """High-level facade for the adaptive-reservoir public API.

    PR0.3 intentionally provides only a safe API draft. Real reservoir dynamics,
    feature extraction, readouts, and channel calculations will be implemented in
    later milestones.
    """

    def __init__(self, config: ReservoirConfig) -> None:
        self.config = config
        self._samples_seen = 0

    @property
    def samples_seen(self) -> int:
        """Number of processed samples."""

        return self._samples_seen

    def step(self, x: Sequence[float], target: float | None = None) -> AdaptiveStepResult:
        """Process one numeric event vector and return an adaptive step result."""

        start = time.perf_counter()
        values = self._validate_input(x)
        if target is not None and not math.isfinite(target):
            msg = "target must be finite when provided"
            raise ValueError(msg)

        self._samples_seen += 1
        elapsed_us = (time.perf_counter() - start) * 1_000_000.0
        return AdaptiveStepResult(
            prediction=None,
            features=values,
            channels=AdaptiveChannels(),
            metrics=StepMetrics(
                samples_seen=self._samples_seen,
                prediction_available=False,
                target_available=target is not None,
                us_per_sample=elapsed_us,
            ),
            extra={"api_stage": "draft"},
        )

    def reset(self) -> None:
        """Reset runtime counters.

        Later milestones will also reset reservoir state, traces, readout state,
        and channel buffers.
        """

        self._samples_seen = 0

    def snapshot(self) -> dict[str, object]:
        """Return a serializable snapshot of the current mathematical state."""

        return {
            "samples_seen": self._samples_seen,
            "config": self.config,
            "api_stage": "draft",
        }

    def _validate_input(self, x: Sequence[float]) -> tuple[float, ...]:
        values = tuple(float(value) for value in x)
        if len(values) != self.config.input_dim:
            msg = f"expected input_dim={self.config.input_dim}, got {len(values)}"
            raise ValueError(msg)
        if not all(math.isfinite(value) for value in values):
            msg = "all input values must be finite"
            raise ValueError(msg)
        return values
