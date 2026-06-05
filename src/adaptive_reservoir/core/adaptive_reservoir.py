"""Public AdaptiveReservoir facade."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

from adaptive_reservoir.core.config import ReservoirConfig
from adaptive_reservoir.core.reservoir import ReservoirCore
from adaptive_reservoir.core.result import AdaptiveChannels, AdaptiveStepResult, StepMetrics
from adaptive_reservoir.features import extract_features


class AdaptiveReservoir:
    """High-level facade for the adaptive-reservoir public API.

    PR3.3 wires built-in feature modes into the public facade. Readouts,
    diagnostics, and channel calculations are added by later milestones.
    """

    def __init__(self, config: ReservoirConfig) -> None:
        self.config = config
        self._core = ReservoirCore.from_config(config)

    @property
    def samples_seen(self) -> int:
        """Number of processed samples."""

        return self._core.state.samples_seen

    def step(self, x: Sequence[float], target: float | None = None) -> AdaptiveStepResult:
        """Process one numeric event vector and return an adaptive step result."""

        start = time.perf_counter()
        if target is not None and not math.isfinite(target):
            msg = "target must be finite when provided"
            raise ValueError(msg)

        state = self._core.step(x)
        elapsed_us = (time.perf_counter() - start) * 1_000_000.0
        features_array = extract_features(state, self.config.feature_mode)
        features = tuple(float(value) for value in features_array)
        return AdaptiveStepResult(
            prediction=None,
            features=features,
            channels=AdaptiveChannels(),
            metrics=StepMetrics(
                samples_seen=state.samples_seen,
                prediction_available=False,
                target_available=target is not None,
                us_per_sample=elapsed_us,
            ),
            state=state,
        )

    def reset(self) -> None:
        """Reset runtime counters and reservoir state.

        Later milestones will also reset traces, readout state, and channel buffers
        when those components become active.
        """

        self._core = ReservoirCore.from_config(self.config)

    def snapshot(self) -> dict[str, object]:
        """Return a serializable snapshot of the current mathematical state."""

        return {
            "samples_seen": self.samples_seen,
            "config": self.config,
            "api_stage": "feature_modes_v1",
        }
