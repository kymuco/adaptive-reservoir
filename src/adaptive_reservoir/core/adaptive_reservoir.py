"""Public AdaptiveReservoir facade."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

from adaptive_reservoir.core.config import ReservoirConfig
from adaptive_reservoir.core.reservoir import ReservoirCore
from adaptive_reservoir.core.result import AdaptiveChannels, AdaptiveStepResult, StepMetrics
from adaptive_reservoir.diagnostics import calculate_state_diagnostics, rms_norm
from adaptive_reservoir.features import extract_features


class AdaptiveReservoir:
    """High-level facade for the adaptive-reservoir public API.

    PR3.4 adds state diagnostics and wires the saturation channel. Readouts,
    windowed channel calculations, and recurrent plasticity are added by later
    milestones.
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

        previous_state = self._core.state
        state = self._core.step(x)
        elapsed_us = (time.perf_counter() - start) * 1_000_000.0
        features_array = extract_features(state, self.config.feature_mode)
        features = tuple(float(value) for value in features_array)
        diagnostics = calculate_state_diagnostics(
            previous=previous_state,
            current=state,
            saturation_threshold=self.config.channels.saturation_threshold,
        )
        return AdaptiveStepResult(
            prediction=None,
            features=features,
            channels=AdaptiveChannels(saturation=diagnostics.saturation_rate),
            metrics=StepMetrics(
                samples_seen=state.samples_seen,
                prediction_available=False,
                target_available=target is not None,
                state_norm=diagnostics.state_norm,
                state_delta=diagnostics.state_delta,
                feature_norm=rms_norm(features_array),
                saturation_rate=diagnostics.saturation_rate,
                trace_norms=diagnostics.trace_norms,
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
            "api_stage": "state_diagnostics_v1",
        }
