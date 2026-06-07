"""Public AdaptiveReservoir facade."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence

from adaptive_reservoir.core.config import ReservoirConfig
from adaptive_reservoir.core.reservoir import ReservoirCore
from adaptive_reservoir.core.result import AdaptiveChannels, AdaptiveStepResult, StepMetrics
from adaptive_reservoir.core.snapshot import (
    SNAPSHOT_API_STAGE,
    SNAPSHOT_SCHEMA_VERSION,
    restore_state,
    snapshot_state,
    validate_runtime_snapshot,
)
from adaptive_reservoir.diagnostics import calculate_state_diagnostics, rms_norm
from adaptive_reservoir.features import extract_features


class AdaptiveReservoir:
    """High-level facade for the adaptive-reservoir public API.

    PR3.5 adds reset/snapshot/restore for mathematical runtime state. Readouts,
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
        """Reset mathematical runtime state to deterministic zero-state."""

        self._core = ReservoirCore.from_config(self.config)

    def snapshot(self) -> dict[str, object]:
        """Return a serializable snapshot of the current mathematical state."""

        return {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "api_stage": SNAPSHOT_API_STAGE,
            "state": snapshot_state(self._core.state),
            "readout_state": None,
            "metrics_buffers": {},
        }

    def restore(self, snapshot: Mapping[str, object]) -> None:
        """Restore mathematical runtime state from a compatible snapshot."""

        runtime_snapshot = validate_runtime_snapshot(snapshot)
        self._core.state = restore_state(
            runtime_snapshot["state"],
            expected_n_cells=self.config.n_cells,
            dtype=self.config.dtype,
        )
