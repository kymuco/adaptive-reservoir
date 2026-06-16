"""Public AdaptiveReservoir facade."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import numpy as np

from adaptive_reservoir.channels import AdaptiveChannelCalculator
from adaptive_reservoir.core.config import ReservoirConfig
from adaptive_reservoir.core.reservoir import ReservoirCore
from adaptive_reservoir.core.result import AdaptiveStepResult, StepMetrics
from adaptive_reservoir.core.snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    ReservoirSnapshot,
    clone_reservoir_state,
)
from adaptive_reservoir.core.state import ReservoirState
from adaptive_reservoir.diagnostics import calculate_state_diagnostics, rms_norm
from adaptive_reservoir.features import extract_features
from adaptive_reservoir.readout.base import ReadoutProtocol
from adaptive_reservoir.readout.factory import create_readout


class AdaptiveReservoir:
    """High-level facade for the adaptive-reservoir public API."""

    def __init__(self, config: ReservoirConfig) -> None:
        self.config = config
        self._core = ReservoirCore.from_config(config)
        self._readout = self._create_readout()
        self._channels = self._create_channel_calculator()

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
        features_array = extract_features(state, self.config.feature_mode)
        prediction = self._readout.predict(features_array)
        channels = self._channels.update(
            input=x,
            state=state,
            features=features_array,
            prediction=prediction,
            target=target,
        )
        prediction_error = None
        readout_updated = False
        if target is not None:
            prediction_error = abs(float(target) - prediction)
            self._readout.update(features_array, target)
            readout_updated = True
        diagnostics = calculate_state_diagnostics(
            previous=previous_state,
            current=state,
            saturation_threshold=self.config.channels.saturation_threshold,
        )
        features = tuple(float(value) for value in features_array)
        elapsed_us = (time.perf_counter() - start) * 1_000_000.0
        return AdaptiveStepResult(
            prediction=prediction,
            features=features,
            channels=channels,
            metrics=StepMetrics(
                samples_seen=state.samples_seen,
                prediction_available=True,
                target_available=target is not None,
                readout_updated=readout_updated,
                state_norm=diagnostics.state_norm,
                state_delta=diagnostics.state_delta,
                feature_norm=rms_norm(features_array),
                saturation_rate=channels.saturation,
                trace_norms=diagnostics.trace_norms,
                prediction_error=prediction_error,
                us_per_sample=elapsed_us,
            ),
            state=state,
        )

    def reset(self) -> None:
        """Reset runtime counters, reservoir state, readout state, and channels."""

        self._core = ReservoirCore.from_config(self.config)
        self._readout = self._create_readout()
        self._channels = self._create_channel_calculator()

    def snapshot(self) -> ReservoirSnapshot:
        """Return an immutable checkpoint of the current numeric state."""

        return ReservoirSnapshot(
            state=clone_reservoir_state(self._core.state),
            readout=self._readout.snapshot(),
            channels=self._channels.snapshot(),
            schema_version=SNAPSHOT_SCHEMA_VERSION,
        )

    def restore(self, snapshot: ReservoirSnapshot) -> None:
        """Restore the reservoir, readout, and channel runtime state."""

        if not isinstance(snapshot, ReservoirSnapshot):
            msg = "snapshot must be a ReservoirSnapshot"
            raise TypeError(msg)
        if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION:
            msg = f"unsupported snapshot schema_version: {snapshot.schema_version}"
            raise ValueError(msg)
        state = snapshot.state
        if not isinstance(state, ReservoirState):
            msg = "snapshot.state must be a ReservoirState"
            raise TypeError(msg)
        self._validate_snapshot_state(state)
        new_core = ReservoirCore.from_config(self.config)
        new_core.state = clone_reservoir_state(state)
        new_readout = self._create_readout()
        new_readout.restore(snapshot.readout)
        new_channels = self._create_channel_calculator()
        new_channels.restore(snapshot.channels)
        self._core = new_core
        self._readout = new_readout
        self._channels = new_channels

    def _create_readout(self) -> ReadoutProtocol:
        features = extract_features(self._core.state, self.config.feature_mode)
        return create_readout(
            config=self.config.readout,
            feature_dim=int(features.size),
            dtype=self.config.dtype,
        )

    def _create_channel_calculator(self) -> AdaptiveChannelCalculator:
        return AdaptiveChannelCalculator(
            config=self.config.channels,
            dtype=self.config.dtype,
        )

    def _validate_snapshot_state(self, state: ReservoirState) -> None:
        """Validate snapshot compatibility with the current model configuration."""

        expected_shape = (self.config.n_cells,)
        if state.activations.shape != expected_shape:
            msg = f"snapshot state activations shape must match {expected_shape}"
            raise ValueError(msg)
        if state.fast_trace.shape != expected_shape:
            msg = f"snapshot state fast_trace shape must match {expected_shape}"
            raise ValueError(msg)
        if state.mid_trace.shape != expected_shape:
            msg = f"snapshot state mid_trace shape must match {expected_shape}"
            raise ValueError(msg)
        if state.slow_trace.shape != expected_shape:
            msg = f"snapshot state slow_trace shape must match {expected_shape}"
            raise ValueError(msg)
        if np.dtype(state.activations.dtype) != np.dtype(self.config.dtype):
            msg = "snapshot state dtype must match config.dtype"
            raise ValueError(msg)
        if state.samples_seen < 0:
            msg = "snapshot state samples_seen must be non-negative"
            raise ValueError(msg)
