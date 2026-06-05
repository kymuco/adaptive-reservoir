"""Built-in feature extraction modes for reservoir state."""

from __future__ import annotations

import numpy as np

from adaptive_reservoir.core.config import FeatureMode
from adaptive_reservoir.core.protocols import FloatArray
from adaptive_reservoir.core.state import ReservoirState


def extract_features(state: ReservoirState, mode: FeatureMode) -> FloatArray:
    """Extract a read-only feature vector from reservoir state."""

    if mode == "state_raw":
        return _readonly_copy(state.activations)
    if mode == "state_slow_raw":
        return _readonly_copy(np.concatenate((state.activations, state.slow_trace)))
    if mode == "multi_raw":
        return _readonly_copy(
            np.concatenate(
                (
                    state.activations,
                    state.fast_trace,
                    state.mid_trace,
                    state.slow_trace,
                )
            )
        )
    msg = f"unsupported feature mode: {mode!r}"
    raise ValueError(msg)


def _readonly_copy(values: FloatArray) -> FloatArray:
    features = np.asarray(values).astype(values.dtype, copy=True)
    features.setflags(write=False)
    return features
