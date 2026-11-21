"""Causal, prequential baselines for TDNR-01.

Each step consumes only the current two-component observation and its
training target. The prediction is computed before the target is used.
No clean targets, latent signals, drift labels, or future data are needed.
Instantiate a fresh baseline for each independent experiment run.
"""

from __future__ import annotations

import math
from collections import deque
from numbers import Real

import numpy as np

from adaptive_reservoir.readout.base import validate_features, validate_target
from adaptive_reservoir.readout.sliding_ridge import SlidingWindowRidgeReadout


class ZeroBaseline:
    """Predict zero at every time step, independently of training data."""

    name = "zero"

    def step(self, sample: object, target: object) -> float:
        """Validate the current pair and return the zero prediction."""
        _validate_sample(sample)
        _validate_training_target(target)
        return 0.0


class PersistenceBaseline:
    """Predict the previous observed training target, initially zero.

    At time t, the prediction is y_train[t-1]. The current target is
    admitted only after that prediction has been computed.
    """

    name = "persistence"

    def __init__(self) -> None:
        self._previous_target: float | None = None

    def step(self, sample: object, target: object) -> float:
        """Predict from the previous target, then remember the current one."""
        _validate_sample(sample)
        current_target = _validate_training_target(target)
        prediction = 0.0 if self._previous_target is None else self._previous_target
        self._previous_target = current_target
        return prediction


class LagBufferRidgeBaseline:
    """Online sliding-window ridge over current and past observations.

    Default features are [x_t, x_(t-1), ..., x_(t-16)], where x is the
    first component of the TDNR-01 input. The second input component is
    validated but ignored because it is a derived first difference.

    Before enough observations exist, missing historical lags are zero.
    No generator warmup history is accessed. A current observation enters
    the lag history only after its pre-update prediction is produced.

    The existing SlidingWindowRidgeReadout fits an unpenalized intercept
    and ridge-regularized feature coefficients. It refits after every
    supervised update using at most window_size recent training pairs.
    """

    name = "lag_buffer_ridge"

    def __init__(
        self,
        *,
        max_lag: int = 16,
        window_size: int = 128,
        alpha: float = 0.001,
    ) -> None:
        self.max_lag = _nonnegative_int("max_lag", max_lag)
        self.window_size = _positive_int("window_size", window_size)
        self.alpha = _positive_float("alpha", alpha)
        self._history: deque[float] = deque(maxlen=self.max_lag)
        self._readout = SlidingWindowRidgeReadout(
            feature_dim=self.max_lag + 1,
            window_size=self.window_size,
            alpha=self.alpha,
            update_interval=1,
            dtype="float64",
        )

    @property
    def samples_seen(self) -> int:
        """Number of completed supervised steps."""
        return self._readout.samples_seen

    @property
    def window_count(self) -> int:
        """Number of retained supervised samples, bounded by window_size."""
        return self._readout.window_count

    def step(self, sample: object, target: object) -> float:
        """Predict with past fit, then train and advance observation history.

        Validation happens before state changes. The target never enters
        the feature vector and cannot affect its own prediction.
        """
        current = _validate_sample(sample)
        current_target = _validate_training_target(target)

        features = np.zeros(self.max_lag + 1, dtype=np.float64)
        features[0] = current
        for lag, previous in enumerate(reversed(self._history), start=1):
            features[lag] = previous

        prediction = self._readout.predict(features)
        self._readout.update(features, current_target)
        self._history.append(current)
        return prediction


def _validate_sample(sample: object) -> float:
    """Validate a complete TDNR-01 input and return its current signal."""
    vector = validate_features(sample, expected_dim=2, dtype="float64")
    return float(vector[0])


def _validate_training_target(target: object) -> float:
    if isinstance(target, bool):
        raise ValueError("target must be a finite numeric scalar, not bool")
    return validate_target(target)


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


__all__ = [
    "ZeroBaseline",
    "PersistenceBaseline",
    "LagBufferRidgeBaseline",
]
