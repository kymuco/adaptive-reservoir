"""Factory for stable scalar readout implementations."""

from __future__ import annotations

from adaptive_reservoir.core.config import ReadoutConfig
from adaptive_reservoir.readout.base import ReadoutProtocol
from adaptive_reservoir.readout.nlms import NLMSReadout
from adaptive_reservoir.readout.replay_ridge import ReplayRidgeReadout
from adaptive_reservoir.readout.sliding_ridge import SlidingWindowRidgeReadout


def create_readout(
    *,
    config: ReadoutConfig,
    feature_dim: int,
    dtype: str,
) -> ReadoutProtocol:
    """Create a stable scalar readout from public readout config."""

    if config.name == "nlms":
        return NLMSReadout(
            feature_dim=feature_dim,
            learning_rate=config.learning_rate,
            dtype=dtype,
        )
    if config.name == "replay_ridge":
        return ReplayRidgeReadout(
            feature_dim=feature_dim,
            buffer_size=config.buffer_size,
            refit_interval=config.update_interval,
            alpha=config.ridge_alpha,
            dtype=dtype,
        )
    if config.name == "sliding_ridge":
        return SlidingWindowRidgeReadout(
            feature_dim=feature_dim,
            window_size=config.window_size,
            update_interval=config.update_interval,
            alpha=config.ridge_alpha,
            dtype=dtype,
        )

    msg = f"unsupported stable readout: {config.name!r}"
    raise ValueError(msg)
