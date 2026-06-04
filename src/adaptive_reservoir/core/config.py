"""Configuration objects for the public adaptive-reservoir API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TopologyName = Literal["random_sparse", "ring_shortcuts", "modular_small_world"]
FeatureMode = Literal["state_raw", "state_slow_raw", "multi_raw"]
ReadoutName = Literal["nlms", "replay_ridge", "sliding_ridge"]
DTypeName = Literal["float32", "float64"]

TOPOLOGY_NAMES = frozenset(("random_sparse", "ring_shortcuts", "modular_small_world"))
FEATURE_MODES = frozenset(("state_raw", "state_slow_raw", "multi_raw"))
READOUT_NAMES = frozenset(("nlms", "replay_ridge", "sliding_ridge"))
DTYPE_NAMES = frozenset(("float32", "float64"))


@dataclass(frozen=True, slots=True)
class TraceConfig:
    """Configuration for multi-timescale trace features."""

    fast_decay: float = 0.5
    mid_decay: float = 0.9
    slow_decay: float = 0.99

    @property
    def decays(self) -> tuple[float, float, float]:
        """Return trace decays ordered as fast, mid, slow."""

        return (self.fast_decay, self.mid_decay, self.slow_decay)

    def __post_init__(self) -> None:
        _validate_decay("fast_decay", self.fast_decay)
        _validate_decay("mid_decay", self.mid_decay)
        _validate_decay("slow_decay", self.slow_decay)


@dataclass(frozen=True, slots=True)
class ReadoutConfig:
    """Configuration for online readout behavior."""

    name: ReadoutName = "sliding_ridge"
    learning_rate: float = 0.05
    ridge_alpha: float = 1e-3
    buffer_size: int = 160
    window_size: int = 128
    update_interval: int = 1

    def __post_init__(self) -> None:
        _validate_choice("readout.name", self.name, READOUT_NAMES)
        if self.learning_rate <= 0.0:
            msg = "learning_rate must be positive"
            raise ValueError(msg)
        if self.ridge_alpha < 0.0:
            msg = "ridge_alpha must be non-negative"
            raise ValueError(msg)
        _validate_positive_int("buffer_size", self.buffer_size)
        _validate_positive_int("window_size", self.window_size)
        _validate_positive_int("update_interval", self.update_interval)


@dataclass(frozen=True, slots=True)
class ChannelConfig:
    """Configuration for adaptive state channel calculations."""

    novelty_window: int = 32
    stability_window: int = 32
    drift_window: int = 64
    saturation_threshold: float = 0.95
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        _validate_positive_int("novelty_window", self.novelty_window)
        _validate_positive_int("stability_window", self.stability_window)
        _validate_positive_int("drift_window", self.drift_window)
        if self.saturation_threshold <= 0.0 or self.saturation_threshold > 1.0:
            msg = "saturation_threshold must be in the range (0.0, 1.0]"
            raise ValueError(msg)
        if self.epsilon <= 0.0:
            msg = "epsilon must be positive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ReservoirConfig:
    """Root configuration for :class:`adaptive_reservoir.AdaptiveReservoir`."""

    input_dim: int
    n_cells: int = 64
    topology: TopologyName = "modular_small_world"
    feature_mode: FeatureMode = "state_slow_raw"
    seed: int = 0
    dtype: DTypeName = "float64"
    trace: TraceConfig = field(default_factory=TraceConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    channels: ChannelConfig = field(default_factory=ChannelConfig)

    @property
    def trace_decays(self) -> tuple[float, float, float]:
        """Return trace decays ordered as fast, mid, slow."""

        return self.trace.decays

    def __post_init__(self) -> None:
        _validate_positive_int("input_dim", self.input_dim)
        _validate_positive_int("n_cells", self.n_cells)
        _validate_choice("topology", self.topology, TOPOLOGY_NAMES)
        _validate_choice("feature_mode", self.feature_mode, FEATURE_MODES)
        _validate_choice("dtype", self.dtype, DTYPE_NAMES)


def _validate_choice(name: str, value: str, allowed: frozenset[str]) -> None:
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        msg = f"{name} must be one of: {allowed_values}; got {value!r}"
        raise ValueError(msg)


def _validate_decay(name: str, value: float) -> None:
    if value < 0.0 or value >= 1.0:
        msg = f"{name} must be in the range [0.0, 1.0)"
        raise ValueError(msg)


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        msg = f"{name} must be positive"
        raise ValueError(msg)
