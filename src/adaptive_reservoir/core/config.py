"""Configuration objects for the public adaptive-reservoir API."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

from adaptive_reservoir.core.serialization import (
    optional_float,
    optional_int,
    optional_mapping,
    optional_str,
    require_int,
    require_mapping,
)

TopologyName = Literal["random_sparse", "ring_shortcuts", "modular_small_world"]
FeatureMode = Literal["state_raw", "state_slow_raw", "multi_raw"]
ReadoutName = Literal["nlms", "replay_ridge", "sliding_ridge"]
DTypeName = Literal["float32", "float64"]

TOPOLOGY_NAMES = frozenset(("random_sparse", "ring_shortcuts", "modular_small_world"))
FEATURE_MODES = frozenset(("state_raw", "state_slow_raw", "multi_raw"))
READOUT_NAMES = frozenset(("nlms", "replay_ridge", "sliding_ridge"))
DTYPE_NAMES = frozenset(("float32", "float64"))
RESERVOIR_CONFIG_DICT_SCHEMA_VERSION = 1


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

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly trace config dictionary."""

        return {
            "fast_decay": self.fast_decay,
            "mid_decay": self.mid_decay,
            "slow_decay": self.slow_decay,
        }

    @classmethod
    def from_dict(cls, data: object) -> TraceConfig:
        """Create a trace config from a mapping."""

        mapping = require_mapping(data, "trace")
        defaults = cls()
        return cls(
            fast_decay=optional_float(mapping, "fast_decay", defaults.fast_decay),
            mid_decay=optional_float(mapping, "mid_decay", defaults.mid_decay),
            slow_decay=optional_float(mapping, "slow_decay", defaults.slow_decay),
        )

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

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly readout config dictionary."""

        return {
            "name": self.name,
            "learning_rate": self.learning_rate,
            "ridge_alpha": self.ridge_alpha,
            "buffer_size": self.buffer_size,
            "window_size": self.window_size,
            "update_interval": self.update_interval,
        }

    @classmethod
    def from_dict(cls, data: object) -> ReadoutConfig:
        """Create a readout config from a mapping."""

        mapping = require_mapping(data, "readout")
        defaults = cls()
        return cls(
            name=cast(
                ReadoutName,
                optional_str(mapping, "name", defaults.name),
            ),
            learning_rate=optional_float(
                mapping,
                "learning_rate",
                defaults.learning_rate,
            ),
            ridge_alpha=optional_float(mapping, "ridge_alpha", defaults.ridge_alpha),
            buffer_size=optional_int(mapping, "buffer_size", defaults.buffer_size),
            window_size=optional_int(mapping, "window_size", defaults.window_size),
            update_interval=optional_int(
                mapping,
                "update_interval",
                defaults.update_interval,
            ),
        )

    def __post_init__(self) -> None:
        _validate_choice("readout.name", self.name, READOUT_NAMES)
        if self.learning_rate <= 0.0:
            msg = "learning_rate must be positive"
            raise ValueError(msg)
        if self.ridge_alpha <= 0.0:
            msg = "ridge_alpha must be positive"
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

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly channel config dictionary."""

        return {
            "novelty_window": self.novelty_window,
            "stability_window": self.stability_window,
            "drift_window": self.drift_window,
            "saturation_threshold": self.saturation_threshold,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, data: object) -> ChannelConfig:
        """Create a channel config from a mapping."""

        mapping = require_mapping(data, "channels")
        defaults = cls()
        return cls(
            novelty_window=optional_int(
                mapping,
                "novelty_window",
                defaults.novelty_window,
            ),
            stability_window=optional_int(
                mapping,
                "stability_window",
                defaults.stability_window,
            ),
            drift_window=optional_int(mapping, "drift_window", defaults.drift_window),
            saturation_threshold=optional_float(
                mapping,
                "saturation_threshold",
                defaults.saturation_threshold,
            ),
            epsilon=optional_float(mapping, "epsilon", defaults.epsilon),
        )

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
    leak_rate: float = 0.3
    input_scale: float = 1.0
    recurrent_scale: float = 1.0
    fatigue_rate: float = 0.0
    trace: TraceConfig = field(default_factory=TraceConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    channels: ChannelConfig = field(default_factory=ChannelConfig)

    @property
    def trace_decays(self) -> tuple[float, float, float]:
        """Return trace decays ordered as fast, mid, slow."""

        return self.trace.decays

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly reservoir config dictionary."""

        return {
            "schema_version": RESERVOIR_CONFIG_DICT_SCHEMA_VERSION,
            "input_dim": self.input_dim,
            "n_cells": self.n_cells,
            "topology": self.topology,
            "feature_mode": self.feature_mode,
            "seed": self.seed,
            "dtype": self.dtype,
            "leak_rate": self.leak_rate,
            "input_scale": self.input_scale,
            "recurrent_scale": self.recurrent_scale,
            "fatigue_rate": self.fatigue_rate,
            "trace": self.trace.to_dict(),
            "readout": self.readout.to_dict(),
            "channels": self.channels.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: object) -> ReservoirConfig:
        """Create a reservoir config from a JSON-friendly mapping."""

        mapping = require_mapping(data, "config")
        schema_version = optional_int(
            mapping,
            "schema_version",
            RESERVOIR_CONFIG_DICT_SCHEMA_VERSION,
        )
        if schema_version != RESERVOIR_CONFIG_DICT_SCHEMA_VERSION:
            msg = f"unsupported config schema_version: {schema_version}"
            raise ValueError(msg)
        defaults = cls(input_dim=1)
        trace_mapping = optional_mapping(mapping, "trace")
        readout_mapping = optional_mapping(mapping, "readout")
        channels_mapping = optional_mapping(mapping, "channels")
        return cls(
            input_dim=require_int(mapping, "input_dim"),
            n_cells=optional_int(mapping, "n_cells", defaults.n_cells),
            topology=cast(
                TopologyName,
                optional_str(mapping, "topology", defaults.topology),
            ),
            feature_mode=cast(
                FeatureMode,
                optional_str(mapping, "feature_mode", defaults.feature_mode),
            ),
            seed=optional_int(mapping, "seed", defaults.seed),
            dtype=cast(DTypeName, optional_str(mapping, "dtype", defaults.dtype)),
            leak_rate=optional_float(mapping, "leak_rate", defaults.leak_rate),
            input_scale=optional_float(mapping, "input_scale", defaults.input_scale),
            recurrent_scale=optional_float(
                mapping,
                "recurrent_scale",
                defaults.recurrent_scale,
            ),
            fatigue_rate=optional_float(
                mapping,
                "fatigue_rate",
                defaults.fatigue_rate,
            ),
            trace=TraceConfig.from_dict(trace_mapping or defaults.trace.to_dict()),
            readout=ReadoutConfig.from_dict(
                readout_mapping or defaults.readout.to_dict(),
            ),
            channels=ChannelConfig.from_dict(
                channels_mapping or defaults.channels.to_dict(),
            ),
        )

    def __post_init__(self) -> None:
        _validate_positive_int("input_dim", self.input_dim)
        _validate_positive_int("n_cells", self.n_cells)
        _validate_choice("topology", self.topology, TOPOLOGY_NAMES)
        _validate_choice("feature_mode", self.feature_mode, FEATURE_MODES)
        _validate_choice("dtype", self.dtype, DTYPE_NAMES)
        _validate_unit_interval_open_closed("leak_rate", self.leak_rate)
        _validate_positive_float("input_scale", self.input_scale)
        _validate_non_negative_float("recurrent_scale", self.recurrent_scale)
        _validate_non_negative_float("fatigue_rate", self.fatigue_rate)


JsonConfigDict = dict[str, object]


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


def _validate_unit_interval_open_closed(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        msg = f"{name} must be in the range (0.0, 1.0]"
        raise ValueError(msg)


def _validate_positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        msg = f"{name} must be finite and positive"
        raise ValueError(msg)


def _validate_non_negative_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        msg = f"{name} must be finite and non-negative"
        raise ValueError(msg)
