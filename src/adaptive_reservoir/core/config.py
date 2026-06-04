"""Configuration objects for the public adaptive-reservoir API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TopologyName = Literal["random_sparse", "ring_shortcuts", "modular_small_world"]
FeatureMode = Literal["state_raw", "state_slow_raw", "multi_raw"]
ReadoutName = Literal["nlms", "replay_ridge", "sliding_ridge"]


@dataclass(frozen=True, slots=True)
class ReservoirConfig:
    """Configuration for :class:`adaptive_reservoir.AdaptiveReservoir`.

    This is intentionally small in PR0.3. Later milestones will split topology,
    trace, channel, and readout settings into dedicated config objects.
    """

    input_dim: int
    n_cells: int = 64
    topology: TopologyName = "modular_small_world"
    feature_mode: FeatureMode = "state_slow_raw"
    readout: ReadoutName = "sliding_ridge"
    seed: int = 0
    trace_decays: tuple[float, float, float] = field(default=(0.5, 0.9, 0.99))

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            msg = "input_dim must be positive"
            raise ValueError(msg)
        if self.n_cells <= 0:
            msg = "n_cells must be positive"
            raise ValueError(msg)
        if len(self.trace_decays) != 3:
            msg = "trace_decays must contain exactly three values: fast, mid, slow"
            raise ValueError(msg)
        if any(decay < 0.0 or decay >= 1.0 for decay in self.trace_decays):
            msg = "trace decays must be in the range [0.0, 1.0)"
            raise ValueError(msg)
