"""Stateful reservoir core update logic."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from adaptive_reservoir.core.config import ReservoirConfig
from adaptive_reservoir.core.protocols import FloatArray, TopologyBuilderProtocol
from adaptive_reservoir.core.state import ReservoirState
from adaptive_reservoir.topology import (
    EdgeList,
    ModularSmallWorldTopologyBuilder,
    RandomSparseTopologyBuilder,
    RingShortcutsTopologyBuilder,
)

_INPUT_PROJECTION_SEED_LABEL = "core.input_projection"


@dataclass(slots=True)
class ReservoirCore:
    """Minimal stateful reservoir substrate.

    PR3.1 intentionally implements only the core step function. Trace updates,
    feature modes, diagnostics, readouts, and recurrent plasticity are added by
    later PRs.
    """

    config: ReservoirConfig
    recurrent_edges: EdgeList
    input_weights: FloatArray
    state: ReservoirState

    @classmethod
    def from_config(cls, config: ReservoirConfig) -> ReservoirCore:
        """Create a deterministic reservoir core from ``config``."""

        topology_builder = _default_topology_builder(config)
        recurrent_edges = topology_builder.build(config)
        input_weights = _build_input_projection(config)
        state = ReservoirState.zeros(n_cells=config.n_cells, dtype=config.dtype)
        return cls(
            config=config,
            recurrent_edges=recurrent_edges,
            input_weights=input_weights,
            state=state,
        )

    def __post_init__(self) -> None:
        dtype = np.dtype(self.config.dtype)
        if self.recurrent_edges.n_nodes != self.config.n_cells:
            msg = "recurrent_edges.n_nodes must match config.n_cells"
            raise ValueError(msg)
        input_weights = np.asarray(self.input_weights, dtype=dtype)
        expected_shape = (self.config.n_cells, self.config.input_dim)
        if input_weights.shape != expected_shape:
            msg = f"input_weights must have shape {expected_shape}"
            raise ValueError(msg)
        if not np.all(np.isfinite(input_weights)):
            msg = "input_weights must contain only finite values"
            raise ValueError(msg)
        if self.state.activations.shape != (self.config.n_cells,):
            msg = "state activations must match config.n_cells"
            raise ValueError(msg)
        object.__setattr__(self, "input_weights", input_weights)

    def step(self, x: Sequence[float]) -> ReservoirState:
        """Advance reservoir state by one input vector."""

        input_vector = _validate_input_vector(x, input_dim=self.config.input_dim)
        input_vector = input_vector.astype(self.input_weights.dtype, copy=False)
        previous = self.state.activations
        input_drive = self.input_weights @ input_vector
        recurrent_drive = _sparse_recurrent_drive(
            edges=self.recurrent_edges,
            state=previous,
            dtype=self.input_weights.dtype,
        )
        fatigue_drive = self.config.fatigue_rate * previous
        pre_activation = (
            input_drive
            + self.config.recurrent_scale * recurrent_drive
            - fatigue_drive
        )
        candidate = np.tanh(pre_activation)
        new_activations = (
            (1.0 - self.config.leak_rate) * previous
            + self.config.leak_rate * candidate
        ).astype(self.input_weights.dtype, copy=False)

        self.state = ReservoirState(
            activations=new_activations,
            fast_trace=self.state.fast_trace,
            mid_trace=self.state.mid_trace,
            slow_trace=self.state.slow_trace,
            samples_seen=self.state.samples_seen + 1,
        )
        return self.state


def _default_topology_builder(config: ReservoirConfig) -> TopologyBuilderProtocol:
    if config.topology == "random_sparse":
        return RandomSparseTopologyBuilder()
    if config.topology == "ring_shortcuts":
        return RingShortcutsTopologyBuilder()
    if config.topology == "modular_small_world":
        return ModularSmallWorldTopologyBuilder()
    msg = f"unsupported topology: {config.topology!r}"
    raise ValueError(msg)


def _build_input_projection(config: ReservoirConfig) -> FloatArray:
    dtype = np.dtype(config.dtype)
    rng = np.random.default_rng(_derive_seed(config.seed, _INPUT_PROJECTION_SEED_LABEL))
    scale = config.input_scale / math.sqrt(config.input_dim)
    return rng.uniform(
        -scale,
        scale,
        size=(config.n_cells, config.input_dim),
    ).astype(dtype)


def _sparse_recurrent_drive(
    *,
    edges: EdgeList,
    state: FloatArray,
    dtype: np.dtype[np.floating],
) -> FloatArray:
    drive = np.zeros(edges.n_nodes, dtype=dtype)
    contributions = edges.weights.astype(dtype, copy=False) * state[edges.sources]
    np.add.at(drive, edges.targets, contributions)
    return drive


def _validate_input_vector(x: Sequence[float], *, input_dim: int) -> FloatArray:
    values = np.asarray(tuple(float(value) for value in x), dtype=np.float64)
    if values.shape != (input_dim,):
        msg = f"expected input_dim={input_dim}, got {values.size}"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "all input values must be finite"
        raise ValueError(msg)
    return values


def _derive_seed(seed: int, label: str) -> int:
    payload = f"{seed}:{label}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)
