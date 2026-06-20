"""Stateful reservoir core update logic."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from adaptive_reservoir.core.config import ReservoirConfig
from adaptive_reservoir.core.protocols import FloatArray, TopologyBuilderProtocol
from adaptive_reservoir.core.state import ReservoirState
from adaptive_reservoir.core.validation import validate_input_vector
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

    PR3.2 updates multi-timescale traces from the post-leaky reservoir state.
    Feature modes, diagnostics, readouts, and recurrent plasticity are added by
    later PRs.
    """

    config: ReservoirConfig
    recurrent_edges: EdgeList
    input_weights: FloatArray
    state: ReservoirState
    _work_input_drive: FloatArray = field(init=False, repr=False)
    _work_recurrent_drive: FloatArray = field(init=False, repr=False)
    _work_pre_activation: FloatArray = field(init=False, repr=False)
    _work_candidate: FloatArray = field(init=False, repr=False)
    _work_new_activations: FloatArray = field(init=False, repr=False)
    _work_fast_trace: FloatArray = field(init=False, repr=False)
    _work_mid_trace: FloatArray = field(init=False, repr=False)
    _work_slow_trace: FloatArray = field(init=False, repr=False)
    _work_edge_sources: FloatArray = field(init=False, repr=False)
    _work_edge_contributions: FloatArray = field(init=False, repr=False)

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
        self._initialize_work_buffers(dtype)

    def step(self, x: Sequence[float]) -> ReservoirState:
        """Advance reservoir state by one input vector."""

        input_vector = validate_input_vector(
            x,
            input_dim=self.config.input_dim,
            dtype=self.config.dtype,
        )
        input_vector = input_vector.astype(self.input_weights.dtype, copy=False)
        previous_state = self.state
        previous = previous_state.activations

        np.matmul(self.input_weights, input_vector, out=self._work_input_drive)
        _sparse_recurrent_drive_into(
            edges=self.recurrent_edges,
            state=previous,
            dtype=self.input_weights.dtype,
            out=self._work_recurrent_drive,
            source_values=self._work_edge_sources,
            contributions=self._work_edge_contributions,
        )

        np.copyto(self._work_pre_activation, self._work_input_drive)
        np.multiply(
            self._work_recurrent_drive,
            self.config.recurrent_scale,
            out=self._work_candidate,
        )
        np.add(
            self._work_pre_activation,
            self._work_candidate,
            out=self._work_pre_activation,
        )
        np.multiply(
            previous,
            self.config.fatigue_rate,
            out=self._work_candidate,
        )
        np.subtract(
            self._work_pre_activation,
            self._work_candidate,
            out=self._work_pre_activation,
        )
        np.tanh(self._work_pre_activation, out=self._work_candidate)

        np.multiply(
            previous,
            1.0 - self.config.leak_rate,
            out=self._work_new_activations,
        )
        np.multiply(
            self._work_candidate,
            self.config.leak_rate,
            out=self._work_candidate,
        )
        np.add(
            self._work_new_activations,
            self._work_candidate,
            out=self._work_new_activations,
        )

        trace_config = self.config.trace
        _update_trace_into(
            out=self._work_fast_trace,
            old_trace=previous_state.fast_trace,
            state=self._work_new_activations,
            decay=trace_config.fast_decay,
            scratch=self._work_candidate,
        )
        _update_trace_into(
            out=self._work_mid_trace,
            old_trace=previous_state.mid_trace,
            state=self._work_new_activations,
            decay=trace_config.mid_decay,
            scratch=self._work_candidate,
        )
        _update_trace_into(
            out=self._work_slow_trace,
            old_trace=previous_state.slow_trace,
            state=self._work_new_activations,
            decay=trace_config.slow_decay,
            scratch=self._work_candidate,
        )

        self.state = ReservoirState(
            activations=self._work_new_activations,
            fast_trace=self._work_fast_trace,
            mid_trace=self._work_mid_trace,
            slow_trace=self._work_slow_trace,
            samples_seen=previous_state.samples_seen + 1,
        )
        return self.state

    def _initialize_work_buffers(self, dtype: np.dtype[np.floating]) -> None:
        n_cells = self.config.n_cells
        n_edges = self.recurrent_edges.n_edges
        self._work_input_drive = np.empty(n_cells, dtype=dtype)
        self._work_recurrent_drive = np.empty(n_cells, dtype=dtype)
        self._work_pre_activation = np.empty(n_cells, dtype=dtype)
        self._work_candidate = np.empty(n_cells, dtype=dtype)
        self._work_new_activations = np.empty(n_cells, dtype=dtype)
        self._work_fast_trace = np.empty(n_cells, dtype=dtype)
        self._work_mid_trace = np.empty(n_cells, dtype=dtype)
        self._work_slow_trace = np.empty(n_cells, dtype=dtype)
        self._work_edge_sources = np.empty(n_edges, dtype=dtype)
        self._work_edge_contributions = np.empty(n_edges, dtype=dtype)


def _default_topology_builder(config: ReservoirConfig) -> TopologyBuilderProtocol:
    if config.topology == "random_sparse":
        return _default_random_sparse_builder(config.n_cells)
    if config.topology == "ring_shortcuts":
        return _default_ring_shortcuts_builder(config.n_cells)
    if config.topology == "modular_small_world":
        return _default_modular_small_world_builder(config.n_cells)
    msg = f"unsupported topology: {config.topology!r}"
    raise ValueError(msg)


def _default_random_sparse_builder(n_cells: int) -> RandomSparseTopologyBuilder:
    if n_cells == 1:
        return RandomSparseTopologyBuilder(in_degree=1, allow_self_loops=True)
    return RandomSparseTopologyBuilder(in_degree=min(8, n_cells - 1))


def _default_ring_shortcuts_builder(n_cells: int) -> RingShortcutsTopologyBuilder:
    if n_cells < 2:
        msg = "ring_shortcuts topology requires n_cells >= 2"
        raise ValueError(msg)
    if n_cells == 2:
        return RingShortcutsTopologyBuilder(shortcuts_per_node=0, bidirectional=False)
    return RingShortcutsTopologyBuilder(shortcuts_per_node=min(1, n_cells - 3))


def _default_modular_small_world_builder(n_cells: int) -> ModularSmallWorldTopologyBuilder:
    if n_cells < 4:
        msg = "modular_small_world topology requires n_cells >= 4"
        raise ValueError(msg)
    n_modules = min(4, max(2, n_cells // 2))
    min_module_size = n_cells // n_modules
    intra_module_degree = min(3, min_module_size - 1)
    inter_module_shortcuts = min(1, n_cells - min_module_size)
    return ModularSmallWorldTopologyBuilder(
        n_modules=n_modules,
        intra_module_degree=intra_module_degree,
        inter_module_shortcuts=inter_module_shortcuts,
    )


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
    source_values = np.empty(edges.n_edges, dtype=dtype)
    contributions = np.empty(edges.n_edges, dtype=dtype)
    return _sparse_recurrent_drive_into(
        edges=edges,
        state=state,
        dtype=dtype,
        out=drive,
        source_values=source_values,
        contributions=contributions,
    )


def _sparse_recurrent_drive_into(
    *,
    edges: EdgeList,
    state: FloatArray,
    dtype: np.dtype[np.floating],
    out: FloatArray,
    source_values: FloatArray,
    contributions: FloatArray,
) -> FloatArray:
    out.fill(0.0)
    np.take(state, edges.sources, out=source_values)
    np.multiply(edges.weights.astype(dtype, copy=False), source_values, out=contributions)
    np.add.at(out, edges.targets, contributions)
    return out


def _update_trace(old_trace: FloatArray, state: FloatArray, decay: float) -> FloatArray:
    updated = np.empty_like(state)
    scratch = np.empty_like(state)
    return _update_trace_into(
        out=updated,
        old_trace=old_trace,
        state=state,
        decay=decay,
        scratch=scratch,
    )


def _update_trace_into(
    *,
    out: FloatArray,
    old_trace: FloatArray,
    state: FloatArray,
    decay: float,
    scratch: FloatArray,
) -> FloatArray:
    np.multiply(old_trace, decay, out=out)
    np.multiply(state, 1.0 - decay, out=scratch)
    np.add(out, scratch, out=out)
    return out


def _derive_seed(seed: int, label: str) -> int:
    payload = f"{seed}:{label}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)
