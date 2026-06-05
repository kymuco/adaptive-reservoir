"""Modular small-world recurrent topology builder."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from adaptive_reservoir.core.config import ReservoirConfig
from adaptive_reservoir.core.protocols import FloatArray

_MODULAR_SMALL_WORLD_SEED_LABEL = "topology.modular_small_world"
_INTRA_MODULE_WEIGHT = 1.0


@dataclass(frozen=True, slots=True)
class ModularSmallWorldTopologyBuilder:
    """Build dense matrices with local modules and cross-module shortcuts.

    Matrix convention: ``weights[target_cell, source_cell]``.
    """

    n_modules: int = 4
    intra_module_degree: int = 3
    inter_module_shortcuts: int = 1
    rewire_prob: float = 0.05

    def __post_init__(self) -> None:
        if self.n_modules < 2:
            msg = "n_modules must be >= 2"
            raise ValueError(msg)
        if self.intra_module_degree <= 0:
            msg = "intra_module_degree must be positive"
            raise ValueError(msg)
        if self.inter_module_shortcuts < 0:
            msg = "inter_module_shortcuts must be non-negative"
            raise ValueError(msg)
        if not math.isfinite(self.rewire_prob) or not 0.0 <= self.rewire_prob <= 1.0:
            msg = "rewire_prob must be finite and in the range [0.0, 1.0]"
            raise ValueError(msg)

    def build(self, config: ReservoirConfig) -> FloatArray:
        """Build a recurrent modular small-world weight matrix for ``config``."""

        if config.topology != "modular_small_world":
            msg = "config.topology must be 'modular_small_world'"
            raise ValueError(msg)

        modules = _module_indices(config.n_cells, self.n_modules)
        self._validate_modules(config.n_cells, modules)

        dtype = np.dtype(config.dtype)
        weights = np.zeros((config.n_cells, config.n_cells), dtype=dtype)
        rng = np.random.default_rng(
            _derive_seed(config.seed, _MODULAR_SMALL_WORLD_SEED_LABEL)
        )
        node_to_module = _node_to_module(modules)

        for module_id, module_nodes in enumerate(modules):
            for local_index, target in enumerate(module_nodes):
                target_index = int(target)
                connected_sources: set[int] = set()
                self._add_intra_module_edges(
                    weights=weights,
                    rng=rng,
                    dtype=dtype,
                    target=target_index,
                    local_index=local_index,
                    module_nodes=module_nodes,
                    module_id=module_id,
                    node_to_module=node_to_module,
                    connected_sources=connected_sources,
                )
                self._add_inter_module_shortcuts(
                    weights=weights,
                    rng=rng,
                    dtype=dtype,
                    target=target_index,
                    module_id=module_id,
                    node_to_module=node_to_module,
                    connected_sources=connected_sources,
                )

        return weights

    def _validate_modules(
        self,
        n_cells: int,
        modules: tuple[np.ndarray, ...],
    ) -> None:
        if self.n_modules > n_cells:
            msg = "n_modules must be <= n_cells"
            raise ValueError(msg)

        module_sizes = [len(module) for module in modules]
        min_module_size = min(module_sizes)
        if self.intra_module_degree > min_module_size - 1:
            msg = f"intra_module_degree must be <= {min_module_size - 1}"
            raise ValueError(msg)

        min_outside_count = n_cells - max(module_sizes)
        if self.inter_module_shortcuts > min_outside_count:
            msg = f"inter_module_shortcuts must be <= {min_outside_count}"
            raise ValueError(msg)

        required_cross_edges = self.inter_module_shortcuts
        if self.rewire_prob > 0.0:
            required_cross_edges += self.intra_module_degree
        if required_cross_edges > min_outside_count:
            msg = (
                "intra_module_degree + inter_module_shortcuts must be "
                f"<= {min_outside_count} when rewire_prob > 0"
            )
            raise ValueError(msg)

    def _add_intra_module_edges(
        self,
        *,
        weights: FloatArray,
        rng: np.random.Generator,
        dtype: np.dtype[np.floating],
        target: int,
        local_index: int,
        module_nodes: np.ndarray,
        module_id: int,
        node_to_module: tuple[int, ...],
        connected_sources: set[int],
    ) -> None:
        for offset in range(1, self.intra_module_degree + 1):
            source = int(module_nodes[(local_index - offset) % len(module_nodes)])
            edge_weight = np.array(_INTRA_MODULE_WEIGHT, dtype=dtype)

            if self.rewire_prob > 0.0 and rng.random() < self.rewire_prob:
                candidates = _cross_module_candidates(
                    node_to_module=node_to_module,
                    module_id=module_id,
                    connected_sources=connected_sources,
                )
                if candidates.size > 0:
                    source = int(rng.choice(candidates))
                    edge_weight = _sample_non_zero_weights(
                        rng=rng,
                        size=1,
                        dtype=dtype,
                    )[0]

            connected_sources.add(source)
            weights[target, source] = edge_weight

    def _add_inter_module_shortcuts(
        self,
        *,
        weights: FloatArray,
        rng: np.random.Generator,
        dtype: np.dtype[np.floating],
        target: int,
        module_id: int,
        node_to_module: tuple[int, ...],
        connected_sources: set[int],
    ) -> None:
        if self.inter_module_shortcuts == 0:
            return

        shortcut_sources = rng.choice(
            _cross_module_candidates(
                node_to_module=node_to_module,
                module_id=module_id,
                connected_sources=connected_sources,
            ),
            size=self.inter_module_shortcuts,
            replace=False,
        )
        weights[target, shortcut_sources] = _sample_non_zero_weights(
            rng=rng,
            size=self.inter_module_shortcuts,
            dtype=dtype,
        )


def _module_indices(n_cells: int, n_modules: int) -> tuple[np.ndarray, ...]:
    base_size = n_cells // n_modules
    remainder = n_cells % n_modules
    modules: list[np.ndarray] = []
    start = 0

    for module_id in range(n_modules):
        size = base_size + (1 if module_id < remainder else 0)
        stop = start + size
        modules.append(np.arange(start, stop, dtype=np.int64))
        start = stop

    return tuple(modules)


def _node_to_module(modules: tuple[np.ndarray, ...]) -> tuple[int, ...]:
    n_cells = sum(len(module) for module in modules)
    mapping = [0] * n_cells
    for module_id, module_nodes in enumerate(modules):
        for node in module_nodes:
            mapping[int(node)] = module_id
    return tuple(mapping)


def _cross_module_candidates(
    *,
    node_to_module: tuple[int, ...],
    module_id: int,
    connected_sources: set[int],
) -> np.ndarray:
    return np.array(
        [
            source
            for source, source_module_id in enumerate(node_to_module)
            if source_module_id != module_id and source not in connected_sources
        ],
        dtype=np.int64,
    )


def _sample_non_zero_weights(
    *,
    rng: np.random.Generator,
    size: int,
    dtype: np.dtype[np.floating],
) -> FloatArray:
    if size == 0:
        return np.array([], dtype=dtype)

    values = rng.uniform(-1.0, 1.0, size=size).astype(dtype)
    zero_mask = values == 0.0
    if np.any(zero_mask):
        values[zero_mask] = np.array(1.0, dtype=dtype)
    return values


def _derive_seed(seed: int, label: str) -> int:
    payload = f"{seed}:{label}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)
