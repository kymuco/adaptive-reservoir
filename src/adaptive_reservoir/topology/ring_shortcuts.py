"""Ring topology builder with seeded shortcut edges."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from adaptive_reservoir.core.config import ReservoirConfig
from adaptive_reservoir.core.protocols import FloatArray

_RING_SHORTCUTS_SEED_LABEL = "topology.ring_shortcuts"


@dataclass(frozen=True, slots=True)
class RingShortcutsTopologyBuilder:
    """Build dense matrices with ring edges and random shortcut inputs.

    Matrix convention: ``weights[target_cell, source_cell]``.
    """

    shortcuts_per_node: int = 1
    bidirectional: bool = True
    allow_self_loops: bool = False
    ring_weight: float = 1.0
    shortcut_weight_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.shortcuts_per_node < 0:
            msg = "shortcuts_per_node must be non-negative"
            raise ValueError(msg)
        if not math.isfinite(self.ring_weight) or self.ring_weight == 0.0:
            msg = "ring_weight must be finite and non-zero"
            raise ValueError(msg)
        if not math.isfinite(self.shortcut_weight_scale) or self.shortcut_weight_scale <= 0.0:
            msg = "shortcut_weight_scale must be finite and positive"
            raise ValueError(msg)

    def build(self, config: ReservoirConfig) -> FloatArray:
        """Build a recurrent ring-shortcuts weight matrix for ``config``."""

        if config.topology != "ring_shortcuts":
            msg = "config.topology must be 'ring_shortcuts'"
            raise ValueError(msg)
        self._validate_n_cells(config.n_cells)

        dtype = np.dtype(config.dtype)
        shortcut_scale = self._validated_shortcut_scale(dtype)
        weights = np.zeros((config.n_cells, config.n_cells), dtype=dtype)
        rng = np.random.default_rng(
            _derive_seed(config.seed, _RING_SHORTCUTS_SEED_LABEL)
        )

        for target in range(config.n_cells):
            protected_sources = set(_ring_sources(target, config.n_cells, self.bidirectional))
            for source in protected_sources:
                weights[target, source] = self.ring_weight

            shortcut_sources = rng.choice(
                _shortcut_candidates(
                    n_cells=config.n_cells,
                    protected_sources=protected_sources,
                    target=target,
                    allow_self_loops=self.allow_self_loops,
                ),
                size=self.shortcuts_per_node,
                replace=False,
            )
            weights[target, shortcut_sources] = _sample_non_zero_weights(
                rng=rng,
                size=self.shortcuts_per_node,
                dtype=dtype,
                weight_scale=shortcut_scale,
            )

        return weights

    def _validate_n_cells(self, n_cells: int) -> None:
        min_cells = 3 if self.bidirectional else 2
        if n_cells < min_cells:
            msg = f"n_cells must be >= {min_cells} for this ring configuration"
            raise ValueError(msg)

        ring_degree = 2 if self.bidirectional else 1
        blocked_sources = ring_degree if self.allow_self_loops else ring_degree + 1
        max_shortcuts = n_cells - blocked_sources
        if self.shortcuts_per_node > max_shortcuts:
            msg = (
                f"shortcuts_per_node must be <= {max_shortcuts} "
                f"for n_cells={n_cells}"
            )
            raise ValueError(msg)

    def _validated_shortcut_scale(self, dtype: np.dtype[np.floating]) -> float:
        dtype_scale = np.array(self.shortcut_weight_scale, dtype=dtype)
        if dtype_scale == 0.0:
            msg = "shortcut_weight_scale is too small for config.dtype"
            raise ValueError(msg)
        return float(dtype_scale)


def _ring_sources(target: int, n_cells: int, bidirectional: bool) -> tuple[int, ...]:
    previous_source = (target - 1) % n_cells
    if not bidirectional:
        return (previous_source,)
    next_source = (target + 1) % n_cells
    return (previous_source, next_source)


def _shortcut_candidates(
    *,
    n_cells: int,
    protected_sources: set[int],
    target: int,
    allow_self_loops: bool,
) -> np.ndarray:
    excluded_sources = set(protected_sources)
    if not allow_self_loops:
        excluded_sources.add(target)
    return np.array(
        [source for source in range(n_cells) if source not in excluded_sources],
        dtype=np.int64,
    )


def _sample_non_zero_weights(
    *,
    rng: np.random.Generator,
    size: int,
    dtype: np.dtype[np.floating],
    weight_scale: float,
) -> FloatArray:
    if size == 0:
        return np.array([], dtype=dtype)

    values = rng.uniform(-weight_scale, weight_scale, size=size).astype(dtype)
    zero_mask = values == 0.0
    if np.any(zero_mask):
        values[zero_mask] = np.array(weight_scale, dtype=dtype)
    return values


def _derive_seed(seed: int, label: str) -> int:
    payload = f"{seed}:{label}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)
