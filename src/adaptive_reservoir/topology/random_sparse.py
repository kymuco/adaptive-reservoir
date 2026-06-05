"""Random sparse recurrent topology builder."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from adaptive_reservoir.core.config import ReservoirConfig
from adaptive_reservoir.core.protocols import FloatArray
from adaptive_reservoir.topology.edges import EdgeList

_RANDOM_SPARSE_SEED_LABEL = "topology.random_sparse"


@dataclass(frozen=True, slots=True)
class RandomSparseTopologyBuilder:
    """Build random sparse fixed in-degree edge lists.

    Matrix convention: ``weights[target_cell, source_cell]``.
    """

    in_degree: int = 8
    allow_self_loops: bool = False
    weight_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.in_degree <= 0:
            msg = "in_degree must be positive"
            raise ValueError(msg)
        if self.weight_scale <= 0.0:
            msg = "weight_scale must be positive"
            raise ValueError(msg)

    def build(self, config: ReservoirConfig) -> EdgeList:
        """Build recurrent edges for ``config``."""

        if config.topology != "random_sparse":
            msg = "config.topology must be 'random_sparse'"
            raise ValueError(msg)
        self._validate_in_degree(config.n_cells)

        dtype = np.dtype(config.dtype)
        rng = np.random.default_rng(
            _derive_seed(config.seed, _RANDOM_SPARSE_SEED_LABEL)
        )
        sources_by_target: list[np.ndarray] = []
        weights_by_target: list[FloatArray] = []

        for target in range(config.n_cells):
            sources = rng.choice(
                _candidate_sources(config.n_cells, target, self.allow_self_loops),
                size=self.in_degree,
                replace=False,
            ).astype(np.int64)
            sources_by_target.append(sources)
            weights_by_target.append(
                _sample_non_zero_weights(
                    rng=rng,
                    size=self.in_degree,
                    dtype=dtype,
                    weight_scale=self.weight_scale,
                )
            )

        return EdgeList(
            n_nodes=config.n_cells,
            sources=np.concatenate(sources_by_target),
            targets=np.repeat(np.arange(config.n_cells, dtype=np.int64), self.in_degree),
            weights=np.concatenate(weights_by_target).astype(dtype, copy=False),
        )

    def _validate_in_degree(self, n_cells: int) -> None:
        max_in_degree = n_cells if self.allow_self_loops else n_cells - 1
        if self.in_degree > max_in_degree:
            loop_policy = (
                "with self-loops" if self.allow_self_loops else "without self-loops"
            )
            msg = (
                f"in_degree must be <= {max_in_degree} "
                f"for n_cells={n_cells} {loop_policy}"
            )
            raise ValueError(msg)


def _candidate_sources(n_cells: int, target: int, allow_self_loops: bool) -> np.ndarray:
    if allow_self_loops:
        return np.arange(n_cells, dtype=np.int64)
    return np.concatenate((np.arange(target), np.arange(target + 1, n_cells))).astype(
        np.int64,
        copy=False,
    )


def _sample_non_zero_weights(
    *,
    rng: np.random.Generator,
    size: int,
    dtype: np.dtype[np.floating],
    weight_scale: float,
) -> FloatArray:
    values = rng.uniform(-weight_scale, weight_scale, size=size).astype(dtype)
    zero_mask = values == 0.0
    if np.any(zero_mask):
        values[zero_mask] = np.array(weight_scale, dtype=dtype)
    return values


def _derive_seed(seed: int, label: str) -> int:
    payload = f"{seed}:{label}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)
