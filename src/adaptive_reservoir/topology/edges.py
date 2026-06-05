"""Common edge-list representation and metrics for topologies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.integer]
FloatArray = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class DegreeStats:
    """Summary statistics for integer node degrees."""

    minimum: int
    maximum: int
    mean: float


@dataclass(frozen=True, slots=True)
class TopologyMetrics:
    """Derived topology metrics."""

    active_edge_ratio: float
    in_degree_stats: DegreeStats
    out_degree_stats: DegreeStats
    module_count: int | None


@dataclass(frozen=True, slots=True)
class EdgeList:
    """Directed weighted recurrent edges.

    Matrix convention: ``dense[target, source] = weight``.
    """

    n_nodes: int
    sources: IntArray
    targets: IntArray
    weights: FloatArray
    module_ids: IntArray | None = None

    def __post_init__(self) -> None:
        if self.n_nodes <= 0:
            msg = "n_nodes must be positive"
            raise ValueError(msg)

        sources = np.asarray(self.sources, dtype=np.int64)
        targets = np.asarray(self.targets, dtype=np.int64)
        weights = _as_float_array(self.weights)
        module_ids = _as_module_ids(self.module_ids, self.n_nodes)

        _validate_edge_arrays(
            n_nodes=self.n_nodes,
            sources=sources,
            targets=targets,
            weights=weights,
        )

        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "module_ids", module_ids)

    @property
    def n_edges(self) -> int:
        """Return the number of active directed edges."""

        return int(self.sources.size)

    @property
    def edge_index(self) -> IntArray:
        """Return edges as ``[sources, targets]`` index rows."""

        return np.vstack((self.sources, self.targets))

    def to_dense(self, *, dtype: str | np.dtype | None = None) -> FloatArray:
        """Return a dense recurrent matrix using ``dense[target, source]``."""

        matrix_dtype = self.weights.dtype if dtype is None else np.dtype(dtype)
        dense = np.zeros((self.n_nodes, self.n_nodes), dtype=matrix_dtype)
        dense[self.targets, self.sources] = self.weights.astype(matrix_dtype)
        return dense

    def metrics(self) -> TopologyMetrics:
        """Compute topology metrics from the edge list."""

        in_degrees = np.bincount(self.targets, minlength=self.n_nodes)
        out_degrees = np.bincount(self.sources, minlength=self.n_nodes)
        return TopologyMetrics(
            active_edge_ratio=self.n_edges / float(self.n_nodes * self.n_nodes),
            in_degree_stats=_degree_stats(in_degrees),
            out_degree_stats=_degree_stats(out_degrees),
            module_count=_module_count(self.module_ids),
        )


def _as_float_array(values: object) -> FloatArray:
    weights = np.asarray(values)
    if not np.issubdtype(weights.dtype, np.floating):
        weights = weights.astype(np.float64)
    return weights


def _as_module_ids(values: object, n_nodes: int) -> IntArray | None:
    if values is None:
        return None
    module_ids = np.asarray(values, dtype=np.int64)
    if module_ids.ndim != 1:
        msg = "module_ids must be a 1D array"
        raise ValueError(msg)
    if module_ids.size != n_nodes:
        msg = "module_ids length must match n_nodes"
        raise ValueError(msg)
    if np.any(module_ids < 0):
        msg = "module_ids must be non-negative"
        raise ValueError(msg)
    return module_ids


def _validate_edge_arrays(
    *,
    n_nodes: int,
    sources: IntArray,
    targets: IntArray,
    weights: FloatArray,
) -> None:
    if sources.ndim != 1:
        msg = "sources must be a 1D array"
        raise ValueError(msg)
    if targets.ndim != 1:
        msg = "targets must be a 1D array"
        raise ValueError(msg)
    if weights.ndim != 1:
        msg = "weights must be a 1D array"
        raise ValueError(msg)
    if sources.size != targets.size or sources.size != weights.size:
        msg = "sources, targets, and weights must have the same length"
        raise ValueError(msg)
    if np.any(sources < 0) or np.any(sources >= n_nodes):
        msg = "sources contain indices outside [0, n_nodes)"
        raise ValueError(msg)
    if np.any(targets < 0) or np.any(targets >= n_nodes):
        msg = "targets contain indices outside [0, n_nodes)"
        raise ValueError(msg)
    if np.any(~np.isfinite(weights)):
        msg = "weights must be finite"
        raise ValueError(msg)
    if np.any(weights == 0.0):
        msg = "weights must be non-zero"
        raise ValueError(msg)
    if _has_duplicate_edges(n_nodes=n_nodes, sources=sources, targets=targets):
        msg = "duplicate directed edges are not allowed"
        raise ValueError(msg)


def _has_duplicate_edges(*, n_nodes: int, sources: IntArray, targets: IntArray) -> bool:
    edge_ids = targets.astype(np.int64) * n_nodes + sources.astype(np.int64)
    return np.unique(edge_ids).size != edge_ids.size


def _degree_stats(degrees: NDArray[np.integer]) -> DegreeStats:
    return DegreeStats(
        minimum=int(degrees.min()),
        maximum=int(degrees.max()),
        mean=float(degrees.mean()),
    )


def _module_count(module_ids: IntArray | None) -> int | None:
    if module_ids is None:
        return None
    return int(np.unique(module_ids).size)
