from dataclasses import replace

import numpy as np
import pytest

from adaptive_reservoir import ReservoirConfig, TopologyBuilderProtocol
from adaptive_reservoir.topology import EdgeList, RandomSparseTopologyBuilder


def test_random_sparse_builder_satisfies_topology_protocol() -> None:
    assert isinstance(RandomSparseTopologyBuilder(), TopologyBuilderProtocol)


def test_random_sparse_topology_returns_edge_list() -> None:
    config = _config(n_cells=8)
    edge_list = RandomSparseTopologyBuilder(in_degree=3).build(config)

    assert isinstance(edge_list, EdgeList)
    assert edge_list.n_nodes == 8
    assert edge_list.n_edges == 24


def test_random_sparse_topology_has_expected_shape_and_dtype() -> None:
    config = _config(n_cells=8, dtype="float32")
    edge_list = RandomSparseTopologyBuilder(in_degree=3).build(config)
    weights = edge_list.to_dense()

    assert weights.shape == (8, 8)
    assert weights.dtype == np.float32


def test_random_sparse_topology_has_fixed_in_degree() -> None:
    config = _config(n_cells=8)
    edge_list = RandomSparseTopologyBuilder(in_degree=3).build(config)
    weights = edge_list.to_dense()

    np.testing.assert_array_equal(
        np.count_nonzero(weights, axis=1),
        np.full(8, 3),
    )
    assert edge_list.metrics().in_degree_stats.minimum == 3
    assert edge_list.metrics().in_degree_stats.maximum == 3
    assert edge_list.metrics().in_degree_stats.mean == 3.0


def test_random_sparse_topology_has_expected_metrics() -> None:
    config = _config(n_cells=8)
    edge_list = RandomSparseTopologyBuilder(in_degree=3).build(config)
    metrics = edge_list.metrics()

    assert metrics.active_edge_ratio == 24 / 64
    assert metrics.out_degree_stats.mean == 3.0
    assert metrics.module_count is None


def test_random_sparse_topology_has_no_invalid_source_indices() -> None:
    config = _config(n_cells=8)
    weights = RandomSparseTopologyBuilder(in_degree=3).build(config).to_dense()

    for target in range(config.n_cells):
        sources = np.flatnonzero(weights[target])
        assert len(sources) == 3
        assert np.all(sources >= 0)
        assert np.all(sources < config.n_cells)


def test_random_sparse_topology_disables_self_loops_by_default() -> None:
    config = _config(n_cells=8)
    weights = RandomSparseTopologyBuilder(in_degree=3).build(config).to_dense()

    np.testing.assert_array_equal(np.diag(weights), np.zeros(config.n_cells))


def test_random_sparse_topology_can_enable_self_loops() -> None:
    config = _config(n_cells=4)
    weights = RandomSparseTopologyBuilder(
        in_degree=4,
        allow_self_loops=True,
    ).build(config).to_dense()

    assert np.all(np.diag(weights) != 0.0)
    np.testing.assert_array_equal(
        np.count_nonzero(weights, axis=1),
        np.full(4, 4),
    )


def test_random_sparse_topology_is_seeded() -> None:
    config = _config(n_cells=8, seed=42)
    builder = RandomSparseTopologyBuilder(in_degree=3)

    left = builder.build(config)
    right = builder.build(config)

    np.testing.assert_array_equal(left.sources, right.sources)
    np.testing.assert_array_equal(left.targets, right.targets)
    np.testing.assert_array_equal(left.weights, right.weights)


def test_random_sparse_topology_changes_with_seed() -> None:
    config = _config(n_cells=8, seed=42)
    builder = RandomSparseTopologyBuilder(in_degree=3)

    left = builder.build(config).to_dense()
    right = builder.build(replace(config, seed=43)).to_dense()

    assert not np.array_equal(left, right)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"in_degree": 0}, "in_degree must be positive"),
        ({"weight_scale": 0.0}, "weight_scale must be positive"),
    ],
)
def test_random_sparse_builder_validates_constructor_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RandomSparseTopologyBuilder(**kwargs)  # type: ignore[arg-type]


def test_random_sparse_topology_validates_topology_name() -> None:
    config = ReservoirConfig(input_dim=2, n_cells=8, topology="modular_small_world")
    builder = RandomSparseTopologyBuilder(in_degree=3)

    with pytest.raises(ValueError, match="config.topology must be 'random_sparse'"):
        builder.build(config)


def test_random_sparse_topology_validates_in_degree_without_self_loops() -> None:
    config = _config(n_cells=4)
    builder = RandomSparseTopologyBuilder(in_degree=4, allow_self_loops=False)

    with pytest.raises(ValueError, match="in_degree must be <= 3"):
        builder.build(config)


def test_random_sparse_topology_validates_in_degree_with_self_loops() -> None:
    config = _config(n_cells=4)
    builder = RandomSparseTopologyBuilder(in_degree=5, allow_self_loops=True)

    with pytest.raises(ValueError, match="in_degree must be <= 4"):
        builder.build(config)


def _config(
    *,
    n_cells: int,
    seed: int = 42,
    dtype: str = "float64",
) -> ReservoirConfig:
    return ReservoirConfig(
        input_dim=2,
        n_cells=n_cells,
        topology="random_sparse",
        seed=seed,
        dtype=dtype,  # type: ignore[arg-type]
    )
