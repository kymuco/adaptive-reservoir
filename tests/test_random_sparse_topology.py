from dataclasses import replace

import numpy as np
import pytest

from adaptive_reservoir import ReservoirConfig, TopologyBuilderProtocol
from adaptive_reservoir.topology import RandomSparseTopologyBuilder


def test_random_sparse_builder_satisfies_topology_protocol() -> None:
    assert isinstance(RandomSparseTopologyBuilder(), TopologyBuilderProtocol)


def test_random_sparse_topology_has_expected_shape_and_dtype() -> None:
    config = _config(n_cells=8, dtype="float32")
    weights = RandomSparseTopologyBuilder(in_degree=3).build(config)

    assert weights.shape == (8, 8)
    assert weights.dtype == np.float32


def test_random_sparse_topology_has_fixed_in_degree() -> None:
    config = _config(n_cells=8)
    weights = RandomSparseTopologyBuilder(in_degree=3).build(config)

    np.testing.assert_array_equal(
        np.count_nonzero(weights, axis=1),
        np.full(8, 3),
    )


def test_random_sparse_topology_has_no_invalid_source_indices() -> None:
    config = _config(n_cells=8)
    weights = RandomSparseTopologyBuilder(in_degree=3).build(config)

    for target in range(config.n_cells):
        sources = np.flatnonzero(weights[target])
        assert len(sources) == 3
        assert np.all(sources >= 0)
        assert np.all(sources < config.n_cells)


def test_random_sparse_topology_disables_self_loops_by_default() -> None:
    config = _config(n_cells=8)
    weights = RandomSparseTopologyBuilder(in_degree=3).build(config)

    np.testing.assert_array_equal(np.diag(weights), np.zeros(config.n_cells))


def test_random_sparse_topology_can_enable_self_loops() -> None:
    config = _config(n_cells=4)
    weights = RandomSparseTopologyBuilder(
        in_degree=4,
        allow_self_loops=True,
    ).build(config)

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

    np.testing.assert_array_equal(left, right)


def test_random_sparse_topology_changes_with_seed() -> None:
    config = _config(n_cells=8, seed=42)
    builder = RandomSparseTopologyBuilder(in_degree=3)

    left = builder.build(config)
    right = builder.build(replace(config, seed=43))

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
