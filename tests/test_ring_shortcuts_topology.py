from dataclasses import replace

import numpy as np
import pytest

from adaptive_reservoir import ReservoirConfig, TopologyBuilderProtocol
from adaptive_reservoir.topology import EdgeList, RingShortcutsTopologyBuilder


def test_ring_shortcuts_builder_satisfies_topology_protocol() -> None:
    assert isinstance(RingShortcutsTopologyBuilder(), TopologyBuilderProtocol)


def test_ring_shortcuts_topology_returns_edge_list() -> None:
    config = _config(n_cells=8)
    edge_list = RingShortcutsTopologyBuilder(shortcuts_per_node=2).build(config)

    assert isinstance(edge_list, EdgeList)
    assert edge_list.n_nodes == 8
    assert edge_list.n_edges == 32


def test_ring_shortcuts_topology_has_expected_shape_and_dtype() -> None:
    config = _config(n_cells=8, dtype="float32")
    edge_list = RingShortcutsTopologyBuilder(shortcuts_per_node=1).build(config)
    weights = edge_list.to_dense()

    assert weights.shape == (8, 8)
    assert weights.dtype == np.float32


def test_ring_shortcuts_topology_has_forward_ring_edges() -> None:
    config = _config(n_cells=6)
    weights = RingShortcutsTopologyBuilder(
        shortcuts_per_node=0,
        bidirectional=False,
    ).build(config).to_dense()

    for target in range(config.n_cells):
        assert weights[target, (target - 1) % config.n_cells] != 0.0
    np.testing.assert_array_equal(
        np.count_nonzero(weights, axis=1),
        np.full(config.n_cells, 1),
    )


def test_ring_shortcuts_topology_has_bidirectional_ring_edges() -> None:
    config = _config(n_cells=6)
    weights = RingShortcutsTopologyBuilder(
        shortcuts_per_node=0,
        bidirectional=True,
    ).build(config).to_dense()

    for target in range(config.n_cells):
        assert weights[target, (target - 1) % config.n_cells] != 0.0
        assert weights[target, (target + 1) % config.n_cells] != 0.0
    np.testing.assert_array_equal(
        np.count_nonzero(weights, axis=1),
        np.full(config.n_cells, 2),
    )


def test_ring_shortcuts_topology_adds_fixed_shortcuts_per_node() -> None:
    config = _config(n_cells=8)
    edge_list = RingShortcutsTopologyBuilder(
        shortcuts_per_node=2,
        bidirectional=True,
    ).build(config)
    weights = edge_list.to_dense()

    np.testing.assert_array_equal(
        np.count_nonzero(weights, axis=1),
        np.full(config.n_cells, 4),
    )
    assert edge_list.metrics().in_degree_stats.minimum == 4
    assert edge_list.metrics().in_degree_stats.maximum == 4
    assert edge_list.metrics().in_degree_stats.mean == 4.0


def test_ring_shortcuts_topology_has_expected_metrics() -> None:
    config = _config(n_cells=8)
    edge_list = RingShortcutsTopologyBuilder(
        shortcuts_per_node=2,
        bidirectional=True,
    ).build(config)
    metrics = edge_list.metrics()

    assert metrics.active_edge_ratio == 32 / 64
    assert metrics.out_degree_stats.mean == 4.0
    assert metrics.module_count is None


def test_ring_shortcuts_topology_is_seeded() -> None:
    config = _config(n_cells=8, seed=42)
    builder = RingShortcutsTopologyBuilder(shortcuts_per_node=2)

    left = builder.build(config)
    right = builder.build(config)

    np.testing.assert_array_equal(left.sources, right.sources)
    np.testing.assert_array_equal(left.targets, right.targets)
    np.testing.assert_array_equal(left.weights, right.weights)


def test_ring_shortcuts_topology_changes_with_seed() -> None:
    config = _config(n_cells=8, seed=42)
    builder = RingShortcutsTopologyBuilder(shortcuts_per_node=2)

    left = builder.build(config).to_dense()
    right = builder.build(replace(config, seed=43)).to_dense()

    assert not np.array_equal(left, right)


def test_ring_shortcuts_topology_disables_self_loops_by_default() -> None:
    config = _config(n_cells=8)
    weights = RingShortcutsTopologyBuilder(shortcuts_per_node=2).build(config).to_dense()

    np.testing.assert_array_equal(np.diag(weights), np.zeros(config.n_cells))


def test_ring_shortcuts_topology_can_enable_self_loop_shortcuts() -> None:
    config = _config(n_cells=4)
    weights = RingShortcutsTopologyBuilder(
        shortcuts_per_node=2,
        bidirectional=True,
        allow_self_loops=True,
    ).build(config).to_dense()

    assert np.all(np.diag(weights) != 0.0)
    np.testing.assert_array_equal(
        np.count_nonzero(weights, axis=1),
        np.full(config.n_cells, 4),
    )


def test_ring_shortcuts_topology_validates_topology_name() -> None:
    config = ReservoirConfig(input_dim=2, n_cells=8, topology="random_sparse")
    builder = RingShortcutsTopologyBuilder(shortcuts_per_node=1)

    with pytest.raises(ValueError, match="config.topology must be 'ring_shortcuts'"):
        builder.build(config)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shortcuts_per_node": -1}, "shortcuts_per_node must be non-negative"),
        ({"ring_weight": 0.0}, "ring_weight must be finite and non-zero"),
        ({"ring_weight": float("nan")}, "ring_weight must be finite and non-zero"),
        (
            {"shortcut_weight_scale": 0.0},
            "shortcut_weight_scale must be finite and positive",
        ),
        (
            {"shortcut_weight_scale": float("inf")},
            "shortcut_weight_scale must be finite and positive",
        ),
    ],
)
def test_ring_shortcuts_builder_validates_constructor_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RingShortcutsTopologyBuilder(**kwargs)  # type: ignore[arg-type]


def test_ring_shortcuts_topology_validates_unidirectional_min_cells() -> None:
    config = _config(n_cells=1)
    builder = RingShortcutsTopologyBuilder(
        shortcuts_per_node=0,
        bidirectional=False,
    )

    with pytest.raises(ValueError, match="n_cells must be >= 2"):
        builder.build(config)


def test_ring_shortcuts_topology_validates_bidirectional_min_cells() -> None:
    config = _config(n_cells=2)
    builder = RingShortcutsTopologyBuilder(
        shortcuts_per_node=0,
        bidirectional=True,
    )

    with pytest.raises(ValueError, match="n_cells must be >= 3"):
        builder.build(config)


def test_ring_shortcuts_topology_validates_shortcut_capacity() -> None:
    config = _config(n_cells=4)
    builder = RingShortcutsTopologyBuilder(
        shortcuts_per_node=2,
        bidirectional=True,
        allow_self_loops=False,
    )

    with pytest.raises(ValueError, match="shortcuts_per_node must be <= 1"):
        builder.build(config)


def test_ring_shortcuts_topology_validates_shortcut_scale_for_dtype() -> None:
    config = _config(n_cells=8, dtype="float32")
    builder = RingShortcutsTopologyBuilder(shortcut_weight_scale=1e-80)

    with pytest.raises(ValueError, match="shortcut_weight_scale is too small"):
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
        topology="ring_shortcuts",
        seed=seed,
        dtype=dtype,  # type: ignore[arg-type]
    )
