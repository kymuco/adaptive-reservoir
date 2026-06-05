from dataclasses import replace

import numpy as np
import pytest

from adaptive_reservoir import ReservoirConfig, TopologyBuilderProtocol
from adaptive_reservoir.topology import ModularSmallWorldTopologyBuilder


def test_modular_small_world_builder_satisfies_topology_protocol() -> None:
    assert isinstance(ModularSmallWorldTopologyBuilder(), TopologyBuilderProtocol)


def test_modular_small_world_topology_has_expected_shape_and_dtype() -> None:
    config = _config(n_cells=12, dtype="float32")
    weights = ModularSmallWorldTopologyBuilder(
        n_modules=3,
        intra_module_degree=2,
        inter_module_shortcuts=1,
    ).build(config)

    assert weights.shape == (12, 12)
    assert weights.dtype == np.float32


def test_modular_small_world_topology_has_local_intra_module_edges() -> None:
    config = _config(n_cells=8)
    builder = ModularSmallWorldTopologyBuilder(
        n_modules=2,
        intra_module_degree=2,
        inter_module_shortcuts=0,
        rewire_prob=0.0,
    )

    weights = builder.build(config)

    np.testing.assert_array_equal(
        np.count_nonzero(weights, axis=1),
        np.full(config.n_cells, 2),
    )
    for target in range(config.n_cells):
        assert set(np.flatnonzero(weights[target])) == _expected_local_sources(target)


def test_modular_small_world_topology_adds_fixed_inter_module_shortcuts() -> None:
    config = _config(n_cells=12)
    builder = ModularSmallWorldTopologyBuilder(
        n_modules=3,
        intra_module_degree=2,
        inter_module_shortcuts=2,
        rewire_prob=0.0,
    )

    weights = builder.build(config)

    np.testing.assert_array_equal(
        np.count_nonzero(weights, axis=1),
        np.full(config.n_cells, 4),
    )
    for target in range(config.n_cells):
        same_module_count, cross_module_count = _source_counts_by_module(
            target,
            weights,
            n_modules=3,
        )
        assert same_module_count == 2
        assert cross_module_count == 2


def test_modular_small_world_topology_is_seeded() -> None:
    config = _config(n_cells=12, seed=42)
    builder = ModularSmallWorldTopologyBuilder(
        n_modules=3,
        intra_module_degree=2,
        inter_module_shortcuts=2,
        rewire_prob=0.5,
    )

    left = builder.build(config)
    right = builder.build(config)

    np.testing.assert_array_equal(left, right)


def test_modular_small_world_topology_changes_with_seed() -> None:
    config = _config(n_cells=12, seed=42)
    builder = ModularSmallWorldTopologyBuilder(
        n_modules=3,
        intra_module_degree=2,
        inter_module_shortcuts=2,
        rewire_prob=0.5,
    )

    left = builder.build(config)
    right = builder.build(replace(config, seed=43))

    assert not np.array_equal(left, right)


def test_modular_small_world_topology_rewires_to_cross_module_edges() -> None:
    config = _config(n_cells=12)
    builder = ModularSmallWorldTopologyBuilder(
        n_modules=3,
        intra_module_degree=2,
        inter_module_shortcuts=0,
        rewire_prob=1.0,
    )

    weights = builder.build(config)

    np.testing.assert_array_equal(
        np.count_nonzero(weights, axis=1),
        np.full(config.n_cells, 2),
    )
    for target in range(config.n_cells):
        same_module_count, cross_module_count = _source_counts_by_module(
            target,
            weights,
            n_modules=3,
        )
        assert same_module_count == 0
        assert cross_module_count == 2


def test_modular_small_world_topology_has_no_self_loops() -> None:
    config = _config(n_cells=12)
    builder = ModularSmallWorldTopologyBuilder(
        n_modules=3,
        intra_module_degree=2,
        inter_module_shortcuts=2,
        rewire_prob=0.5,
    )

    weights = builder.build(config)

    np.testing.assert_array_equal(np.diag(weights), np.zeros(config.n_cells))


def test_modular_small_world_topology_validates_topology_name() -> None:
    config = ReservoirConfig(input_dim=2, n_cells=12, topology="ring_shortcuts")
    builder = ModularSmallWorldTopologyBuilder()

    with pytest.raises(
        ValueError,
        match="config.topology must be 'modular_small_world'",
    ):
        builder.build(config)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_modules": 1}, "n_modules must be >= 2"),
        ({"intra_module_degree": 0}, "intra_module_degree must be positive"),
        (
            {"inter_module_shortcuts": -1},
            "inter_module_shortcuts must be non-negative",
        ),
        ({"rewire_prob": -0.1}, "rewire_prob must be finite"),
        ({"rewire_prob": 1.1}, "rewire_prob must be finite"),
        ({"rewire_prob": float("nan")}, "rewire_prob must be finite"),
    ],
)
def test_modular_small_world_builder_validates_constructor_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ModularSmallWorldTopologyBuilder(**kwargs)  # type: ignore[arg-type]


def test_modular_small_world_topology_validates_module_count() -> None:
    config = _config(n_cells=4)
    builder = ModularSmallWorldTopologyBuilder(n_modules=5)

    with pytest.raises(ValueError, match="n_modules must be <= n_cells"):
        builder.build(config)


def test_modular_small_world_topology_validates_intra_module_capacity() -> None:
    config = _config(n_cells=6)
    builder = ModularSmallWorldTopologyBuilder(
        n_modules=3,
        intra_module_degree=2,
        inter_module_shortcuts=0,
    )

    with pytest.raises(ValueError, match="intra_module_degree must be <= 1"):
        builder.build(config)


def test_modular_small_world_topology_validates_inter_module_capacity() -> None:
    config = _config(n_cells=8)
    builder = ModularSmallWorldTopologyBuilder(
        n_modules=2,
        intra_module_degree=1,
        inter_module_shortcuts=5,
    )

    with pytest.raises(ValueError, match="inter_module_shortcuts must be <= 4"):
        builder.build(config)


def test_modular_small_world_topology_validates_rewire_capacity() -> None:
    config = _config(n_cells=8)
    builder = ModularSmallWorldTopologyBuilder(
        n_modules=2,
        intra_module_degree=3,
        inter_module_shortcuts=2,
        rewire_prob=0.5,
    )

    with pytest.raises(ValueError, match="when rewire_prob > 0"):
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
        topology="modular_small_world",
        seed=seed,
        dtype=dtype,  # type: ignore[arg-type]
    )


def _expected_local_sources(target: int) -> set[int]:
    module_start = 0 if target < 4 else 4
    module_nodes = list(range(module_start, module_start + 4))
    local_index = target - module_start
    return {
        module_nodes[(local_index - 1) % len(module_nodes)],
        module_nodes[(local_index - 2) % len(module_nodes)],
    }


def _source_counts_by_module(
    target: int,
    weights: np.ndarray,
    *,
    n_modules: int,
) -> tuple[int, int]:
    sources = np.flatnonzero(weights[target])
    target_module = _module_id(target, n_cells=weights.shape[0], n_modules=n_modules)
    same_module_count = sum(
        _module_id(int(source), n_cells=weights.shape[0], n_modules=n_modules)
        == target_module
        for source in sources
    )
    return same_module_count, len(sources) - same_module_count


def _module_id(node: int, *, n_cells: int, n_modules: int) -> int:
    base_size = n_cells // n_modules
    remainder = n_cells % n_modules
    cursor = 0
    for module_id in range(n_modules):
        size = base_size + (1 if module_id < remainder else 0)
        if cursor <= node < cursor + size:
            return module_id
        cursor += size
    raise AssertionError("node index outside module partition")
