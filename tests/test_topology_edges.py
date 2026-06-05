import numpy as np
import pytest

from adaptive_reservoir.topology import EdgeList


def test_edge_list_reports_edge_count() -> None:
    edge_list = _sample_edge_list()

    assert edge_list.n_edges == 3


def test_edge_list_returns_edge_index() -> None:
    edge_list = _sample_edge_list()

    np.testing.assert_array_equal(
        edge_list.edge_index,
        np.array(
            [
                [0, 0, 1],
                [1, 2, 2],
            ],
            dtype=np.int64,
        ),
    )


def test_edge_list_to_dense_uses_target_source_convention() -> None:
    edge_list = _sample_edge_list()

    dense = edge_list.to_dense()

    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [-0.2, 1.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(dense, expected)


def test_edge_list_to_dense_can_override_dtype() -> None:
    edge_list = _sample_edge_list()

    dense = edge_list.to_dense(dtype="float32")

    assert dense.dtype == np.float32


def test_edge_list_metrics_compute_active_edge_ratio() -> None:
    edge_list = _sample_edge_list()

    assert edge_list.metrics().active_edge_ratio == 3 / 9


def test_edge_list_metrics_compute_in_degree_stats() -> None:
    edge_list = _sample_edge_list()
    stats = edge_list.metrics().in_degree_stats

    assert stats.minimum == 0
    assert stats.maximum == 2
    assert stats.mean == 1.0


def test_edge_list_metrics_compute_out_degree_stats() -> None:
    edge_list = _sample_edge_list()
    stats = edge_list.metrics().out_degree_stats

    assert stats.minimum == 0
    assert stats.maximum == 2
    assert stats.mean == 1.0


def test_edge_list_metrics_reports_no_module_count_without_modules() -> None:
    edge_list = _sample_edge_list()

    assert edge_list.metrics().module_count is None


def test_edge_list_metrics_reports_module_count() -> None:
    edge_list = EdgeList(
        n_nodes=3,
        sources=np.array([0, 0, 1]),
        targets=np.array([1, 2, 2]),
        weights=np.array([0.5, -0.2, 1.0]),
        module_ids=np.array([0, 0, 1]),
    )

    assert edge_list.metrics().module_count == 2


def test_edge_list_validates_positive_node_count() -> None:
    with pytest.raises(ValueError, match="n_nodes must be positive"):
        EdgeList(
            n_nodes=0,
            sources=np.array([], dtype=np.int64),
            targets=np.array([], dtype=np.int64),
            weights=np.array([], dtype=np.float64),
        )


@pytest.mark.parametrize(
    ("sources", "targets", "weights", "message"),
    [
        (
            np.array([[0]]),
            np.array([1]),
            np.array([1.0]),
            "sources must be a 1D array",
        ),
        (
            np.array([0]),
            np.array([[1]]),
            np.array([1.0]),
            "targets must be a 1D array",
        ),
        (
            np.array([0]),
            np.array([1]),
            np.array([[1.0]]),
            "weights must be a 1D array",
        ),
        (
            np.array([0, 1]),
            np.array([1]),
            np.array([1.0]),
            "sources, targets, and weights must have the same length",
        ),
        (
            np.array([-1]),
            np.array([1]),
            np.array([1.0]),
            "sources contain indices outside",
        ),
        (
            np.array([0]),
            np.array([3]),
            np.array([1.0]),
            "targets contain indices outside",
        ),
        (
            np.array([0]),
            np.array([1]),
            np.array([float("nan")]),
            "weights must be finite",
        ),
        (
            np.array([0]),
            np.array([1]),
            np.array([0.0]),
            "weights must be non-zero",
        ),
    ],
)
def test_edge_list_validates_edge_arrays(
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        EdgeList(n_nodes=3, sources=sources, targets=targets, weights=weights)


def test_edge_list_rejects_duplicate_directed_edges() -> None:
    with pytest.raises(ValueError, match="duplicate directed edges"):
        EdgeList(
            n_nodes=3,
            sources=np.array([0, 0]),
            targets=np.array([1, 1]),
            weights=np.array([0.5, 1.0]),
        )


def test_edge_list_validates_module_ids_shape() -> None:
    with pytest.raises(ValueError, match="module_ids must be a 1D array"):
        EdgeList(
            n_nodes=3,
            sources=np.array([0]),
            targets=np.array([1]),
            weights=np.array([0.5]),
            module_ids=np.array([[0, 1, 2]]),
        )


def test_edge_list_validates_module_ids_length() -> None:
    with pytest.raises(ValueError, match="module_ids length must match n_nodes"):
        EdgeList(
            n_nodes=3,
            sources=np.array([0]),
            targets=np.array([1]),
            weights=np.array([0.5]),
            module_ids=np.array([0, 1]),
        )


def test_edge_list_validates_module_ids_non_negative() -> None:
    with pytest.raises(ValueError, match="module_ids must be non-negative"):
        EdgeList(
            n_nodes=3,
            sources=np.array([0]),
            targets=np.array([1]),
            weights=np.array([0.5]),
            module_ids=np.array([0, -1, 1]),
        )


def _sample_edge_list() -> EdgeList:
    return EdgeList(
        n_nodes=3,
        sources=np.array([0, 0, 1]),
        targets=np.array([1, 2, 2]),
        weights=np.array([0.5, -0.2, 1.0]),
    )
