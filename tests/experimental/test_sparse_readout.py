from __future__ import annotations

import numpy as np
import pytest

from adaptive_reservoir.core.config import READOUT_NAMES, ReadoutConfig
from adaptive_reservoir.experimental.sparse_readout import (
    SPARSE_ONLINE_READOUT_NAME,
    SparseOnlineReadout,
)
from adaptive_reservoir.readout.base import READOUT_SNAPSHOT_SCHEMA_VERSION, ReadoutSnapshot


def test_predict_starts_at_zero() -> None:
    readout = SparseOnlineReadout(feature_dim=3)

    assert readout.predict([1.0, -2.0, 0.5]) == 0.0
    assert readout.samples_seen == 0


def test_update_moves_prediction_toward_target() -> None:
    readout = SparseOnlineReadout(feature_dim=2, learning_rate=0.1, l1_strength=0.0)

    before = readout.predict([1.0, 0.0])
    readout.update([1.0, 0.0], 1.0)
    after = readout.predict([1.0, 0.0])

    assert before == 0.0
    assert after > before
    assert readout.samples_seen == 1


def test_l1_shrinkage_increases_near_zero_weights() -> None:
    dense = SparseOnlineReadout(feature_dim=4, learning_rate=0.1, l1_strength=0.0)
    sparse = SparseOnlineReadout(feature_dim=4, learning_rate=0.1, l1_strength=0.02)
    samples = (
        ([1.0, 0.25, -0.5, 0.1], 0.8),
        ([-0.5, 1.0, 0.25, -0.2], -0.4),
        ([0.25, -0.5, 1.0, 0.3], 0.6),
        ([0.1, 0.25, -0.3, 1.0], -0.2),
    )

    for _ in range(24):
        for features, target in samples:
            dense.update(features, target)
            sparse.update(features, target)

    dense_sparsity = float(np.mean(np.abs(dense.weights) <= 1e-8))
    sparse_sparsity = float(np.mean(np.abs(sparse.weights) <= 1e-8))

    assert sparse_sparsity >= dense_sparsity


def test_strong_l1_can_zero_small_weight_updates() -> None:
    readout = SparseOnlineReadout(feature_dim=2, learning_rate=0.05, l1_strength=0.1)

    readout.update([0.1, 0.0], 0.1)

    assert readout.weights[0] == 0.0
    assert readout.weights[1] == 0.0
    assert readout.bias != 0.0


def test_weights_are_read_only_copies() -> None:
    readout = SparseOnlineReadout(feature_dim=2, learning_rate=0.1, l1_strength=0.0)
    readout.update([1.0, 0.0], 1.0)

    weights = readout.weights

    assert not weights.flags.writeable
    with pytest.raises(ValueError):
        weights[0] = 99.0
    assert readout.weights[0] != 99.0


def test_snapshot_restore_roundtrip() -> None:
    readout = SparseOnlineReadout(
        feature_dim=3,
        learning_rate=0.2,
        l1_strength=0.01,
        epsilon=1e-7,
        dtype="float32",
    )
    readout.update([1.0, 0.5, -0.25], 0.7)
    readout.update([-0.25, 1.0, 0.5], -0.2)
    snapshot = readout.snapshot()

    restored = SparseOnlineReadout(
        feature_dim=3,
        learning_rate=0.2,
        l1_strength=0.01,
        epsilon=1e-7,
        dtype="float32",
    )
    restored.restore(snapshot)

    features = [0.25, -0.5, 1.0]
    assert snapshot.name == SPARSE_ONLINE_READOUT_NAME
    assert restored.samples_seen == readout.samples_seen
    assert restored.bias == readout.bias
    assert np.allclose(restored.weights, readout.weights)
    assert restored.predict(features) == pytest.approx(readout.predict(features))


@pytest.mark.parametrize(
    "snapshot",
    [
        ReadoutSnapshot(
            schema_version=READOUT_SNAPSHOT_SCHEMA_VERSION,
            name="nlms",
            state={
                "feature_dim": 2,
                "dtype": "float64",
                "learning_rate": 0.05,
                "l1_strength": 1e-4,
                "epsilon": 1e-8,
                "weights": (0.0, 0.0),
                "bias": 0.0,
                "samples_seen": 0,
            },
        ),
        ReadoutSnapshot(
            schema_version=READOUT_SNAPSHOT_SCHEMA_VERSION,
            name=SPARSE_ONLINE_READOUT_NAME,
            state={
                "feature_dim": 3,
                "dtype": "float64",
                "learning_rate": 0.05,
                "l1_strength": 1e-4,
                "epsilon": 1e-8,
                "weights": (0.0, 0.0, 0.0),
                "bias": 0.0,
                "samples_seen": 0,
            },
        ),
        ReadoutSnapshot(
            schema_version=READOUT_SNAPSHOT_SCHEMA_VERSION,
            name=SPARSE_ONLINE_READOUT_NAME,
            state={
                "feature_dim": 2,
                "dtype": "float32",
                "learning_rate": 0.05,
                "l1_strength": 1e-4,
                "epsilon": 1e-8,
                "weights": (0.0, 0.0),
                "bias": 0.0,
                "samples_seen": 0,
            },
        ),
        ReadoutSnapshot(
            schema_version=READOUT_SNAPSHOT_SCHEMA_VERSION,
            name=SPARSE_ONLINE_READOUT_NAME,
            state={
                "feature_dim": 2,
                "dtype": "float64",
                "learning_rate": 0.1,
                "l1_strength": 1e-4,
                "epsilon": 1e-8,
                "weights": (0.0, 0.0),
                "bias": 0.0,
                "samples_seen": 0,
            },
        ),
        ReadoutSnapshot(
            schema_version=READOUT_SNAPSHOT_SCHEMA_VERSION,
            name=SPARSE_ONLINE_READOUT_NAME,
            state={
                "feature_dim": 2,
                "dtype": "float64",
                "learning_rate": 0.05,
                "l1_strength": 0.01,
                "epsilon": 1e-8,
                "weights": (0.0, 0.0),
                "bias": 0.0,
                "samples_seen": 0,
            },
        ),
        ReadoutSnapshot(
            schema_version=READOUT_SNAPSHOT_SCHEMA_VERSION,
            name=SPARSE_ONLINE_READOUT_NAME,
            state={
                "feature_dim": 2,
                "dtype": "float64",
                "learning_rate": 0.05,
                "l1_strength": 1e-4,
                "epsilon": 1e-7,
                "weights": (0.0, 0.0),
                "bias": 0.0,
                "samples_seen": 0,
            },
        ),
        ReadoutSnapshot(
            schema_version=READOUT_SNAPSHOT_SCHEMA_VERSION,
            name=SPARSE_ONLINE_READOUT_NAME,
            state={
                "feature_dim": 2,
                "dtype": "float64",
                "learning_rate": 0.05,
                "l1_strength": 1e-4,
                "epsilon": 1e-8,
                "weights": (0.0, 0.0),
                "bias": 0.0,
                "samples_seen": -1,
            },
        ),
    ],
)
def test_restore_rejects_incompatible_snapshots(snapshot: ReadoutSnapshot) -> None:
    readout = SparseOnlineReadout(feature_dim=2)

    with pytest.raises(ValueError):
        readout.restore(snapshot)


def test_restore_rejects_non_readout_snapshot() -> None:
    readout = SparseOnlineReadout(feature_dim=2)

    with pytest.raises(TypeError):
        readout.restore({})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_dim": 0},
        {"feature_dim": True},
        {"feature_dim": 2, "learning_rate": 0.0},
        {"feature_dim": 2, "learning_rate": float("inf")},
        {"feature_dim": 2, "l1_strength": -1.0},
        {"feature_dim": 2, "l1_strength": float("nan")},
        {"feature_dim": 2, "epsilon": 0.0},
        {"feature_dim": 2, "dtype": "int64"},
    ],
)
def test_constructor_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SparseOnlineReadout(**kwargs)  # type: ignore[arg-type]


def test_predict_and_update_validate_inputs() -> None:
    readout = SparseOnlineReadout(feature_dim=2)

    with pytest.raises(ValueError):
        readout.predict([1.0])
    with pytest.raises(ValueError):
        readout.predict([1.0, float("nan")])
    with pytest.raises(ValueError):
        readout.update([1.0, 2.0], float("inf"))


def test_sparse_readout_is_not_registered_as_stable_readout() -> None:
    assert SPARSE_ONLINE_READOUT_NAME not in READOUT_NAMES

    with pytest.raises(ValueError):
        ReadoutConfig(name=SPARSE_ONLINE_READOUT_NAME)  # type: ignore[arg-type]
