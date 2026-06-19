from __future__ import annotations

import numpy as np
import pytest

import adaptive_reservoir
from adaptive_reservoir.experimental.oja import (
    OJA_COMPRESSOR_NAME,
    OJA_COMPRESSOR_SNAPSHOT_SCHEMA_VERSION,
    OjaCompressor,
    OjaCompressorSnapshot,
)

_COMPONENTS_4X2 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
)
_COMPONENTS_5X2 = (
    (1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0, 0.0),
)
_COMPONENTS_4X3 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
)
_COMPONENTS_BAD_SHAPE = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)


def test_transform_returns_expected_shape() -> None:
    compressor = OjaCompressor(input_dim=5, output_dim=2, seed=7)

    compressed = compressor.transform([1.0, 0.0, -1.0, 0.5, 0.25])

    assert compressed.shape == (2,)
    assert compressed.dtype == np.dtype("float64")
    assert np.all(np.isfinite(compressed))


def test_transform_does_not_mutate_state() -> None:
    compressor = OjaCompressor(input_dim=4, output_dim=2, seed=1)
    before = compressor.components

    compressor.transform([1.0, -0.5, 0.25, 0.75])

    assert compressor.samples_seen == 0
    assert np.allclose(compressor.components, before)


def test_update_mutates_components_and_increments_samples_seen() -> None:
    compressor = OjaCompressor(input_dim=4, output_dim=2, learning_rate=0.05, seed=3)
    before = compressor.components

    compressor.update([1.0, 2.0, -1.0, 0.5])

    assert compressor.samples_seen == 1
    assert not np.allclose(compressor.components, before)


def test_step_returns_projection_before_update() -> None:
    compressor = OjaCompressor(input_dim=4, output_dim=2, learning_rate=0.05, seed=11)
    features = [1.0, 0.25, -0.5, 0.75]
    before_components = compressor.components
    before_projection = compressor.transform(features)

    step_projection = compressor.step(features)

    assert np.allclose(step_projection, before_projection)
    assert compressor.samples_seen == 1
    assert not np.allclose(compressor.components, before_components)


def test_same_seed_initializes_same_components() -> None:
    first = OjaCompressor(input_dim=6, output_dim=3, seed=42)
    second = OjaCompressor(input_dim=6, output_dim=3, seed=42)

    assert np.allclose(first.components, second.components)


def test_different_seed_initializes_different_components() -> None:
    first = OjaCompressor(input_dim=6, output_dim=3, seed=1)
    second = OjaCompressor(input_dim=6, output_dim=3, seed=2)

    assert not np.allclose(first.components, second.components)


def test_components_are_read_only_copies() -> None:
    compressor = OjaCompressor(input_dim=4, output_dim=2, seed=5)

    components = compressor.components

    assert not components.flags.writeable
    with pytest.raises(ValueError):
        components[0, 0] = 99.0
    assert compressor.components[0, 0] != 99.0


def test_update_keeps_components_finite_and_row_normalized() -> None:
    compressor = OjaCompressor(input_dim=5, output_dim=2, learning_rate=0.03, seed=9)
    samples = (
        [1.0, 0.5, -0.25, 0.1, 0.3],
        [-0.5, 1.0, 0.25, -0.2, 0.4],
        [0.25, -0.5, 1.0, 0.3, -0.1],
        [0.1, 0.25, -0.3, 1.0, 0.2],
    )

    for _ in range(32):
        for features in samples:
            compressor.update(features)

    components = compressor.components
    row_norms = np.linalg.norm(components, axis=1)

    assert np.all(np.isfinite(components))
    assert np.allclose(row_norms, np.ones(2), atol=1e-6)


def test_snapshot_restore_roundtrip_preserves_transform() -> None:
    compressor = OjaCompressor(
        input_dim=5,
        output_dim=2,
        learning_rate=0.02,
        seed=13,
        dtype="float32",
    )
    compressor.update([1.0, -0.5, 0.25, 0.75, -0.1])
    compressor.update([-0.25, 1.0, 0.5, -0.5, 0.2])
    snapshot = compressor.snapshot()

    restored = OjaCompressor(
        input_dim=5,
        output_dim=2,
        learning_rate=0.02,
        seed=999,
        dtype="float32",
    )
    restored.restore(snapshot)

    features = [0.1, -0.2, 0.3, 0.4, -0.5]
    assert snapshot.name == OJA_COMPRESSOR_NAME
    assert snapshot.schema_version == OJA_COMPRESSOR_SNAPSHOT_SCHEMA_VERSION
    assert restored.samples_seen == compressor.samples_seen
    assert np.allclose(restored.components, compressor.components)
    assert np.allclose(restored.transform(features), compressor.transform(features))


def test_snapshot_to_dict_and_from_dict_roundtrip() -> None:
    compressor = OjaCompressor(input_dim=4, output_dim=2, seed=21)
    compressor.update([1.0, 0.5, -0.5, 0.25])

    restored_snapshot = OjaCompressorSnapshot.from_dict(compressor.snapshot().to_dict())
    restored = OjaCompressor(input_dim=4, output_dim=2, seed=99)
    restored.restore(restored_snapshot)

    assert np.allclose(restored.components, compressor.components)
    assert restored.samples_seen == compressor.samples_seen


@pytest.mark.parametrize(
    "snapshot",
    [
        _snapshot(name="wrong"),
        _snapshot(input_dim=5, components=_COMPONENTS_5X2),
        _snapshot(output_dim=3, components=_COMPONENTS_4X3),
        _snapshot(learning_rate=0.02),
        _snapshot(dtype="float32"),
        _snapshot(components=_COMPONENTS_BAD_SHAPE),
        _snapshot(samples_seen=-1),
    ],
)
def test_restore_rejects_incompatible_snapshots(snapshot: OjaCompressorSnapshot) -> None:
    compressor = OjaCompressor(input_dim=4, output_dim=2, seed=1)

    with pytest.raises(ValueError):
        compressor.restore(snapshot)


def test_restore_rejects_non_oja_snapshot() -> None:
    compressor = OjaCompressor(input_dim=4, output_dim=2, seed=1)

    with pytest.raises(TypeError):
        compressor.restore({})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_dim": 0, "output_dim": 1},
        {"input_dim": True, "output_dim": 1},
        {"input_dim": 4, "output_dim": 0},
        {"input_dim": 4, "output_dim": True},
        {"input_dim": 4, "output_dim": 4},
        {"input_dim": 4, "output_dim": 5},
        {"input_dim": 4, "output_dim": 2, "learning_rate": 0.0},
        {"input_dim": 4, "output_dim": 2, "learning_rate": float("inf")},
        {"input_dim": 4, "output_dim": 2, "seed": True},
        {"input_dim": 4, "output_dim": 2, "seed": 1.5},
        {"input_dim": 4, "output_dim": 2, "dtype": "int64"},
    ],
)
def test_constructor_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        OjaCompressor(**kwargs)  # type: ignore[arg-type]


def test_transform_and_update_validate_features() -> None:
    compressor = OjaCompressor(input_dim=4, output_dim=2, seed=1)

    with pytest.raises(ValueError):
        compressor.transform([1.0, 2.0])
    with pytest.raises(ValueError):
        compressor.transform([1.0, 2.0, 3.0, float("nan")])
    with pytest.raises(ValueError):
        compressor.update([1.0, 2.0])
    with pytest.raises(ValueError):
        compressor.step([1.0, 2.0, 3.0, float("inf")])


def test_oja_compressor_is_not_root_public_api() -> None:
    assert "OjaCompressor" not in adaptive_reservoir.__all__
    assert not hasattr(adaptive_reservoir, "OjaCompressor")


def _snapshot(
    *,
    schema_version: int = OJA_COMPRESSOR_SNAPSHOT_SCHEMA_VERSION,
    name: str = OJA_COMPRESSOR_NAME,
    input_dim: int = 4,
    output_dim: int = 2,
    learning_rate: float = 0.01,
    seed: int | None = 1,
    dtype: str = "float64",
    components: tuple[tuple[float, ...], ...] = _COMPONENTS_4X2,
    samples_seen: int = 0,
) -> OjaCompressorSnapshot:
    return OjaCompressorSnapshot(
        schema_version=schema_version,
        name=name,
        state={
            "input_dim": input_dim,
            "output_dim": output_dim,
            "learning_rate": learning_rate,
            "seed": seed,
            "dtype": dtype,
            "components": components,
            "samples_seen": samples_seen,
        },
    )
