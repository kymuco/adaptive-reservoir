from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from adaptive_reservoir.readout import (
    READOUT_SNAPSHOT_SCHEMA_VERSION,
    ReadoutProtocol,
    ReadoutSnapshot,
    validate_features,
    validate_snapshot_mapping,
    validate_target,
)


class DummyReadout:
    def predict(self, features: np.ndarray) -> float:
        return float(features[0])

    def update(self, features: np.ndarray, target: float) -> None:
        _ = (features, target)

    def snapshot(self) -> Mapping[str, object]:
        return {"schema_version": 1, "weights": [1.0]}

    def restore(self, snapshot: Mapping[str, object]) -> None:
        _ = snapshot


def test_readout_protocol_accepts_structural_implementation() -> None:
    assert isinstance(DummyReadout(), ReadoutProtocol)


def test_readout_snapshot_is_immutable() -> None:
    snapshot = ReadoutSnapshot(
        schema_version=READOUT_SNAPSHOT_SCHEMA_VERSION,
        name="dummy",
        state={"weights": [1.0], "bias": 0.0},
    )

    with pytest.raises(FrozenInstanceError):
        snapshot.name = "other"  # type: ignore[misc]


def test_readout_snapshot_validates_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version must be a positive integer"):
        ReadoutSnapshot(schema_version=0, name="dummy", state={})

    with pytest.raises(ValueError, match="schema_version must be a positive integer"):
        ReadoutSnapshot(schema_version=True, name="dummy", state={})  # type: ignore[arg-type]


def test_readout_snapshot_validates_name() -> None:
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        ReadoutSnapshot(schema_version=1, name="", state={})


def test_readout_snapshot_validates_state_mapping() -> None:
    with pytest.raises(ValueError, match="snapshot must be a mapping"):
        ReadoutSnapshot(schema_version=1, name="dummy", state=[])  # type: ignore[arg-type]


def test_readout_snapshot_boundary_fields_are_numeric_state_only() -> None:
    snapshot = ReadoutSnapshot(
        schema_version=READOUT_SNAPSHOT_SCHEMA_VERSION,
        name="dummy",
        state={"weights": [0.1, 0.2], "bias": 0.0, "samples_seen": 2},
    )

    assert set(snapshot.__dataclass_fields__) == {"schema_version", "name", "state"}
    assert "raw_inputs" not in snapshot.state
    assert "messages" not in snapshot.state
    assert "policy_decision" not in snapshot.state


def test_validate_features_accepts_1d_float_vector() -> None:
    features = validate_features([1, 2.5, -3], expected_dim=3)

    np.testing.assert_allclose(features, np.array([1.0, 2.5, -3.0]))
    assert features.dtype == np.float64
    assert features.flags.writeable is False


def test_validate_features_accepts_numpy_array_copy() -> None:
    source = np.array([1.0, 2.0], dtype=np.float32)

    features = validate_features(source, expected_dim=2, dtype="float32")

    assert features.dtype == np.float32
    assert features.flags.writeable is False
    assert features is not source


def test_validate_features_rejects_string_input() -> None:
    with pytest.raises(ValueError, match="features must be a 1D numeric vector"):
        validate_features("1.0")


def test_validate_features_rejects_non_numeric_values() -> None:
    with pytest.raises(ValueError, match="features must contain only numeric values"):
        validate_features([1.0, object()])


def test_validate_features_rejects_2d_vector() -> None:
    with pytest.raises(ValueError, match="features must be a 1D numeric vector"):
        validate_features([[1.0, 2.0]])


def test_validate_features_rejects_empty_vector() -> None:
    with pytest.raises(ValueError, match="features must not be empty"):
        validate_features([])


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_validate_features_rejects_non_finite_values(bad_value: float) -> None:
    with pytest.raises(ValueError, match="features must contain only finite values"):
        validate_features([bad_value])


def test_validate_features_checks_expected_dim() -> None:
    with pytest.raises(ValueError, match="expected feature_dim=3, got 2"):
        validate_features([1.0, 2.0], expected_dim=3)


@pytest.mark.parametrize("bad_dim", [0, -1, True])
def test_validate_features_rejects_invalid_expected_dim(bad_dim: int) -> None:
    with pytest.raises(ValueError, match="expected_dim must be a positive integer"):
        validate_features([1.0], expected_dim=bad_dim)


def test_validate_target_accepts_finite_scalar() -> None:
    assert validate_target(1) == 1.0
    assert validate_target(-0.25) == -0.25


def test_validate_target_rejects_string() -> None:
    with pytest.raises(ValueError, match="target must be numeric"):
        validate_target("1.0")


@pytest.mark.parametrize("bad_target", [float("nan"), float("inf"), -float("inf")])
def test_validate_target_rejects_non_finite_values(bad_target: float) -> None:
    with pytest.raises(ValueError, match="target must be finite"):
        validate_target(bad_target)


def test_validate_snapshot_mapping_accepts_mapping() -> None:
    snapshot = {"weights": [1.0]}

    assert validate_snapshot_mapping(snapshot) is snapshot


def test_validate_snapshot_mapping_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="snapshot must be a mapping"):
        validate_snapshot_mapping([])
