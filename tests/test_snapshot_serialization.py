from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from adaptive_reservoir import (
    AdaptiveReservoir,
    AdaptiveReservoirMetricsSnapshot,
    ChannelCalculatorSnapshot,
    ReadoutConfig,
    ReservoirConfig,
    ReservoirSnapshot,
    ReservoirState,
)
from adaptive_reservoir.readout import ReadoutSnapshot


def test_reservoir_state_dict_roundtrip() -> None:
    state = ReservoirState.zeros(n_cells=3, dtype="float32")

    restored = ReservoirState.from_dict(json.loads(json.dumps(state.to_dict())))

    assert restored.activations.dtype == np.float32
    assert restored.samples_seen == state.samples_seen
    np.testing.assert_allclose(restored.activations, state.activations)
    np.testing.assert_allclose(restored.fast_trace, state.fast_trace)
    np.testing.assert_allclose(restored.mid_trace, state.mid_trace)
    np.testing.assert_allclose(restored.slow_trace, state.slow_trace)


def test_readout_snapshot_dict_roundtrip() -> None:
    snapshot = ReadoutSnapshot(
        schema_version=1,
        name="smoke",
        state={"weights": (1.0, 2.0), "bias": np.float64(0.5)},
    )

    data = snapshot.to_dict()
    restored = ReadoutSnapshot.from_dict(json.loads(json.dumps(data)))

    assert data == {
        "schema_version": 1,
        "name": "smoke",
        "state": {"weights": [1.0, 2.0], "bias": 0.5},
    }
    assert restored.schema_version == snapshot.schema_version
    assert restored.name == snapshot.name
    assert restored.state == {"weights": [1.0, 2.0], "bias": 0.5}


def test_channel_snapshot_dict_roundtrip() -> None:
    snapshot = ChannelCalculatorSnapshot(
        samples_seen=2,
        feature_dim=2,
        feature_window=((1.0, 2.0),),
        activation_window=((0.1, 0.2),),
        state_delta_window=(0.3,),
        prediction_window=(0.4,),
        prediction_error_window=(0.5,),
        previous_activations=(0.1, 0.2),
    )

    data = snapshot.to_dict()
    restored = ChannelCalculatorSnapshot.from_dict(json.loads(json.dumps(data)))

    assert data["feature_window"] == [[1.0, 2.0]]
    assert restored == snapshot


def test_metrics_snapshot_dict_roundtrip() -> None:
    snapshot = AdaptiveReservoirMetricsSnapshot(
        samples_seen=2,
        us_per_sample_avg=10.0,
        readout_update_count=1,
        readout_solve_count=1,
        saturation_rate_avg=0.25,
    )

    restored = AdaptiveReservoirMetricsSnapshot.from_dict(
        json.loads(json.dumps(snapshot.to_dict()))
    )

    assert restored == snapshot


def test_reservoir_snapshot_dict_roundtrip() -> None:
    config = _config(dtype="float32")
    model = AdaptiveReservoir(config)
    model.step([0.1, -0.2], target=1.0)
    snapshot = model.snapshot()

    data = snapshot.to_dict()
    restored = ReservoirSnapshot.from_dict(json.loads(json.dumps(data)))

    assert restored.schema_version == snapshot.schema_version
    assert restored.state.activations.dtype == np.float32
    assert restored.readout == snapshot.readout
    assert restored.channels == snapshot.channels
    assert restored.metrics == snapshot.metrics


def test_reservoir_snapshot_dict_is_json_serializable() -> None:
    model = AdaptiveReservoir(_config())
    model.step([0.1, -0.2], target=1.0)

    payload = json.loads(json.dumps(model.snapshot().to_dict()))
    restored = ReservoirSnapshot.from_dict(payload)

    assert isinstance(restored, ReservoirSnapshot)


def test_restore_after_snapshot_dict_roundtrip_is_deterministic() -> None:
    config = _config(readout=ReadoutConfig(name="sliding_ridge", update_interval=2))
    model = AdaptiveReservoir(config)
    model.step([0.1, -0.2], target=1.0)
    model.step([0.2, -0.1], target=2.0)
    payload = json.loads(json.dumps(model.snapshot().to_dict()))
    restored_snapshot = ReservoirSnapshot.from_dict(payload)

    copy = AdaptiveReservoir(config)
    copy.restore(restored_snapshot)

    assert copy.metrics_snapshot() == model.metrics_snapshot()
    expected = model.step([0.3, -0.4]).prediction
    actual = copy.step([0.3, -0.4]).prediction
    assert actual == pytest.approx(expected)


def test_snapshot_from_dict_rejects_bad_schema_version() -> None:
    model = AdaptiveReservoir(_config())
    data = model.snapshot().to_dict()
    data["schema_version"] = 999

    with pytest.raises(ValueError, match="schema_version"):
        ReservoirSnapshot.from_dict(data)


def test_snapshot_from_dict_rejects_missing_state() -> None:
    model = AdaptiveReservoir(_config())
    data = model.snapshot().to_dict()
    del data["state"]

    with pytest.raises(ValueError, match="state"):
        ReservoirSnapshot.from_dict(data)


def test_snapshot_to_dict_contains_no_numpy_or_tuple_values() -> None:
    model = AdaptiveReservoir(_config(dtype="float32"))
    model.step([0.1, -0.2], target=1.0)

    data = model.snapshot().to_dict()

    _assert_json_friendly(data)


def test_snapshot_to_dict_contains_no_config_or_semantic_fields() -> None:
    model = AdaptiveReservoir(_config())
    data = model.snapshot().to_dict()

    assert "config" not in data
    forbidden = {
        "raw_inputs",
        "targets",
        "input_history",
        "target_history",
        "user_data",
        "semantic_labels",
        "domain_events",
        "conversation_data",
        "policy_decisions",
        "hde_data",
    }
    flattened_keys = set(_walk_keys(data))
    assert forbidden.isdisjoint(flattened_keys)


def _config(
    *,
    dtype: str = "float64",
    readout: ReadoutConfig | None = None,
) -> ReservoirConfig:
    return ReservoirConfig(
        input_dim=2,
        n_cells=4,
        topology="ring_shortcuts",
        seed=42,
        dtype=dtype,  # type: ignore[arg-type]
        readout=readout or ReadoutConfig(name="sliding_ridge"),
    )


def _assert_json_friendly(value: object) -> None:
    if value is None or isinstance(value, str | int | float | bool):
        return
    assert not isinstance(value, tuple)
    assert not isinstance(value, np.ndarray)
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert isinstance(key, str)
            _assert_json_friendly(item)
        return
    if isinstance(value, Sequence):
        for item in value:
            _assert_json_friendly(item)
        return
    raise AssertionError(f"unexpected non-json-friendly value: {type(value)!r}")


def _walk_keys(value: object) -> list[str]:
    if isinstance(value, Mapping):
        keys: list[str] = []
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(item))
        return keys
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        keys = []
        for item in value:
            keys.extend(_walk_keys(item))
        return keys
    return []
