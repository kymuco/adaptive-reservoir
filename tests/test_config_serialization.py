from __future__ import annotations

import json

import pytest

from adaptive_reservoir import ChannelConfig, ReadoutConfig, ReservoirConfig, TraceConfig


def test_trace_config_dict_roundtrip() -> None:
    config = TraceConfig(fast_decay=0.25, mid_decay=0.75, slow_decay=0.95)

    restored = TraceConfig.from_dict(config.to_dict())

    assert restored == config


def test_readout_config_dict_roundtrip() -> None:
    config = ReadoutConfig(
        name="replay_ridge",
        learning_rate=0.1,
        ridge_alpha=0.01,
        buffer_size=32,
        window_size=16,
        update_interval=3,
    )

    restored = ReadoutConfig.from_dict(config.to_dict())

    assert restored == config


def test_channel_config_dict_roundtrip() -> None:
    config = ChannelConfig(
        novelty_window=8,
        stability_window=16,
        drift_window=24,
        saturation_threshold=0.9,
        epsilon=1e-6,
    )

    restored = ChannelConfig.from_dict(config.to_dict())

    assert restored == config


def test_reservoir_config_dict_roundtrip() -> None:
    config = ReservoirConfig(
        input_dim=3,
        n_cells=8,
        topology="ring_shortcuts",
        feature_mode="multi_raw",
        seed=123,
        dtype="float32",
        leak_rate=0.5,
        input_scale=0.75,
        recurrent_scale=0.25,
        fatigue_rate=0.1,
        trace=TraceConfig(fast_decay=0.2, mid_decay=0.8, slow_decay=0.98),
        readout=ReadoutConfig(name="replay_ridge", update_interval=2),
        channels=ChannelConfig(novelty_window=4, stability_window=5, drift_window=6),
    )

    restored = ReservoirConfig.from_dict(config.to_dict())

    assert restored == config


def test_reservoir_config_dict_is_json_serializable() -> None:
    config = ReservoirConfig(input_dim=2, dtype="float32")

    payload = json.loads(json.dumps(config.to_dict()))
    restored = ReservoirConfig.from_dict(payload)

    assert restored == config


def test_reservoir_config_from_minimal_dict_uses_defaults() -> None:
    restored = ReservoirConfig.from_dict({"schema_version": 1, "input_dim": 2})

    assert restored == ReservoirConfig(input_dim=2)


def test_reservoir_config_from_dict_rejects_bad_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        ReservoirConfig.from_dict({"schema_version": 999, "input_dim": 2})


def test_reservoir_config_from_dict_rejects_bad_nested_config() -> None:
    with pytest.raises(ValueError, match="trace"):
        ReservoirConfig.from_dict({"schema_version": 1, "input_dim": 2, "trace": "bad"})


def test_reservoir_config_from_dict_runs_constructor_validation() -> None:
    data = ReservoirConfig(input_dim=2).to_dict()
    data["leak_rate"] = 2.0

    with pytest.raises(ValueError, match="leak_rate"):
        ReservoirConfig.from_dict(data)


def test_reservoir_config_to_dict_contains_expected_schema() -> None:
    data = ReservoirConfig(input_dim=2).to_dict()

    assert data["schema_version"] == 1
    assert set(data) == {
        "channels",
        "dtype",
        "fatigue_rate",
        "feature_mode",
        "input_dim",
        "input_scale",
        "leak_rate",
        "n_cells",
        "readout",
        "recurrent_scale",
        "schema_version",
        "seed",
        "topology",
        "trace",
    }
