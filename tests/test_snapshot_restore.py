from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from adaptive_reservoir import AdaptiveReservoir, ReservoirConfig, ReservoirSnapshot, ReservoirState


def test_snapshot_captures_numeric_runtime_state() -> None:
    model = AdaptiveReservoir(_config())

    result = model.step([0.5, -0.25])
    snapshot = model.snapshot()

    assert snapshot.schema_version == 1
    assert snapshot.state.samples_seen == result.metrics.samples_seen
    np.testing.assert_allclose(snapshot.state.activations, result.state.activations)
    np.testing.assert_allclose(snapshot.state.fast_trace, result.state.fast_trace)
    np.testing.assert_allclose(snapshot.state.mid_trace, result.state.mid_trace)
    np.testing.assert_allclose(snapshot.state.slow_trace, result.state.slow_trace)


def test_snapshot_is_independent_from_future_model_steps() -> None:
    model = AdaptiveReservoir(_config())

    model.step([0.5, -0.25])
    snapshot = model.snapshot()
    captured_state = snapshot.state

    model.step([0.25, 0.75])
    model.step([-1.0, 0.5])

    np.testing.assert_allclose(snapshot.state.activations, captured_state.activations)
    np.testing.assert_allclose(snapshot.state.fast_trace, captured_state.fast_trace)
    np.testing.assert_allclose(snapshot.state.mid_trace, captured_state.mid_trace)
    np.testing.assert_allclose(snapshot.state.slow_trace, captured_state.slow_trace)
    assert snapshot.state.samples_seen == captured_state.samples_seen


def test_restore_rewinds_model_state_and_continuation_is_deterministic() -> None:
    model = AdaptiveReservoir(_config())

    model.step([0.5, -0.25])
    snapshot = model.snapshot()

    expected = model.step([0.25, 0.75])

    model.step([-1.0, 0.5])
    model.step([0.0, 1.0])

    model.restore(snapshot)
    actual = model.step([0.25, 0.75])

    assert actual.metrics.samples_seen == expected.metrics.samples_seen
    np.testing.assert_allclose(actual.state.activations, expected.state.activations)
    np.testing.assert_allclose(actual.state.fast_trace, expected.state.fast_trace)
    np.testing.assert_allclose(actual.state.mid_trace, expected.state.mid_trace)
    np.testing.assert_allclose(actual.state.slow_trace, expected.state.slow_trace)


def test_reset_matches_fresh_model_behavior() -> None:
    config = _config()
    model = AdaptiveReservoir(config)
    fresh = AdaptiveReservoir(config)

    model.step([0.5, -0.25])
    model.step([0.25, 0.75])
    model.reset()

    reset_result = model.step([0.5, -0.25])
    fresh_result = fresh.step([0.5, -0.25])

    assert reset_result.metrics.samples_seen == fresh_result.metrics.samples_seen
    np.testing.assert_allclose(reset_result.state.activations, fresh_result.state.activations)
    np.testing.assert_allclose(reset_result.state.fast_trace, fresh_result.state.fast_trace)
    np.testing.assert_allclose(reset_result.state.mid_trace, fresh_result.state.mid_trace)
    np.testing.assert_allclose(reset_result.state.slow_trace, fresh_result.state.slow_trace)


def test_restore_rejects_wrong_snapshot_type() -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises(TypeError, match="ReservoirSnapshot"):
        model.restore({"schema_version": 1, "state": None})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("snapshot", "message"),
    [
        (
            ReservoirSnapshot(
                state=ReservoirState(
                    activations=np.zeros(3, dtype=np.float64),
                    fast_trace=np.zeros(3, dtype=np.float64),
                    mid_trace=np.zeros(3, dtype=np.float64),
                    slow_trace=np.zeros(3, dtype=np.float64),
                ),
            ),
            "shape",
        ),
        (
            ReservoirSnapshot(
                state=ReservoirState(
                    activations=np.zeros(4, dtype=np.float32),
                    fast_trace=np.zeros(4, dtype=np.float32),
                    mid_trace=np.zeros(4, dtype=np.float32),
                    slow_trace=np.zeros(4, dtype=np.float32),
                ),
            ),
            "dtype",
        ),
        (
            ReservoirSnapshot(
                state=ReservoirState(
                    activations=np.zeros(4, dtype=np.float64),
                    fast_trace=np.zeros(4, dtype=np.float64),
                    mid_trace=np.zeros(4, dtype=np.float64),
                    slow_trace=np.zeros(4, dtype=np.float64),
                ),
                schema_version=999,
            ),
            "schema_version",
        ),
    ],
)
def test_restore_rejects_incompatible_snapshot(snapshot: ReservoirSnapshot, message: str) -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises((TypeError, ValueError), match=message):
        model.restore(snapshot)


def test_snapshot_contains_no_semantic_or_domain_fields() -> None:
    model = AdaptiveReservoir(_config())
    snapshot = model.snapshot()

    assert {field.name for field in dataclasses.fields(snapshot)} == {
        "state",
        "schema_version",
    }
    forbidden_names = {
        "raw_inputs",
        "targets",
        "input_history",
        "target_history",
        "user_data",
        "semantic_labels",
        "domain_events",
        "conversation_data",
        "policy_decisions",
        "host_system_metadata",
        "hde_data",
        "config",
    }
    for name in forbidden_names:
        assert not hasattr(snapshot, name)


def _config() -> ReservoirConfig:
    return ReservoirConfig(
        input_dim=2,
        n_cells=4,
        topology="ring_shortcuts",
        seed=42,
        feature_mode="state_slow_raw",
    )
