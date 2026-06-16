from __future__ import annotations

import dataclasses

import pytest

from adaptive_reservoir import (
    AdaptiveReservoir,
    AdaptiveReservoirMetricsSnapshot,
    ReadoutConfig,
    ReservoirConfig,
)


def test_metrics_snapshot_starts_zero() -> None:
    model = AdaptiveReservoir(_config())

    metrics = model.metrics_snapshot()

    assert metrics == AdaptiveReservoirMetricsSnapshot(samples_seen=0)


def test_metrics_snapshot_updates_after_steps() -> None:
    model = AdaptiveReservoir(_config())

    model.step([0.1, -0.1])
    model.step([0.2, -0.2], target=1.0)
    metrics = model.metrics_snapshot()

    assert metrics.samples_seen == 2
    assert metrics.us_per_sample_avg >= 0.0
    assert metrics.readout_update_count == 1
    assert 0.0 <= metrics.saturation_rate_avg <= 1.0


def test_metrics_snapshot_counts_readout_updates_only_with_target() -> None:
    model = AdaptiveReservoir(_config())

    model.step([0.1, -0.1])
    model.step([0.1, -0.1], target=1.0)
    model.step([0.1, -0.1], target=2.0)

    metrics = model.metrics_snapshot()

    assert metrics.samples_seen == 3
    assert metrics.readout_update_count == 2


def test_metrics_snapshot_reports_sliding_ridge_solve_count() -> None:
    model = AdaptiveReservoir(
        _config(readout=ReadoutConfig(name="sliding_ridge", update_interval=2))
    )

    model.step([0.1, -0.1], target=1.0)
    first = model.metrics_snapshot()
    model.step([0.2, -0.2], target=1.0)
    second = model.metrics_snapshot()

    assert first.readout_update_count == 1
    assert first.readout_solve_count == 0
    assert second.readout_update_count == 2
    assert second.readout_solve_count == 1


def test_metrics_snapshot_does_not_change_after_predict() -> None:
    model = AdaptiveReservoir(_config())
    model.step([0.1, -0.1], target=1.0)
    before = model.metrics_snapshot()

    model.predict()
    model.predict([0.2, -0.2])
    after = model.metrics_snapshot()

    assert after == before


def test_reset_clears_metrics_snapshot() -> None:
    model = AdaptiveReservoir(_config())
    model.step([0.1, -0.1], target=1.0)

    model.reset()

    assert model.metrics_snapshot() == AdaptiveReservoirMetricsSnapshot(samples_seen=0)


def test_snapshot_restore_preserves_metrics_snapshot() -> None:
    config = _config(readout=ReadoutConfig(name="sliding_ridge", update_interval=2))
    model = AdaptiveReservoir(config)
    model.step([0.1, -0.1], target=1.0)
    model.step([0.2, -0.2], target=2.0)
    snapshot = model.snapshot()

    restored = AdaptiveReservoir(config)
    restored.restore(snapshot)

    assert restored.metrics_snapshot() == model.metrics_snapshot()


def test_restore_rejects_metrics_sample_mismatch() -> None:
    model = AdaptiveReservoir(_config())
    snapshot = model.snapshot()
    bad_snapshot = dataclasses.replace(
        snapshot,
        metrics=dataclasses.replace(snapshot.metrics, samples_seen=1),
    )

    with pytest.raises(ValueError, match="metrics samples_seen"):
        model.restore(bad_snapshot)


def test_metrics_snapshot_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="readout_update_count"):
        AdaptiveReservoirMetricsSnapshot(samples_seen=1, readout_update_count=2)
    with pytest.raises(ValueError, match="saturation_rate_avg"):
        AdaptiveReservoirMetricsSnapshot(samples_seen=1, saturation_rate_avg=2.0)


def _config(readout: ReadoutConfig | None = None) -> ReservoirConfig:
    return ReservoirConfig(
        input_dim=2,
        n_cells=4,
        topology="ring_shortcuts",
        seed=42,
        readout=readout or ReadoutConfig(name="sliding_ridge"),
    )
