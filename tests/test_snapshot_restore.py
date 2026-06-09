from adaptive_reservoir import AdaptiveReservoir, ReservoirConfig


def test_snapshot_restore_smoke() -> None:
    model = AdaptiveReservoir(ReservoirConfig(input_dim=2, n_cells=4))
    model.step([0.5, -0.25])
    snapshot = model.snapshot()

    model.step([0.25, 0.75])
    model.restore(snapshot)

    assert model.samples_seen == 1
    assert snapshot["api_stage"] == "snapshot_restore_v1"
