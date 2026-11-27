"""Stage 4C orchestration tests using synthetic records, never model runs."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from adaptive_reservoir.benchmarks import tdnr01_actions as actions
from adaptive_reservoir.benchmarks.tdnr_compare import MODEL_NAMES, WINDOW_NAMES
from adaptive_reservoir.benchmarks.tdnr_experiment import (
    ARTIFACT_NAMES,
    PARTITION_SEEDS,
    build_evidence,
    verify_evidence,
)
from adaptive_reservoir.benchmarks.tdnr_noise_sweep import (
    DEFAULT_NOISE_LEVELS,
    NOISE_AXES,
    PairedNoiseScore,
)

ORCHESTRATOR_SHA = "a" * 40


def _fake_rows(seed: int) -> tuple[PairedNoiseScore, ...]:
    result = []
    for axis in NOISE_AXES:
        for level in DEFAULT_NOISE_LEVELS:
            for index, model in enumerate(MODEL_NAMES):
                for window in WINDOW_NAMES:
                    base = 0.8 + 0.1 * index + 0.01 * (seed % 5)
                    base_rmse = base + 0.2
                    multiplier = 2 if axis == "target" else 1
                    change = level * 0.001 * (index + 1) * multiplier * (seed % 7 + 1)
                    mae = base + change
                    rmse = base_rmse + 2.0 * change
                    result.append(PairedNoiseScore(
                        axis=axis,
                        level=level,
                        seed=seed,
                        model=model,
                        window=window,
                        mae=mae,
                        rmse=rmse,
                        baseline_mae=base,
                        baseline_rmse=base_rmse,
                        delta_mae=mae - base,
                        delta_rmse=rmse - base_rmse,
                    ))
    return tuple(result)


def _fake_sweep(
    config, *, levels, axes, seeds, score_window, reservoir_config
):
    assert config == actions.FIXED_CONFIG
    assert reservoir_config == actions.FIXED_RESERVOIR
    assert levels == DEFAULT_NOISE_LEVELS
    assert axes == NOISE_AXES
    assert score_window == actions.SCORE_WINDOW
    return tuple(row for seed in seeds for row in _fake_rows(seed))


def _setup(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "123456")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setattr(actions, "_verify_sources", lambda *args: None)
    monkeypatch.setattr(actions, "run_tdnr_noise_sweep", _fake_sweep)


def _shard(tmp_path, seed):
    directory = tmp_path / "shards" / f"tdnr01-seed-{seed}"
    actions.shard(
        seed=seed,
        output_dir=directory,
        frozen_root=tmp_path,
        orchestrator_root=tmp_path,
        orchestrator_commit=ORCHESTRATOR_SHA,
    )
    return directory


def _assemble(tmp_path):
    output = tmp_path / "evidence"
    actions.assemble(
        shards_dir=tmp_path / "shards",
        output_dir=output,
        frozen_root=tmp_path,
        orchestrator_root=tmp_path,
        orchestrator_commit=ORCHESTRATOR_SHA,
    )
    return output


def test_partition_boundaries_and_grid_validation():
    assert actions._partition(0) == "development"
    assert actions._partition(9) == "development"
    assert actions._partition(100) == "holdout"
    assert actions._partition(129) == "holdout"
    for seed in (-1, 10, 99, 130):
        with pytest.raises(ValueError, match="outside"):
            actions._partition(seed)
    rows = _fake_rows(0)
    assert len(rows) == 144
    actions._validate_rows(rows, 0)
    with pytest.raises(ValueError, match="incomplete"):
        actions._validate_rows(rows[:-1], 0)
    with pytest.raises(ValueError, match="incomplete"):
        actions._validate_rows(rows + rows[:1], 0)


def test_checkout_identity_rejects_wrong_commit(monkeypatch, tmp_path):
    def fake_git(arguments, **kwargs):
        assert kwargs["check"] is True
        assert kwargs["capture_output"] is True
        if "rev-parse" in arguments:
            return SimpleNamespace(stdout="b" * 40 + "\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(actions.subprocess, "run", fake_git)
    with pytest.raises(ValueError, match="commit"):
        actions._verify_checkout(tmp_path, actions.FROZEN_SHA)


def test_shard_provenance_and_corruption(monkeypatch, tmp_path):
    _setup(monkeypatch)
    (tmp_path / "shards").mkdir()
    directory = _shard(tmp_path, 100)
    rows, metadata, digest = actions._read_shard(
        directory,
        seed=100,
        orchestrator_commit=ORCHESTRATOR_SHA,
        run_id="123456",
        attempt="1",
    )
    assert rows == _fake_rows(100)
    assert metadata["frozen_source_commit"] == actions.FROZEN_SHA
    assert len(digest) == 64
    with pytest.raises(ValueError, match="provenance"):
        actions._read_shard(
            directory,
            seed=100,
            orchestrator_commit=ORCHESTRATOR_SHA,
            run_id="123456",
            attempt="2",
        )
    with pytest.raises(ValueError, match="provenance"):
        actions._read_shard(
            directory,
            seed=100,
            orchestrator_commit="b" * 40,
            run_id="123456",
            attempt="1",
        )
    (directory / "rows.json").write_bytes(
        (directory / "rows.json").read_bytes() + b" "
    )
    with pytest.raises(ValueError, match="noncanonical"):
        actions._read_shard(
            directory,
            seed=100,
            orchestrator_commit=ORCHESTRATOR_SHA,
            run_id="123456",
            attempt="1",
        )


def test_failed_seed_can_retry_only_into_fresh_destination(monkeypatch, tmp_path):
    _setup(monkeypatch)
    (tmp_path / "shards").mkdir()
    original = actions.run_tdnr_noise_sweep

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(actions, "run_tdnr_noise_sweep", fail)
    with pytest.raises(RuntimeError, match="interruption"):
        _shard(tmp_path, 0)
    assert not (tmp_path / "shards/tdnr01-seed-0").exists()
    assert not (tmp_path / "shards/tdnr01-seed-0.partial").exists()

    monkeypatch.setattr(actions, "run_tdnr_noise_sweep", original)
    directory = _shard(tmp_path, 0)
    assert directory.is_dir()
    with pytest.raises(FileExistsError):
        _shard(tmp_path, 0)

    interrupted = tmp_path / "shards/tdnr01-seed-1.partial"
    interrupted.mkdir()
    with pytest.raises(FileExistsError):
        _shard(tmp_path, 1)
    assert not (tmp_path / "shards/tdnr01-seed-1").exists()


def test_sharded_assembly_matches_unsharded_frozen_evidence(
    monkeypatch, tmp_path
):
    _setup(monkeypatch)
    (tmp_path / "shards").mkdir()
    for seed in actions.ALL_SEEDS:
        _shard(tmp_path, seed)

    missing = tmp_path / "shards/tdnr01-seed-129"
    withheld = tmp_path / "withheld-seed-129"
    missing.rename(withheld)
    with pytest.raises(ValueError, match="shard set"):
        _assemble(tmp_path)
    assert not (tmp_path / "evidence").exists()
    assert not (tmp_path / "evidence.partial").exists()

    withheld.rename(missing)
    output = _assemble(tmp_path)
    assert output.is_dir()
    for partition, seeds in PARTITION_SEEDS.items():
        expected = build_evidence(
            _fake_sweep(
                actions.FIXED_CONFIG,
                levels=DEFAULT_NOISE_LEVELS,
                axes=NOISE_AXES,
                seeds=seeds,
                score_window=actions.SCORE_WINDOW,
                reservoir_config=actions.FIXED_RESERVOIR,
            ),
            partition=partition,
            declared_source_commit=actions.FROZEN_SHA,
        )
        for name in (*ARTIFACT_NAMES, "manifest.json"):
            assert (output / partition / name).read_bytes() == expected[name]
        assert verify_evidence(output / partition)["partition"] == partition

    report = json.loads((output / "report.json").read_bytes())
    assert "development" in report["results"]
    assert "holdout" in report["results"]
    manifest = json.loads((output / "run_manifest.json").read_bytes())
    assert manifest["status"] == "evidence_complete_independent_review_pending"
    assert manifest["score_records"] == {
        "development": 1440,
        "holdout": 4320,
    }
    assert len(manifest["shard_receipts"]) == 40
    assert manifest["source_commit_independently_verified"] is False
    assert manifest["independent_scientific_review"] == "pending"
    with pytest.raises(FileExistsError):
        _assemble(tmp_path)
