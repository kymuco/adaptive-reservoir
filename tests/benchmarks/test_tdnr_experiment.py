"""TDNR-01 Stage 4A fixed-protocol and evidence integrity tests.

Tests use synthetic score fixtures. They do not execute the full study.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from adaptive_reservoir.benchmarks import tdnr_experiment as experiment
from adaptive_reservoir.benchmarks.tdnr_compare import MODEL_NAMES, WINDOW_NAMES
from adaptive_reservoir.benchmarks.tdnr_noise_sweep import (
    DEFAULT_NOISE_LEVELS,
    NOISE_AXES,
    PairedNoiseScore,
)
from adaptive_reservoir.benchmarks.tdnr_statistics import (
    DEVELOPMENT_SEEDS,
    HOLDOUT_SEEDS,
)

COMMIT = "48aa695b4f4cb68f080727fc2c02db2314a932a3"


def _fake_rows(seeds):
    rows = []
    for axis in NOISE_AXES:
        for level in DEFAULT_NOISE_LEVELS:
            for seed in seeds:
                for model in MODEL_NAMES:
                    for window in WINDOW_NAMES:
                        model_index = MODEL_NAMES.index(model)
                        window_index = WINDOW_NAMES.index(window)
                        base_mae = (
                            1.0 + 0.1 * model_index + 0.02 * window_index
                            + 0.01 * (seed % 17)
                        )
                        base_rmse = base_mae + 0.5
                        axis_factor = 2.0 if axis == "target" else 1.0
                        change = (
                            level * 0.03 * (model_index + 1)
                            * (seed % 5 + 1) * axis_factor
                        )
                        mae = base_mae + change
                        rmse = base_rmse + 1.5 * change
                        rows.append(
                            PairedNoiseScore(
                                axis=axis,
                                level=level,
                                seed=seed,
                                model=model,
                                window=window,
                                mae=mae,
                                rmse=rmse,
                                baseline_mae=base_mae,
                                baseline_rmse=base_rmse,
                                delta_mae=mae - base_mae,
                                delta_rmse=rmse - base_rmse,
                            )
                        )
    return tuple(rows)


def _build(rows, partition="development"):
    return experiment.build_evidence(
        rows,
        partition=partition,
        declared_source_commit=COMMIT,
    )


def test_protocol_is_fixed_and_partitions_are_disjoint() -> None:
    assert tuple(range(10)) == DEVELOPMENT_SEEDS
    assert tuple(range(100, 130)) == HOLDOUT_SEEDS
    assert not set(DEVELOPMENT_SEEDS).intersection(HOLDOUT_SEEDS)
    assert experiment.SCORE_WINDOW == 128
    assert experiment.CONFIDENCE == 0.95
    assert experiment.FIXED_CONFIG.input_noise == 0.0
    assert experiment.FIXED_CONFIG.target_noise == 0.0
    assert experiment.FIXED_RESERVOIR.n_cells == 96


def test_complete_development_bundle_is_deterministic() -> None:
    rows = _fake_rows(DEVELOPMENT_SEEDS)
    first = _build(rows)
    second = _build(reversed(rows))
    assert first == second
    assert set(first) == {*experiment.ARTIFACT_NAMES, "manifest.json"}

    manifest = json.loads(first["manifest.json"])
    assert manifest["partition"] == "development"
    assert manifest["seeds"] == list(DEVELOPMENT_SEEDS)
    assert manifest["source_commit_verified"] is False
    assert manifest["declared_source_commit"] == COMMIT
    assert manifest["protocol"]["parameter_selection"] == "none"

    expected_counts = {
        "scores.json": 1440,
        "aggregates.json": 144,
        "contrasts.json": 108,
    }
    for name, count in expected_counts.items():
        payload = json.loads(first[name])
        assert count == len(payload["records"])
        assert count == manifest["artifacts"][name]["records"]
        assert hashlib.sha256(first[name]).hexdigest() == (
            manifest["artifacts"][name]["sha256"]
        )


def test_holdout_has_its_own_seed_partition() -> None:
    artifacts = _build(_fake_rows(HOLDOUT_SEEDS), partition="holdout")
    manifest = json.loads(artifacts["manifest.json"])
    assert manifest["partition"] == "holdout"
    assert manifest["seeds"] == list(HOLDOUT_SEEDS)
    assert 4320 == len(json.loads(artifacts["scores.json"])["records"])
    assert 144 == len(json.loads(artifacts["aggregates.json"])["records"])
    assert 108 == len(json.loads(artifacts["contrasts.json"])["records"])


def test_incomplete_grid_is_rejected() -> None:
    rows = _fake_rows(DEVELOPMENT_SEEDS)
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        _build(rows[:-1])
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        _build(rows + (rows[0],))
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        _build(rows, partition="holdout")


def test_invalid_request_is_rejected() -> None:
    rows = _fake_rows(DEVELOPMENT_SEEDS)
    with pytest.raises(ValueError, match="partition"):
        experiment.build_evidence(
            rows,
            partition="exploratory",
            declared_source_commit=COMMIT,
        )
    with pytest.raises(ValueError, match="declared_source_commit"):
        experiment.build_evidence(
            rows,
            partition="development",
            declared_source_commit="unverified",
        )


def test_run_partition_uses_fixed_arguments_and_writes_manifest_last(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []

    def fake_sweep(config, *, levels, axes, seeds, score_window, reservoir_config):
        calls.append((config, levels, axes, seeds, score_window, reservoir_config))
        return _fake_rows(seeds)

    monkeypatch.setattr(experiment, "run_tdnr_noise_sweep", fake_sweep)
    output = tmp_path / "development-evidence"
    artifacts = experiment.run_partition(
        "development", output, declared_source_commit=COMMIT
    )

    assert calls == [
        (
            experiment.FIXED_CONFIG,
            DEFAULT_NOISE_LEVELS,
            NOISE_AXES,
            DEVELOPMENT_SEEDS,
            experiment.SCORE_WINDOW,
            experiment.FIXED_RESERVOIR,
        )
    ]
    assert set(path.name for path in output.iterdir()) == set(artifacts)
    assert (output / "manifest.json").read_bytes() == artifacts["manifest.json"]
    assert experiment.verify_evidence(output)["partition"] == "development"

    with pytest.raises(FileExistsError):
        experiment.run_partition(
            "development", output, declared_source_commit=COMMIT
        )
    assert len(calls) == 1


def test_corrupted_artifact_is_rejected(monkeypatch, tmp_path: Path) -> None:
    def fake_sweep(config, *, levels, axes, seeds, score_window, reservoir_config):
        return _fake_rows(seeds)

    monkeypatch.setattr(experiment, "run_tdnr_noise_sweep", fake_sweep)
    output = tmp_path / "evidence"
    experiment.run_partition("development", output, declared_source_commit=COMMIT)

    score_path = output / "scores.json"
    original = score_path.read_bytes()
    score_path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="size mismatch"):
        experiment.verify_evidence(output)


def test_missing_manifest_is_not_complete(tmp_path: Path) -> None:
    output = tmp_path / "incomplete"
    output.mkdir()
    (output / "scores.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest is missing"):
        experiment.verify_evidence(output)
