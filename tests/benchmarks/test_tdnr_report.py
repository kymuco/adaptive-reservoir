"""TDNR-01 Stage 4B evidence audit tests; no full experiment runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from adaptive_reservoir.benchmarks import tdnr_report
from adaptive_reservoir.benchmarks.tdnr_compare import MODEL_NAMES, WINDOW_NAMES
from adaptive_reservoir.benchmarks.tdnr_experiment import (
    ARTIFACT_NAMES,
    build_evidence,
    verify_evidence,
)
from adaptive_reservoir.benchmarks.tdnr_noise_sweep import (
    DEFAULT_NOISE_LEVELS,
    NOISE_AXES,
    PairedNoiseScore,
)
from adaptive_reservoir.benchmarks.tdnr_statistics import (
    DEVELOPMENT_SEEDS,
    HOLDOUT_SEEDS,
)

SOURCE_COMMIT = "3f79d7a4928f7429f577f981aa1a6c2066c63745"


def _rows(seeds):
    rows = []
    for axis in NOISE_AXES:
        for level in DEFAULT_NOISE_LEVELS:
            for seed in seeds:
                for model in MODEL_NAMES:
                    for window in WINDOW_NAMES:
                        index = MODEL_NAMES.index(model)
                        base = 0.8 if model == "zero" else 1.0 + 0.1 * index
                        base += 0.02 * WINDOW_NAMES.index(window)
                        base += 0.01 * (seed % 17)
                        change = (
                            level * 0.03 * (index + 1) * (seed % 5 + 1)
                            * (2.0 if axis == "target" else 1.0)
                        )
                        mae = base + change
                        rmse = base + 0.5 + 1.5 * change
                        rows.append(
                            PairedNoiseScore(
                                axis=axis,
                                level=level,
                                seed=seed,
                                model=model,
                                window=window,
                                mae=mae,
                                rmse=rmse,
                                baseline_mae=base,
                                baseline_rmse=base + 0.5,
                                delta_mae=mae - base,
                                delta_rmse=rmse - (base + 0.5),
                            )
                        )
    return tuple(rows)


@pytest.fixture(scope="module")
def evidence_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("tdnr-report")
    directories = {}
    for partition, seeds in (
        ("development", DEVELOPMENT_SEEDS),
        ("holdout", HOLDOUT_SEEDS),
    ):
        directory = root / partition
        directory.mkdir()
        artifacts = build_evidence(
            _rows(seeds),
            partition=partition,
            declared_source_commit=SOURCE_COMMIT,
        )
        for name, content in artifacts.items():
            (directory / name).write_bytes(content)
        directories[partition] = directory
    return directories


def test_complete_report_is_deterministic(evidence_dirs) -> None:
    development = evidence_dirs["development"]
    holdout = evidence_dirs["holdout"]
    first = tdnr_report.build_report(development, holdout)
    second = tdnr_report.build_report(development, holdout)
    assert first == second

    report = json.loads(first)
    assert report["schema"] == tdnr_report.REPORT_SCHEMA
    assert report["source_commit_independently_verified"] is False
    assert report["interpretation"]["significance_claim"] == "none"
    assert report["declared_source_commit"] == SOURCE_COMMIT

    for partition, expected_scores in (
        ("development", 1440),
        ("holdout", 4320),
    ):
        result = report["results"][partition]
        assert result["score_record_count"] == expected_scores
        assert len(result["aggregates"]) == 144
        assert len(result["contrasts"]) == 108
        assert result["interval_category_counts"]["mae"][
            "baseline_favored"
        ] > 0
        assert result["interval_category_counts"]["mae"][
            "adaptive_favored"
        ] > 0
        assert result["interval_category_counts"]["noise_mae"][
            "interval_includes_zero"
        ] > 0


def test_all_negative_and_uncertain_results_are_retained(evidence_dirs) -> None:
    report = json.loads(
        tdnr_report.build_report(
            evidence_dirs["development"], evidence_dirs["holdout"]
        )
    )
    contrasts = report["results"]["holdout"]["contrasts"]
    assert len(contrasts) == 108
    zero = next(
        item for item in contrasts
        if item["contrast"]["baseline"] == "zero"
        and item["contrast"]["level"] == 0.0
        and item["contrast"]["window"] == "pre"
        and item["contrast"]["axis"] == "input"
    )
    assert zero["contrast"]["mean_mae_advantage"] < 0.0
    assert zero["interval_categories"]["mae"] == "baseline_favored"
    assert zero["interval_categories"]["noise_mae"] == "interval_includes_zero"


def test_swapped_or_identical_partitions_are_rejected(evidence_dirs) -> None:
    development = evidence_dirs["development"]
    holdout = evidence_dirs["holdout"]
    with pytest.raises(ValueError, match="must differ"):
        tdnr_report.build_report(development, development)
    with pytest.raises(ValueError, match="partition mismatch"):
        tdnr_report.build_report(holdout, development)


def test_rehashed_inconsistent_summary_is_rejected(
    evidence_dirs, tmp_path: Path
) -> None:
    original = evidence_dirs["development"]
    modified = tmp_path / "modified"
    modified.mkdir()
    for name in (*ARTIFACT_NAMES, "manifest.json"):
        (modified / name).write_bytes((original / name).read_bytes())

    name = "aggregates.json"
    payload = json.loads((modified / name).read_bytes())
    payload["records"][0]["mean_mae"] += 1.0
    changed = tdnr_report._encode(payload)
    (modified / name).write_bytes(changed)

    manifest = json.loads((modified / "manifest.json").read_bytes())
    manifest["artifacts"][name]["sha256"] = hashlib.sha256(changed).hexdigest()
    manifest["artifacts"][name]["bytes"] = len(changed)
    (modified / "manifest.json").write_bytes(tdnr_report._encode(manifest))

    assert verify_evidence(modified)["partition"] == "development"
    with pytest.raises(ValueError, match="recomputed evidence mismatch"):
        tdnr_report.build_report(modified, evidence_dirs["holdout"])


def test_missing_or_corrupt_manifest_is_rejected(
    evidence_dirs, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="manifest is missing"):
        tdnr_report.build_report(empty, evidence_dirs["holdout"])
