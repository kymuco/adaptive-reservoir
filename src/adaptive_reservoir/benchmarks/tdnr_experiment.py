"""TDNR-01 Stage 4A: reproducible experiment and evidence artifacts.

The official protocol is fixed in source. Development and holdout are
executed separately, with no parameter selection or automatic tuning.

A manifest records the declared source commit, complete configurations,
software environment, artifact sizes and SHA-256 digests. Digests detect
accidental modification but do not authenticate the source commit.

This module performs no experiment or filesystem writes at import time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
from collections.abc import Iterable
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

import numpy as np

from adaptive_reservoir import ReadoutConfig, ReservoirConfig
from adaptive_reservoir.benchmarks.tdnr_compare import MODEL_NAMES, WINDOW_NAMES
from adaptive_reservoir.benchmarks.tdnr_contrasts import (
    summarize_baseline_contrasts,
)
from adaptive_reservoir.benchmarks.tdnr_noise_sweep import (
    DEFAULT_NOISE_LEVELS,
    NOISE_AXES,
    PairedNoiseScore,
    run_tdnr_noise_sweep,
)
from adaptive_reservoir.benchmarks.tdnr_statistics import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEVELOPMENT_SEEDS,
    HOLDOUT_SEEDS,
    summarize_paired_noise,
)
from adaptive_reservoir.benchmarks.temporal_drift_noise import (
    TemporalDriftNoiseConfig,
)

SCHEMA = "tdnr-01.stage4a.v1"
SCORE_WINDOW = 128
CONFIDENCE = 0.95
BOOTSTRAP_SEED = 0
PARTITION_SEEDS = {
    "development": DEVELOPMENT_SEEDS,
    "holdout": HOLDOUT_SEEDS,
}
ARTIFACT_NAMES = ("scores.json", "aggregates.json", "contrasts.json")

FIXED_CONFIG = TemporalDriftNoiseConfig(
    seed=0,
    n_samples=4000,
    drift_at=2000,
    delay_before=3,
    delay_after=12,
    input_noise=0.0,
    target_noise=0.0,
    scenario="abrupt",
    signal_kind="sinusoidal",
)
FIXED_RESERVOIR = ReservoirConfig(
    input_dim=2,
    n_cells=96,
    topology="ring_shortcuts",
    feature_mode="multi_raw",
    seed=0,
    dtype="float64",
    readout=ReadoutConfig(name="sliding_ridge", update_interval=1),
)


def _encode(value: object) -> bytes:
    """Encode canonical, finite JSON with a trailing newline."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def _validate_request(partition: str, declared_source_commit: str) -> tuple[int, ...]:
    if partition not in PARTITION_SEEDS:
        raise ValueError("partition must be development or holdout")
    if not isinstance(declared_source_commit, str) or re.fullmatch(
        r"[0-9a-f]{40}", declared_source_commit
    ) is None:
        raise ValueError("declared_source_commit must be a lowercase 40-digit SHA")
    return PARTITION_SEEDS[partition]


def _package_version() -> str:
    try:
        return metadata.version("adaptive-reservoir")
    except metadata.PackageNotFoundError:
        return "not-installed"


def _protocol() -> dict[str, object]:
    return {
        "generator": asdict(FIXED_CONFIG),
        "reservoir": FIXED_RESERVOIR.to_dict(),
        "reservoir_seed_policy": "replace configuration seed with dataset seed",
        "noise_axes": list(NOISE_AXES),
        "noise_levels": list(DEFAULT_NOISE_LEVELS),
        "models": list(MODEL_NAMES),
        "windows": list(WINDOW_NAMES),
        "score_window": SCORE_WINDOW,
        "bootstrap": {
            "confidence": CONFIDENCE,
            "replicates": DEFAULT_BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "unit": "whole paired dataset seed",
        },
        "training_targets": "noisy supervised targets",
        "evaluation_targets": "clean targets, offline only",
        "parameter_selection": "none",
    }


def build_evidence(
    rows: Iterable[PairedNoiseScore],
    *,
    partition: str,
    declared_source_commit: str,
) -> dict[str, bytes]:
    """Build complete evidence in memory without running models or writing files.

    Require the entire fixed grid. Missing rows, duplicate rows, omitted
    baselines, omitted noise levels and incomplete seed sets are rejected.
    Negative and inconclusive results are preserved without filtering.
    """
    seeds = _validate_request(partition, declared_source_commit)
    materialized = tuple(rows)
    if any(not isinstance(row, PairedNoiseScore) for row in materialized):
        raise ValueError("rows must contain PairedNoiseScore instances")

    expected = {
        (axis, level, seed, model, window)
        for axis in NOISE_AXES
        for level in DEFAULT_NOISE_LEVELS
        for seed in seeds
        for model in MODEL_NAMES
        for window in WINDOW_NAMES
    }
    actual = [
        (row.axis, row.level, row.seed, row.model, row.window)
        for row in materialized
    ]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("incomplete or duplicate fixed-protocol grid")

    ordered = tuple(
        sorted(
            materialized,
            key=lambda row: (
                NOISE_AXES.index(row.axis),
                row.level,
                row.seed,
                MODEL_NAMES.index(row.model),
                WINDOW_NAMES.index(row.window),
            ),
        )
    )
    aggregates = summarize_paired_noise(
        ordered,
        partition=partition,
        confidence=CONFIDENCE,
        bootstrap_replicates=DEFAULT_BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    contrasts = summarize_baseline_contrasts(
        ordered,
        partition=partition,
        confidence=CONFIDENCE,
        bootstrap_replicates=DEFAULT_BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )

    records = {
        "scores.json": [asdict(row) for row in ordered],
        "aggregates.json": [asdict(row) for row in aggregates],
        "contrasts.json": [asdict(row) for row in contrasts],
    }
    artifacts = {
        name: _encode({"schema": SCHEMA, "partition": partition, "records": data})
        for name, data in records.items()
    }
    descriptions = {
        name: {
            "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob),
            "records": len(records[name]),
        }
        for name, blob in artifacts.items()
    }

    manifest = {
        "schema": SCHEMA,
        "artifact_status": "complete",
        "partition": partition,
        "seeds": list(seeds),
        "declared_source_commit": declared_source_commit,
        "source_commit_verified": False,
        "protocol": _protocol(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "adaptive_reservoir": _package_version(),
        },
        "artifacts": descriptions,
        "interpretation": {
            "noise_delta": "noisy error minus zero-noise error",
            "baseline_advantage": "baseline error minus adaptive error",
            "noise_advantage": "baseline degradation minus adaptive degradation",
            "positive_advantage": "favors adaptive reservoir",
            "confidence_intervals": "descriptive, unadjusted percentile bootstrap",
        },
        "limitations": [
            "Synthetic data only; no real-world generalization claim.",
            "Bootstrap intervals do not establish statistical significance.",
            "No multiple-comparison correction or model selection.",
            "SHA-256 checks integrity, not authenticity or source provenance.",
            "Holdout is reported separately and must not be used for tuning.",
        ],
    }
    artifacts["manifest.json"] = _encode(manifest)
    return artifacts


def run_partition(
    partition: str,
    output_dir: str | Path,
    *,
    declared_source_commit: str,
) -> dict[str, bytes]:
    """Execute one complete fixed partition and persist evidence once.

    The destination parent must already exist. Existing destinations are
    never overwritten. The manifest is written last as the completion
    marker. An interrupted write can leave an incomplete directory,
    which verify_evidence rejects.
    """
    seeds = _validate_request(partition, declared_source_commit)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"evidence destination already exists: {destination}")

    rows = run_tdnr_noise_sweep(
        FIXED_CONFIG,
        levels=DEFAULT_NOISE_LEVELS,
        axes=NOISE_AXES,
        seeds=seeds,
        score_window=SCORE_WINDOW,
        reservoir_config=FIXED_RESERVOIR,
    )
    artifacts = build_evidence(
        rows,
        partition=partition,
        declared_source_commit=declared_source_commit,
    )

    destination.mkdir(exist_ok=False)
    for name in (*ARTIFACT_NAMES, "manifest.json"):
        with (destination / name).open("xb") as handle:
            handle.write(artifacts[name])
            handle.flush()
            os.fsync(handle.fileno())
    return artifacts


def verify_evidence(output_dir: str | Path) -> dict[str, object]:
    """Verify canonical artifact bytes, record counts and manifest digests.

    This checks internal consistency only. A modified manifest and modified
    data can be made mutually consistent; independent source verification
    remains necessary.
    """
    directory = Path(output_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("complete evidence manifest is missing")

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or _encode(manifest) != manifest_bytes:
        raise ValueError("manifest is not canonical JSON")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported evidence schema")
    if manifest.get("artifact_status") != "complete":
        raise ValueError("evidence is not complete")
    if set(manifest.get("artifacts", {})) != set(ARTIFACT_NAMES):
        raise ValueError("unexpected evidence artifact set")

    partition = manifest.get("partition")
    seeds = _validate_request(partition, manifest.get("declared_source_commit"))
    if manifest.get("seeds") != list(seeds):
        raise ValueError("manifest seed partition mismatch")
    if manifest.get("protocol") != _protocol():
        raise ValueError("manifest protocol mismatch")

    expected_counts = {
        "scores.json": len(NOISE_AXES) * len(DEFAULT_NOISE_LEVELS)
        * len(seeds) * len(MODEL_NAMES) * len(WINDOW_NAMES),
        "aggregates.json": len(NOISE_AXES) * len(DEFAULT_NOISE_LEVELS)
        * len(MODEL_NAMES) * len(WINDOW_NAMES),
        "contrasts.json": len(NOISE_AXES) * len(DEFAULT_NOISE_LEVELS)
        * (len(MODEL_NAMES) - 1) * len(WINDOW_NAMES),
    }
    for name in ARTIFACT_NAMES:
        data = (directory / name).read_bytes()
        description = manifest["artifacts"][name]
        if len(data) != description["bytes"]:
            raise ValueError(f"artifact size mismatch: {name}")
        if hashlib.sha256(data).hexdigest() != description["sha256"]:
            raise ValueError(f"artifact SHA-256 mismatch: {name}")
        payload = json.loads(data)
        if _encode(payload) != data:
            raise ValueError(f"noncanonical artifact: {name}")
        if payload.get("schema") != SCHEMA or payload.get("partition") != partition:
            raise ValueError(f"artifact identity mismatch: {name}")
        if len(payload["records"]) != expected_counts[name]:
            raise ValueError(f"incomplete artifact records: {name}")
        if description["records"] != expected_counts[name]:
            raise ValueError(f"manifest record count mismatch: {name}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TDNR-01 fixed-protocol experiment")
    parser.add_argument("--partition", choices=tuple(PARTITION_SEEDS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args(argv)

    artifacts = run_partition(
        args.partition,
        args.output_dir,
        declared_source_commit=args.source_commit,
    )
    print(hashlib.sha256(artifacts["manifest.json"]).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
