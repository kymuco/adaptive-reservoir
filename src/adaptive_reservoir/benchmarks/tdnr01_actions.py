"""TDNR-01 Stage 4C: bounded orchestration of the frozen experiment.

Run one complete seed at a time, then reconstruct the original Stage 4A
partition artifacts and Stage 4B report. This module does not change the
scientific protocol, perform model selection, or execute work on import.

The CLI is intended for a separately approved execution environment with
an orchestration checkout and an independent checkout of frozen PR #65.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import time
from dataclasses import asdict
from numbers import Real
from pathlib import Path

import numpy as np

from adaptive_reservoir.benchmarks import (
    tdnr_experiment,
    tdnr_noise_sweep,
    tdnr_report,
)
from adaptive_reservoir.benchmarks.tdnr_compare import MODEL_NAMES, WINDOW_NAMES
from adaptive_reservoir.benchmarks.tdnr_experiment import (
    ARTIFACT_NAMES,
    FIXED_CONFIG,
    FIXED_RESERVOIR,
    PARTITION_SEEDS,
    SCORE_WINDOW,
    build_evidence,
    verify_evidence,
)
from adaptive_reservoir.benchmarks.tdnr_noise_sweep import (
    DEFAULT_NOISE_LEVELS,
    NOISE_AXES,
    PairedNoiseScore,
    run_tdnr_noise_sweep,
)
from adaptive_reservoir.benchmarks.tdnr_report import build_report

FROZEN_SHA = "fc4823b96e3c8c065952485a5765a01a67ce3001"
SHARD_SCHEMA = "tdnr-01.stage4c.shard.v1"
RUN_SCHEMA = "tdnr-01.stage4c.execution.v1"
ALL_SEEDS = PARTITION_SEEDS["development"] + PARTITION_SEEDS["holdout"]
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _encode(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _partition(seed: int) -> str:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    for name, seeds in PARTITION_SEEDS.items():
        if seed in seeds:
            return name
    raise ValueError("seed is outside the frozen partitions")


def _validate_commit(commit: str) -> None:
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("orchestrator commit must be a lowercase 40-character SHA")


def _git_output(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot verify Git checkout: {root}") from exc
    return result.stdout.strip()


def _verify_checkout(root: Path, expected_sha: str) -> None:
    if not root.is_dir():
        raise ValueError(f"checkout is missing: {root}")
    if _git_output(root, "rev-parse", "HEAD") != expected_sha:
        raise ValueError("checkout commit does not match approved SHA")
    if _git_output(root, "status", "--porcelain"):
        raise ValueError("checkout must be clean")


def _verify_sources(
    frozen_root: Path,
    orchestrator_root: Path,
    orchestrator_commit: str,
) -> None:
    _validate_commit(orchestrator_commit)
    _verify_checkout(frozen_root, FROZEN_SHA)
    _verify_checkout(orchestrator_root, orchestrator_commit)

    frozen_src = (frozen_root / "src").resolve()
    for module in (tdnr_experiment, tdnr_noise_sweep, tdnr_report):
        if not Path(module.__file__).resolve().is_relative_to(frozen_src):
            raise ValueError("experimental imports must resolve to frozen checkout")

    expected_script = (
        orchestrator_root
        / "src/adaptive_reservoir/benchmarks/tdnr01_actions.py"
    ).resolve()
    if Path(__file__).resolve() != expected_script:
        raise ValueError("orchestration script does not match checkout path")


def _run_identity() -> tuple[str, str]:
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if not run_id.isdecimal() or not attempt.isdecimal():
        raise ValueError("GitHub Actions run ID and attempt are required")
    if int(run_id) <= 0 or int(attempt) <= 0:
        raise ValueError("GitHub Actions run identity must be positive")
    return run_id, attempt


def _protocol_sha() -> str:
    return _digest(_encode(tdnr_experiment._protocol()))


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-10)


def _validate_rows(rows: tuple[PairedNoiseScore, ...], seed: int) -> None:
    expected = {
        (axis, level, seed, model, window)
        for axis in NOISE_AXES
        for level in DEFAULT_NOISE_LEVELS
        for model in MODEL_NAMES
        for window in WINDOW_NAMES
    }
    if any(not isinstance(row, PairedNoiseScore) for row in rows):
        raise ValueError("shard contains an invalid score record")
    actual = [
        (row.axis, row.level, row.seed, row.model, row.window)
        for row in rows
    ]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("incomplete, duplicated or unexpected seed score grid")

    references = {
        (row.axis, row.model, row.window): row
        for row in rows
        if row.level == 0.0
    }
    for row in rows:
        values = (
            row.mae,
            row.rmse,
            row.baseline_mae,
            row.baseline_rmse,
            row.delta_mae,
            row.delta_rmse,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("non-finite or invalid score value")
        if min(row.mae, row.rmse, row.baseline_mae, row.baseline_rmse) < 0.0:
            raise ValueError("negative absolute prediction error")
        reference = references[(row.axis, row.model, row.window)]
        if not _close(row.baseline_mae, reference.mae):
            raise ValueError("paired MAE reference mismatch")
        if not _close(row.baseline_rmse, reference.rmse):
            raise ValueError("paired RMSE reference mismatch")
        if not _close(row.delta_mae, row.mae - row.baseline_mae):
            raise ValueError("paired MAE delta mismatch")
        if not _close(row.delta_rmse, row.rmse - row.baseline_rmse):
            raise ValueError("paired RMSE delta mismatch")

    for model in MODEL_NAMES:
        for window in WINDOW_NAMES:
            left = references[(NOISE_AXES[0], model, window)]
            right = references[(NOISE_AXES[1], model, window)]
            if left.mae != right.mae or left.rmse != right.rmse:
                raise ValueError("zero-noise reference differs across axes")


def _stage_path(output_dir: Path) -> Path:
    return output_dir.with_name(output_dir.name + ".partial")


def _check_destination(output_dir: Path) -> Path:
    staging = _stage_path(output_dir)
    if output_dir.exists() or staging.exists():
        raise FileExistsError("output or interrupted staging directory exists")
    if not output_dir.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    return staging


def shard(
    *,
    seed: int,
    output_dir: Path,
    frozen_root: Path,
    orchestrator_root: Path,
    orchestrator_commit: str,
    max_seconds: int | None = None,
) -> None:
    partition = _partition(seed)
    _verify_sources(frozen_root, orchestrator_root, orchestrator_commit)
    run_id, attempt = _run_identity()
    staging = _check_destination(output_dir)
    if max_seconds is not None and (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, int)
        or max_seconds <= 0
    ):
        raise ValueError("max_seconds must be a positive integer")

    started = time.perf_counter()
    rows = run_tdnr_noise_sweep(
        FIXED_CONFIG,
        levels=DEFAULT_NOISE_LEVELS,
        axes=NOISE_AXES,
        seeds=(seed,),
        score_window=SCORE_WINDOW,
        reservoir_config=FIXED_RESERVOIR,
    )
    elapsed = time.perf_counter() - started
    if max_seconds is not None and elapsed > max_seconds:
        raise RuntimeError("seed execution exceeded approved runtime limit")
    _validate_rows(rows, seed)

    score_bytes = _encode({
        "schema": SHARD_SCHEMA,
        "seed": seed,
        "records": [asdict(row) for row in rows],
    })
    metadata = {
        "schema": SHARD_SCHEMA,
        "frozen_source_commit": FROZEN_SHA,
        "orchestrator_commit": orchestrator_commit,
        "run_id": run_id,
        "run_attempt": attempt,
        "partition": partition,
        "seed": seed,
        "protocol_sha256": _protocol_sha(),
        "score_sha256": _digest(score_bytes),
        "score_bytes": len(score_bytes),
        "record_count": len(rows),
        "elapsed_seconds": elapsed,
        "environment": _environment(),
    }
    staging.mkdir()
    _write_new(staging / "rows.json", score_bytes)
    _write_new(staging / "metadata.json", _encode(metadata))
    os.replace(staging, output_dir)
    print(f"seed={seed} partition={partition} rows={len(rows)} sha256={_digest(score_bytes)}")


def _read_shard(
    directory: Path,
    *,
    seed: int,
    orchestrator_commit: str,
    run_id: str,
    attempt: str,
) -> tuple[tuple[PairedNoiseScore, ...], dict[str, object], str]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"missing or invalid shard directory: {seed}")
    if {item.name for item in directory.iterdir()} != {"rows.json", "metadata.json"}:
        raise ValueError(f"incomplete or unexpected shard files: {seed}")
    metadata_bytes = (directory / "metadata.json").read_bytes()
    score_bytes = (directory / "rows.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    scores = json.loads(score_bytes)
    if _encode(metadata) != metadata_bytes or _encode(scores) != score_bytes:
        raise ValueError(f"noncanonical shard: {seed}")
    if not isinstance(metadata, dict) or not isinstance(scores, dict):
        raise ValueError(f"invalid shard structure: {seed}")

    expected = {
        "schema": SHARD_SCHEMA,
        "frozen_source_commit": FROZEN_SHA,
        "orchestrator_commit": orchestrator_commit,
        "run_id": run_id,
        "run_attempt": attempt,
        "partition": _partition(seed),
        "seed": seed,
        "protocol_sha256": _protocol_sha(),
        "score_sha256": _digest(score_bytes),
        "score_bytes": len(score_bytes),
        "record_count": (
            len(NOISE_AXES) * len(DEFAULT_NOISE_LEVELS)
            * len(MODEL_NAMES) * len(WINDOW_NAMES)
        ),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"shard provenance or integrity mismatch: {seed}")
    if scores.get("schema") != SHARD_SCHEMA or scores.get("seed") != seed:
        raise ValueError(f"shard score identity mismatch: {seed}")
    rows = tuple(PairedNoiseScore(**record) for record in scores["records"])
    _validate_rows(rows, seed)
    return rows, metadata, _digest(metadata_bytes)


def assemble(
    *,
    shards_dir: Path,
    output_dir: Path,
    frozen_root: Path,
    orchestrator_root: Path,
    orchestrator_commit: str,
) -> None:
    _verify_sources(frozen_root, orchestrator_root, orchestrator_commit)
    run_id, attempt = _run_identity()
    staging = _check_destination(output_dir)
    if not shards_dir.is_dir():
        raise ValueError("shards directory is missing")

    expected_names = {f"tdnr01-seed-{seed}" for seed in ALL_SEEDS}
    actual_names = {item.name for item in shards_dir.iterdir()}
    if actual_names != expected_names:
        raise ValueError("incomplete or unexpected shard set")

    partition_rows: dict[str, list[PairedNoiseScore]] = {
        "development": [],
        "holdout": [],
    }
    shard_receipts = []
    environments = set()
    for seed in ALL_SEEDS:
        rows, metadata, metadata_digest = _read_shard(
            shards_dir / f"tdnr01-seed-{seed}",
            seed=seed,
            orchestrator_commit=orchestrator_commit,
            run_id=run_id,
            attempt=attempt,
        )
        partition_rows[_partition(seed)].extend(rows)
        environment = metadata["environment"]
        environments.add((
            environment["python"],
            environment["numpy"],
            tuple(sorted(environment["threads"].items())),
        ))
        shard_receipts.append({
            "seed": seed,
            "metadata_sha256": metadata_digest,
            "score_sha256": metadata["score_sha256"],
        })
    if len(environments) != 1:
        raise ValueError("inconsistent shard execution environments")

    official = {
        partition: build_evidence(
            tuple(partition_rows[partition]),
            partition=partition,
            declared_source_commit=FROZEN_SHA,
        )
        for partition in PARTITION_SEEDS
    }

    staging.mkdir()
    for partition, artifacts in official.items():
        directory = staging / partition
        directory.mkdir()
        for name in (*ARTIFACT_NAMES, "manifest.json"):
            _write_new(directory / name, artifacts[name])
        manifest = verify_evidence(directory)
        if manifest["declared_source_commit"] != FROZEN_SHA:
            raise ValueError("official evidence source mismatch")

    report = build_report(staging / "development", staging / "holdout")
    _write_new(staging / "report.json", report)
    run_manifest = {
        "schema": RUN_SCHEMA,
        "status": "evidence_complete_independent_review_pending",
        "frozen_source_commit": FROZEN_SHA,
        "source_commit_independently_verified": False,
        "orchestrator_commit": orchestrator_commit,
        "run_id": run_id,
        "run_attempt": attempt,
        "protocol_sha256": _protocol_sha(),
        "partition_seeds": {
            name: list(seeds) for name, seeds in PARTITION_SEEDS.items()
        },
        "score_records": {
            name: len(rows) for name, rows in partition_rows.items()
        },
        "development_manifest_sha256": _digest(
            official["development"]["manifest.json"]
        ),
        "holdout_manifest_sha256": _digest(
            official["holdout"]["manifest.json"]
        ),
        "report_sha256": _digest(report),
        "shard_receipts": shard_receipts,
        "independent_scientific_review": "pending",
        "limitations": [
            "Automated consistency checks are not independent scientific review.",
            "Synthetic evidence does not establish real-world superiority.",
            "Unadjusted bootstrap intervals are descriptive.",
            "A declared Git commit and artifact hashes do not prove remote authenticity.",
        ],
    }
    _write_new(staging / "run_manifest.json", _encode(run_manifest))
    os.replace(staging, output_dir)
    print(f"evidence_complete report_sha256={_digest(report)} review=pending")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TDNR-01 frozen seed orchestration")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("shard", "assemble"):
        item = commands.add_parser(command)
        item.add_argument("--output-dir", type=Path, required=True)
        item.add_argument("--frozen-root", type=Path, required=True)
        item.add_argument("--orchestrator-root", type=Path, required=True)
        item.add_argument("--orchestrator-commit", required=True)
        if command == "shard":
            item.add_argument("--seed", type=int, required=True)
            item.add_argument("--max-seconds", type=int)
        else:
            item.add_argument("--shards-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    common = {
        "output_dir": args.output_dir,
        "frozen_root": args.frozen_root,
        "orchestrator_root": args.orchestrator_root,
        "orchestrator_commit": args.orchestrator_commit,
    }
    if args.command == "shard":
        shard(seed=args.seed, max_seconds=args.max_seconds, **common)
    else:
        assemble(shards_dir=args.shards_dir, **common)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
