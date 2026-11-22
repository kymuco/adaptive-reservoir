"""TDNR-01 Stage 4B: audit and report frozen experimental evidence.

This module never runs models or selects configurations. It independently
recomputes statistics from saved per-seed scores, retains every result,
and explicitly distinguishes descriptive intervals from significance.
No work occurs at import time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from adaptive_reservoir.benchmarks.tdnr_experiment import (
    ARTIFACT_NAMES,
    build_evidence,
    verify_evidence,
)
from adaptive_reservoir.benchmarks.tdnr_noise_sweep import PairedNoiseScore

REPORT_SCHEMA = "tdnr-01.stage4b.report.v1"
METRICS = {
    "mae": ("mae_ci_low", "mae_ci_high"),
    "rmse": ("rmse_ci_low", "rmse_ci_high"),
    "noise_mae": ("noise_mae_ci_low", "noise_mae_ci_high"),
    "noise_rmse": ("noise_rmse_ci_low", "noise_rmse_ci_high"),
}
CATEGORIES = (
    "adaptive_favored",
    "baseline_favored",
    "interval_includes_zero",
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


def _load_partition(
    directory: Path,
    expected_partition: str,
) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    """Check artifact hashes and independently reproduce every summary."""
    manifest = verify_evidence(directory)
    if manifest["partition"] != expected_partition:
        raise ValueError("evidence partition mismatch")

    payloads = {
        name: json.loads((directory / name).read_bytes())
        for name in ARTIFACT_NAMES
    }
    rows = tuple(
        PairedNoiseScore(**record)
        for record in payloads["scores.json"]["records"]
    )
    regenerated = build_evidence(
        rows,
        partition=expected_partition,
        declared_source_commit=manifest["declared_source_commit"],
    )
    for name in ARTIFACT_NAMES:
        if regenerated[name] != (directory / name).read_bytes():
            raise ValueError(f"recomputed evidence mismatch: {name}")

    records = {
        name: payloads[name]["records"]
        for name in ARTIFACT_NAMES
    }
    return manifest, records


def _interval_category(low: float, high: float) -> str:
    if low > 0.0:
        return "adaptive_favored"
    if high < 0.0:
        return "baseline_favored"
    return "interval_includes_zero"


def _annotate_contrasts(
    contrasts: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    annotated = []
    counts = {metric: Counter() for metric in METRICS}
    for contrast in contrasts:
        categories = {}
        for metric, (low_key, high_key) in METRICS.items():
            category = _interval_category(
                contrast[low_key], contrast[high_key]
            )
            categories[metric] = category
            counts[metric][category] += 1
        annotated.append({"contrast": contrast, "interval_categories": categories})

    totals = {
        metric: {category: counts[metric][category] for category in CATEGORIES}
        for metric in METRICS
    }
    return annotated, totals


def build_report(
    development_dir: str | Path,
    holdout_dir: str | Path,
) -> bytes:
    """Build a deterministic complete report from two verified partitions.

    The report includes all aggregates and all baseline contrasts, even
    when results favor baselines or intervals overlap zero. Classification
    is descriptive only, not a hypothesis test or model selection rule.
    """
    development = Path(development_dir)
    holdout = Path(holdout_dir)
    if development.resolve() == holdout.resolve():
        raise ValueError("development and holdout directories must differ")

    manifests = {}
    results = {}
    for partition, directory in (
        ("development", development),
        ("holdout", holdout),
    ):
        manifest, records = _load_partition(directory, partition)
        manifests[partition] = manifest
        annotated, counts = _annotate_contrasts(records["contrasts.json"])
        results[partition] = {
            "environment": manifest["environment"],
            "seeds": manifest["seeds"],
            "input_manifest_sha256": hashlib.sha256(
                (directory / "manifest.json").read_bytes()
            ).hexdigest(),
            "input_artifacts": manifest["artifacts"],
            "score_record_count": len(records["scores.json"]),
            "aggregates": records["aggregates.json"],
            "contrasts": annotated,
            "interval_category_counts": counts,
        }

    dev_manifest = manifests["development"]
    holdout_manifest = manifests["holdout"]
    if (
        dev_manifest["declared_source_commit"]
        != holdout_manifest["declared_source_commit"]
    ):
        raise ValueError("development and holdout source commits differ")
    if dev_manifest["protocol"] != holdout_manifest["protocol"]:
        raise ValueError("development and holdout protocols differ")

    report = {
        "schema": REPORT_SCHEMA,
        "status": "descriptive_evidence_pending_independent_review",
        "declared_source_commit": dev_manifest["declared_source_commit"],
        "source_commit_independently_verified": False,
        "protocol": dev_manifest["protocol"],
        "results": results,
        "interpretation": {
            "advantage_sign": "positive favors adaptive; negative favors baseline",
            "interval_categories": "descriptive percentile-bootstrap interval relative to zero",
            "interval_includes_zero": "no directional interval separation",
            "significance_claim": "none",
            "multiple_comparisons_adjusted": False,
            "holdout_used_for_tuning": False,
            "selection_performed": False,
        },
        "limitations": [
            "All results are synthetic and conditional on the fixed protocol.",
            "Bootstrap intervals are descriptive, not significance tests.",
            "No correction for multiple comparisons was performed.",
            "No real-world performance claim is established.",
            "Source commit authenticity requires independent verification.",
            "Independent scientific review remains necessary.",
        ],
    }
    return _encode(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TDNR-01 complete evidence report")
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_report(args.development_dir, args.holdout_dir)
    with args.output.open("xb") as handle:
        handle.write(report)
        handle.flush()
        os.fsync(handle.fileno())
    print(hashlib.sha256(report).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
