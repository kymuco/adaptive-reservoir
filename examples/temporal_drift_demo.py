"""Temporal drift demo for adaptive-reservoir.

This example uses a deterministic synthetic numeric stream only. It does not use
real user data, semantic data, or product integration hooks.
"""

from __future__ import annotations

from adaptive_reservoir.benchmarks import (
    format_markdown_report,
    run_temporal_drift_benchmark,
)


def main() -> int:
    """Run a small temporal drift demo and print metrics."""

    result = run_temporal_drift_benchmark(
        seed=0,
        n_samples=360,
        drift_at=180,
        delay_before=3,
        delay_after=12,
        score_window=48,
    )

    print("Temporal drift demo")
    print("===================")
    print()
    print("This demo runs a deterministic synthetic temporal drift stream.")
    print("stream -> prediction -> metrics")
    print()
    print(f"benchmark: {result.benchmark}")
    print(f"samples_seen: {result.samples_seen}")
    print(f"pre_score: {result.pre_score:.3f}")
    print(f"post_score: {result.post_score:.3f}")
    print(f"final_score: {result.final_score:.3f}")
    print(f"adapt_steps: {_format_optional_int(result.adapt_steps)}")
    print(f"us_per_sample: {result.us_per_sample:.3f}")
    print()
    print("Markdown report:")
    print()
    print(format_markdown_report([result]))
    return 0


def _format_optional_int(value: int | None) -> str:
    if value is None:
        return "none"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
