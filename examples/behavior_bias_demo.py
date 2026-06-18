"""Behavior bias demo for adaptive-reservoir.

This demo uses deterministic synthetic numeric interaction events only. It does
not infer real user state, does not process text content, and does not integrate
with Character_OS, HDE, or any host application.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from adaptive_reservoir import AdaptiveReservoir, ReadoutConfig, ReservoirConfig

EVENT_NAMES = (
    "message_length",
    "pause_seconds",
    "interruptions",
    "error_count",
    "time_pressure",
)

CHANNELS = (
    "initiative_bias",
    "interrupt_risk",
    "confidence",
    "drift_pressure",
)

CHECKPOINTS = (60, 120, 180, 240)
N_STEPS = 240


@dataclass(frozen=True, slots=True)
class BehaviorSnapshot:
    """One checkpoint of behavior-bias channel predictions."""

    step: int
    phase: str
    initiative_bias: float
    interrupt_risk: float
    confidence: float
    drift_pressure: float


def main() -> int:
    """Run the behavior bias demo and print checkpoint summaries."""

    snapshots = run_behavior_bias_demo()

    print("Behavior bias demo")
    print("==================")
    print()
    print("This demo uses synthetic numeric interaction events only.")
    print("It does not infer real user state and does not integrate with Character_OS.")
    print("events -> adaptive channels -> host decision hints")
    print()
    print("Synthetic input event fields:")
    for event_name in EVENT_NAMES:
        print(f"- {event_name}")
    print()
    print("Adaptive output channels:")
    for channel in CHANNELS:
        print(f"- {channel}")
    print()
    print(_format_snapshot_table(snapshots))
    print()
    print("Channel meanings:")
    print("initiative_bias: numeric hint for proactive assistance")
    print("interrupt_risk: numeric hint for avoiding interruption")
    print("confidence: numeric stability hint")
    print("drift_pressure: numeric stream-shift hint")
    return 0


def run_behavior_bias_demo() -> tuple[BehaviorSnapshot, ...]:
    """Run synthetic behavior events through one model per adaptive channel."""

    models = _make_channel_models()
    snapshots: list[BehaviorSnapshot] = []
    for step in range(1, N_STEPS + 1):
        phase, event = synthetic_behavior_event(step)
        targets = behavior_targets(event, phase=phase)
        predictions: dict[str, float] = {}
        for channel, model in models.items():
            result = model.step(event, target=targets[channel])
            predictions[channel] = _clip01(result.prediction)
        if step in CHECKPOINTS:
            snapshots.append(_snapshot_from_predictions(step, phase, predictions))
    return tuple(snapshots)


def synthetic_behavior_event(step: int) -> tuple[str, tuple[float, ...]]:
    """Return a deterministic synthetic interaction event for a demo step."""

    phase = phase_for_step(step)
    oscillation = math.sin(step * 0.173)
    secondary = math.sin(step * 0.071 + 0.5)
    if phase == "calm":
        values = (
            0.48 + 0.08 * oscillation,
            0.68 + 0.08 * secondary,
            0.10 + 0.04 * abs(oscillation),
            0.08 + 0.03 * abs(secondary),
            0.18 + 0.05 * abs(oscillation),
        )
    elif phase == "pressure":
        values = (
            0.34 + 0.12 * oscillation,
            0.22 + 0.07 * secondary,
            0.68 + 0.10 * abs(oscillation),
            0.52 + 0.12 * abs(secondary),
            0.82 + 0.08 * abs(oscillation),
        )
    else:
        values = (
            0.46 + 0.08 * oscillation,
            0.52 + 0.08 * secondary,
            0.32 + 0.08 * abs(oscillation),
            0.24 + 0.08 * abs(secondary),
            0.42 + 0.08 * abs(oscillation),
        )
    return phase, tuple(_clip01(value) for value in values)


def behavior_targets(event: tuple[float, ...], *, phase: str) -> dict[str, float]:
    """Return synthetic target values for behavior-bias adaptive channels."""

    _validate_event(event)
    message_length, pause_seconds, interruptions, error_count, time_pressure = event
    phase_boost = _phase_boost(phase)
    initiative_bias = _clip01(
        0.15 * message_length
        + 0.35 * pause_seconds
        + 0.20 * (1.0 - interruptions)
        + 0.15 * (1.0 - error_count)
        + 0.15 * (1.0 - time_pressure)
    )
    interrupt_risk = _clip01(
        0.35 * interruptions
        + 0.30 * time_pressure
        + 0.20 * error_count
        + 0.15 * (1.0 - pause_seconds)
    )
    confidence = _clip01(
        0.35 * (1.0 - error_count)
        + 0.25 * (1.0 - interruptions)
        + 0.20 * pause_seconds
        + 0.20 * (1.0 - time_pressure)
    )
    drift_pressure = _clip01(
        0.40 * time_pressure
        + 0.25 * interruptions
        + 0.20 * error_count
        + phase_boost
    )
    return {
        "initiative_bias": initiative_bias,
        "interrupt_risk": interrupt_risk,
        "confidence": confidence,
        "drift_pressure": drift_pressure,
    }


def phase_for_step(step: int) -> str:
    """Return the synthetic behavior phase for a step."""

    if step <= 80:
        return "calm"
    if step <= 160:
        return "pressure"
    return "recovery"


def _make_channel_models() -> dict[str, AdaptiveReservoir]:
    return {
        "initiative_bias": _make_channel_model(seed=11),
        "interrupt_risk": _make_channel_model(seed=12),
        "confidence": _make_channel_model(seed=13),
        "drift_pressure": _make_channel_model(seed=14),
    }


def _make_channel_model(*, seed: int) -> AdaptiveReservoir:
    return AdaptiveReservoir(
        ReservoirConfig(
            input_dim=len(EVENT_NAMES),
            n_cells=32,
            topology="ring_shortcuts",
            feature_mode="multi_raw",
            seed=seed,
            readout=ReadoutConfig(name="sliding_ridge", update_interval=1),
        )
    )


def _snapshot_from_predictions(
    step: int,
    phase: str,
    predictions: dict[str, float],
) -> BehaviorSnapshot:
    return BehaviorSnapshot(
        step=step,
        phase=phase,
        initiative_bias=predictions["initiative_bias"],
        interrupt_risk=predictions["interrupt_risk"],
        confidence=predictions["confidence"],
        drift_pressure=predictions["drift_pressure"],
    )


def _format_snapshot_table(snapshots: tuple[BehaviorSnapshot, ...]) -> str:
    lines = [
        "step  phase     initiative_bias  interrupt_risk  confidence  drift_pressure",
    ]
    for snapshot in snapshots:
        lines.append(
            f"{snapshot.step:03d}   "
            f"{snapshot.phase:<8}  "
            f"{snapshot.initiative_bias:.3f}            "
            f"{snapshot.interrupt_risk:.3f}           "
            f"{snapshot.confidence:.3f}       "
            f"{snapshot.drift_pressure:.3f}"
        )
    return "\n".join(lines)


def _phase_boost(phase: str) -> float:
    if phase == "calm":
        return 0.0
    if phase == "pressure":
        return 0.20
    if phase == "recovery":
        return 0.08
    msg = f"unknown phase: {phase}"
    raise ValueError(msg)


def _validate_event(event: tuple[float, ...]) -> None:
    if len(event) != len(EVENT_NAMES):
        msg = f"event must contain {len(EVENT_NAMES)} values"
        raise ValueError(msg)
    if any(value < 0.0 or value > 1.0 for value in event):
        msg = "event values must be bounded in [0, 1]"
        raise ValueError(msg)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


if __name__ == "__main__":
    raise SystemExit(main())
