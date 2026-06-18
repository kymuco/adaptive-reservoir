"""Presence state demo for adaptive-reservoir.

This demo uses deterministic synthetic numeric desktop/system-style events only.
It does not read real desktop activity, does not infer real user state, and does
not integrate with Character_OS, HDE, or any host application.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from adaptive_reservoir import AdaptiveReservoir, ReadoutConfig, ReservoirConfig

EVENT_NAMES = (
    "idle_time",
    "window_switch_rate",
    "typing_burst",
    "failed_action_count",
    "notification_density",
)

CHANNELS = (
    "should_wait",
    "should_notify",
    "attention_state",
)

CHECKPOINTS = (60, 120, 180, 240)
N_STEPS = 240


@dataclass(frozen=True, slots=True)
class PresenceSnapshot:
    """One checkpoint of presence-channel predictions."""

    step: int
    phase: str
    should_wait: float
    should_notify: float
    attention_state: float


def main() -> int:
    """Run the presence state demo and print checkpoint summaries."""

    snapshots = run_presence_state_demo()

    print("Presence state demo")
    print("===================")
    print()
    print("This demo uses synthetic numeric desktop/system events only.")
    print("It does not read real desktop activity and does not infer real user state.")
    print("events -> presence channels -> host decision hints")
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
    print("should_wait: numeric hint to delay interruption")
    print("should_notify: numeric hint that notification may be acceptable")
    print("attention_state: synthetic operational attention channel")
    return 0


def run_presence_state_demo() -> tuple[PresenceSnapshot, ...]:
    """Run synthetic presence events through one model per adaptive channel."""

    models = _make_channel_models()
    snapshots: list[PresenceSnapshot] = []
    for step in range(1, N_STEPS + 1):
        phase, event = synthetic_presence_event(step)
        targets = presence_targets(event, phase=phase)
        predictions: dict[str, float] = {}
        for channel, model in models.items():
            result = model.step(event, target=targets[channel])
            predictions[channel] = _clip01(result.prediction)
        if step in CHECKPOINTS:
            snapshots.append(_snapshot_from_predictions(step, phase, predictions))
    return tuple(snapshots)


def synthetic_presence_event(step: int) -> tuple[str, tuple[float, ...]]:
    """Return a deterministic synthetic desktop/system-style event."""

    phase = phase_for_step(step)
    oscillation = math.sin(step * 0.139)
    secondary = math.sin(step * 0.083 + 0.35)
    if phase == "focused":
        values = (
            0.12 + 0.04 * abs(secondary),
            0.28 + 0.08 * abs(oscillation),
            0.78 + 0.10 * abs(secondary),
            0.10 + 0.04 * abs(oscillation),
            0.24 + 0.06 * abs(secondary),
        )
    elif phase == "overloaded":
        values = (
            0.08 + 0.04 * abs(secondary),
            0.78 + 0.10 * abs(oscillation),
            0.58 + 0.12 * abs(secondary),
            0.62 + 0.12 * abs(oscillation),
            0.76 + 0.12 * abs(secondary),
        )
    else:
        values = (
            0.58 + 0.12 * abs(secondary),
            0.18 + 0.06 * abs(oscillation),
            0.32 + 0.08 * abs(secondary),
            0.12 + 0.04 * abs(oscillation),
            0.18 + 0.06 * abs(secondary),
        )
    return phase, tuple(_clip01(value) for value in values)


def presence_targets(event: tuple[float, ...], *, phase: str) -> dict[str, float]:
    """Return synthetic target values for presence adaptive channels."""

    _validate_event(event)
    _validate_phase(phase)
    idle_time, window_switch_rate, typing_burst, failed_action_count, notification_density = event
    should_wait = _clip01(
        0.30 * typing_burst
        + 0.25 * window_switch_rate
        + 0.20 * failed_action_count
        + 0.15 * notification_density
        + 0.10 * (1.0 - idle_time)
    )
    should_notify = _clip01(
        0.35 * idle_time
        + 0.25 * (1.0 - typing_burst)
        + 0.20 * (1.0 - notification_density)
        + 0.10 * (1.0 - failed_action_count)
        + 0.10 * (1.0 - window_switch_rate)
    )
    attention_state = _clip01(
        0.35 * typing_burst
        + 0.25 * (1.0 - idle_time)
        + 0.20 * (1.0 - failed_action_count)
        + 0.10 * (1.0 - notification_density)
        + 0.10 * (1.0 - window_switch_rate)
    )
    return {
        "should_wait": should_wait,
        "should_notify": should_notify,
        "attention_state": attention_state,
    }


def phase_for_step(step: int) -> str:
    """Return the synthetic presence phase for a step."""

    if step <= 80:
        return "focused"
    if step <= 160:
        return "overloaded"
    return "available"


def _make_channel_models() -> dict[str, AdaptiveReservoir]:
    return {
        "should_wait": _make_channel_model(seed=21),
        "should_notify": _make_channel_model(seed=22),
        "attention_state": _make_channel_model(seed=23),
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
) -> PresenceSnapshot:
    return PresenceSnapshot(
        step=step,
        phase=phase,
        should_wait=predictions["should_wait"],
        should_notify=predictions["should_notify"],
        attention_state=predictions["attention_state"],
    )


def _format_snapshot_table(snapshots: tuple[PresenceSnapshot, ...]) -> str:
    lines = [
        "step  phase       should_wait  should_notify  attention_state",
    ]
    for snapshot in snapshots:
        lines.append(
            f"{snapshot.step:03d}   "
            f"{snapshot.phase:<10}  "
            f"{snapshot.should_wait:.3f}        "
            f"{snapshot.should_notify:.3f}          "
            f"{snapshot.attention_state:.3f}"
        )
    return "\n".join(lines)


def _validate_event(event: tuple[float, ...]) -> None:
    if len(event) != len(EVENT_NAMES):
        msg = f"event must contain {len(EVENT_NAMES)} values"
        raise ValueError(msg)
    if any(value < 0.0 or value > 1.0 for value in event):
        msg = "event values must be bounded in [0, 1]"
        raise ValueError(msg)


def _validate_phase(phase: str) -> None:
    if phase not in {"focused", "overloaded", "available"}:
        msg = f"unknown phase: {phase}"
        raise ValueError(msg)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


if __name__ == "__main__":
    raise SystemExit(main())
