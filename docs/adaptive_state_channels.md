# Adaptive State Channels

Adaptive state channels are numeric outputs that help a host application interpret
a temporal stream. They are not product decisions by themselves.

In this project, adaptive state channels appear in two related forms:

1. Benchmark/report metrics that summarize model behavior during deterministic
   benchmark runs.
2. Example channels that demonstrate how a host could map numeric event streams
   into adaptive decision hints.

The general pattern is:

```text
numeric stream -> model/readout/metrics -> adaptive state channels -> host interpretation
```

## Important boundary

Adaptive state channels are numeric hints.

They are not:

- user-state inference
- consent logic
- policy enforcement
- memory storage
- identity storage
- semantic interpretation
- final product decisions

The host application owns interpretation, policy, consent, privacy, audit, and
product behavior. `adaptive-reservoir` only processes numeric feature streams and
returns numeric predictions or metrics.

## Channel groups

This document covers three channel groups:

- benchmark/report metrics
- behavior-bias demo channels
- presence-state demo channels

Benchmark/report metrics are produced by benchmark runs. Demo channels are
example-specific and are not a stable runtime API.

## Benchmark/report metrics

Benchmark/report metrics describe how a benchmark run behaved. They are useful
for comparing configurations and checking whether the model adapts to synthetic
stream changes.

### `pre_score`

**Meaning:** Evaluation score before the benchmark's drift or change region.

**How computed:** Benchmark-specific scoring over the pre-change window.
Regression benchmarks use bounded regression scores. Classification-style
benchmarks use bounded task scores such as accuracy.

**Expected range:** `0.0` to `1.0`; higher is better.

**How to interpret:** A higher value means the model handled the initial stream
regime better before the synthetic change happened.

**Known limitations:** It is a benchmark-local metric. It should not be treated
as a general product quality score, and comparisons are only meaningful when the
benchmark setup is aligned.

### `post_score`

**Meaning:** Evaluation score shortly after the benchmark's drift or change
region.

**How computed:** Benchmark-specific scoring over the post-change window.

**Expected range:** `0.0` to `1.0`; higher is better.

**How to interpret:** A low value can be expected when the stream changes
abruptly. It shows how much the model is disrupted immediately after the change.

**Known limitations:** A low value is not automatically bad. Some benchmarks are
intentionally designed to create a difficult transition.

### `final_score`

**Meaning:** Evaluation score near the end of the benchmark run.

**How computed:** Benchmark-specific scoring over the final evaluation window.

**Expected range:** `0.0` to `1.0`; higher is better.

**How to interpret:** A higher value suggests that the model eventually adapted
to the later stream regime.

**Known limitations:** `final_score` alone does not prove fast adaptation. It
should be interpreted together with `adapt_steps`, `us_per_sample`, and diagnostic
metrics such as `saturation_rate`.

### `adapt_steps`

**Meaning:** Number of samples needed for the rolling score to recover past a
benchmark-defined threshold after the change point.

**How computed:** The benchmark scans rolling scores after the drift/change point
and returns the first step where the score reaches the recovery threshold.

**Expected range:** Non-negative integer or `None`.

**How to interpret:** Lower values indicate faster recovery. `None` means the
recovery threshold was not reached during the measured region.

**Known limitations:** It depends on the benchmark's score window and recovery
threshold. It should not be compared across unrelated benchmark definitions
without checking those settings.

### `us_per_sample`

**Meaning:** Average processing time in microseconds per sample.

**How computed:** Benchmark runtime divided by the number of processed samples.

**Expected range:** Non-negative finite float.

**How to interpret:** Lower values mean faster processing in that environment.
This helps check whether a configuration is realistic for streaming use.

**Known limitations:** It is environment-dependent. CPU, Python version, process
load, and CI runner differences can change this value.

### `saturation_rate`

**Meaning:** Fraction/rate of saturated reservoir state observed during the run.

**How computed:** Benchmark diagnostics summarize how often reservoir state
values reach saturation-like bounds.

**Expected range:** `0.0` to `1.0`.

**How to interpret:** Lower is usually safer. A high value can indicate that the
reservoir is spending too much time near saturated states, which may reduce
useful dynamics.

**Known limitations:** It is a diagnostic hint, not a failure condition by
itself. Some configurations or streams may tolerate higher saturation than
others.

### `readout_sparsity`

**Meaning:** Fraction of near-zero readout weights when readout weights are
available.

**How computed:** The helper inspects readout weights and counts the fraction
whose absolute value is below a small epsilon.

**Expected range:** `0.0` to `1.0`, or `None` when readout weights are not
available.

**How to interpret:** Higher values mean more readout weights are near zero. This
can hint at a simpler or less active readout.

**Known limitations:** It is only available for readouts that expose compatible
weights. `None` means the metric could not be computed, not that sparsity is zero.

### `samples_seen`

**Meaning:** Number of samples processed during the benchmark run.

**How computed:** Count of stream samples consumed by the benchmark.

**Expected range:** Non-negative integer.

**How to interpret:** Use it to confirm benchmark scale and to contextualize
runtime metrics.

**Known limitations:** More samples do not automatically mean a stronger test.
The benchmark task and drift structure matter.

## Behavior-bias demo channels

Behavior-bias channels are demonstrated in
`examples/behavior_bias_demo.py`. They use deterministic synthetic numeric
interaction events only.

Input fields in that demo are:

- `message_length`
- `pause_seconds`
- `interruptions`
- `error_count`
- `time_pressure`

Each output channel is predicted by its own reservoir model. These channels are
examples of host-facing hints, not built-in product behavior.

### `initiative_bias`

**Meaning:** Numeric hint that a host could interpret as support for more
proactive assistance.

**How computed:** In the demo, a synthetic target is derived from
`message_length`, `pause_seconds`, inverse `interruptions`, inverse
`error_count`, and inverse `time_pressure`. A dedicated channel model predicts
that target from the event stream.

**Expected range:** `0.0` to `1.0` after presentation clipping.

**How to interpret:** Higher values mean the synthetic stream currently favors
more initiative. A host may use this as one input when deciding whether to offer
help.

**Known limitations:** It does not detect real user intent. It does not decide
whether the system should speak, interrupt, or act.

### `interrupt_risk`

**Meaning:** Numeric hint that interruption may be risky or disruptive.

**How computed:** In the demo, a synthetic target is derived from
`interruptions`, `time_pressure`, `error_count`, and inverse `pause_seconds`. A
dedicated channel model predicts that target from the event stream.

**Expected range:** `0.0` to `1.0` after presentation clipping.

**How to interpret:** Higher values suggest that a host should be more careful
about interruption.

**Known limitations:** It is not a permission system and not a user-state
classifier. Host policy must decide what to do with the hint.

### `confidence`

**Meaning:** Numeric stability hint for the synthetic behavior stream.

**How computed:** In the demo, a synthetic target is derived from inverse
`error_count`, inverse `interruptions`, `pause_seconds`, and inverse
`time_pressure`. A dedicated channel model predicts that target from the event
stream.

**Expected range:** `0.0` to `1.0` after presentation clipping.

**How to interpret:** Higher values suggest that the synthetic stream is more
stable or predictable under the demo's target definition.

**Known limitations:** It is not universal truth confidence. It only reflects the
synthetic target used by the example.

### `drift_pressure`

**Meaning:** Numeric stream-shift hint for the synthetic behavior stream.

**How computed:** In the demo, a synthetic target is derived from
`time_pressure`, `interruptions`, `error_count`, and a synthetic phase boost. A
dedicated channel model predicts that target from the event stream.

**Expected range:** `0.0` to `1.0` after presentation clipping.

**How to interpret:** Higher values suggest that the stream is under stronger
synthetic shift or pressure.

**Known limitations:** It is not a general drift detector for all applications.
The exact meaning depends on the host's feature design and target definition.

## Presence-state demo channels

Presence-state channels are demonstrated in
`examples/presence_state_demo.py`. They use deterministic synthetic
system-style numeric events only. The demo does not read real desktop activity.

Input fields in that demo are:

- `idle_time`
- `window_switch_rate`
- `typing_burst`
- `failed_action_count`
- `notification_density`

Each output channel is predicted by its own reservoir model. These channels are
examples of presence hints, not desktop monitoring behavior.

### `should_wait`

**Meaning:** Numeric hint to delay interruption.

**How computed:** In the demo, a synthetic target is derived from
`typing_burst`, `window_switch_rate`, `failed_action_count`,
`notification_density`, and inverse `idle_time`. A dedicated channel model
predicts that target from the event stream.

**Expected range:** `0.0` to `1.0` after presentation clipping.

**How to interpret:** Higher values suggest that a host should wait before
interrupting or notifying.

**Known limitations:** It does not enforce notification policy. It does not read
real activity. It is only a numeric example channel.

### `should_notify`

**Meaning:** Numeric hint that a notification may be acceptable.

**How computed:** In the demo, a synthetic target is derived from `idle_time`,
inverse `typing_burst`, inverse `notification_density`, inverse
`failed_action_count`, and inverse `window_switch_rate`. A dedicated channel
model predicts that target from the event stream.

**Expected range:** `0.0` to `1.0` after presentation clipping.

**How to interpret:** Higher values suggest that the synthetic stream looks more
compatible with notification.

**Known limitations:** It is not permission to notify by itself. The host system
must still apply policy, user settings, consent, and context.

### `attention_state`

**Meaning:** Synthetic operational attention channel.

**How computed:** In the demo, a synthetic target is derived from `typing_burst`,
inverse `idle_time`, inverse `failed_action_count`, inverse
`notification_density`, and inverse `window_switch_rate`. A dedicated channel
model predicts that target from the event stream.

**Expected range:** `0.0` to `1.0` after presentation clipping.

**How to interpret:** Higher values suggest stronger synthetic operational
attention under the demo's feature design.

**Known limitations:** It is not mental-state inference, emotion detection,
surveillance output, or focus tracking. It is a synthetic numeric example.

## General interpretation rules

Use channels as inputs to a larger host decision process, not as decisions.

Recommended interpretation pattern:

```text
channel value + host policy + user settings + consent + context -> product behavior
```

Avoid interpreting a single channel in isolation. Prefer looking at trends,
multiple channels, and host-specific thresholds.

## Known limitations

- Channel meaning depends on feature design.
- Demo channels use synthetic targets and synthetic streams.
- Benchmark metrics are benchmark-local and should not be treated as product
  guarantees.
- Values in `[0.0, 1.0]` are normalized hints, not probabilities unless a host
  explicitly calibrates them that way.
- The library does not own privacy, policy, consent, identity, or memory.

## Non-goals

This document does not define a stable adapter API. It also does not document HDE
integration, Character_OS integration, benchmark methodology, or deployment
policy. Those topics belong in separate integration and methodology documents.
