# Benchmark Methodology

This document explains the benchmark methodology used by `adaptive-reservoir`.

The benchmarks are deterministic synthetic checks. They provide evidence about
online temporal adaptation behavior, but they are not product guarantees,
real-world evaluations, or integration safety proofs.

## Purpose

The benchmark suite checks whether `adaptive-reservoir` can adapt online to
numeric streams with different kinds of difficulty:

- abrupt input-to-target mapping changes
- shifted temporal dependencies
- delayed memory combined with nonlinear logic

The suite is intentionally small and CPU-friendly. It is designed to make
regressions visible, compare configurations, and explain what kind of behavior
the library is meant to support.

## Benchmark summary

| Benchmark | Main question | Stress type |
|---|---|---|
| `concept-drift` | Can the model adapt when the input-to-target mapping flips? | Abrupt mapping change |
| `temporal-drift` | Can the model adapt when the relevant delay changes? | Temporal dependency shift |
| `delayed-xor` | Can the model combine delayed memory with nonlinear logic? | Memory + nonlinearity |

Together, these benchmarks check adaptation, temporal memory, nonlinear readout
usefulness, runtime cost, and diagnostic stability.

## Concept drift benchmark

The concept drift benchmark tests adaptation after an abrupt mapping change.

Before the drift point, the target depends positively on the main input feature.
After the drift point, the main feature flips sign in the target mapping. The
input stream remains numeric and synthetic, but the meaning of the same feature
changes.

### Why concept drift?

Concept drift is a minimal test for online adaptation. A streaming system should
not only learn an initial mapping; it should also recover when that mapping
changes.

This benchmark asks:

- does the model learn the first mapping?
- does performance drop after the mapping changes?
- does performance recover after online updates?
- how many samples are needed for recovery?

### What it tests

- adaptation to abrupt mapping changes
- readout update behavior
- recovery after disruption
- score-window behavior around a known drift point

### What it does not test

- semantic drift
- real user behavior
- privacy or consent logic
- long-term memory
- production robustness

## Temporal drift benchmark

The temporal drift benchmark tests adaptation when the predictive delay changes.

Before the drift point, the target depends on a shorter delay in the synthetic
signal. After the drift point, the target depends on a different delay. The input
stream stays numeric, but the relevant time horizon changes.

### Why temporal drift?

A temporal adaptation layer should not only model current input. It should also
handle changes in which part of recent history matters.

This benchmark asks:

- can the model use temporal state to predict a delayed target?
- does performance drop when the relevant delay changes?
- can the model adapt to the new dependency horizon?
- how expensive is that adaptation per sample?

### What it tests

- temporal memory
- adaptation to shifted delay structure
- streaming prediction under changed temporal dependency
- recovery after a known change point

### What it does not test

- arbitrary sequence reasoning
- natural-language context handling
- agent planning
- unbounded long-horizon memory
- real-world sensor dynamics

## Delayed XOR benchmark

The delayed XOR benchmark tests memory combined with nonlinearity.

The target is the XOR of two delayed binary values from the synthetic stream:

```text
bit[t - delay_a] XOR bit[t - delay_b]
```

This requires information from the past and a nonlinear combination of that
information.

### Why delayed XOR?

Delayed XOR is useful because it is small, deterministic, and difficult in the
right way. A model must preserve delayed information and make a nonlinear binary
decision from it.

This benchmark asks:

- can the model preserve useful delayed information?
- can the readout solve a nonlinear binary task from reservoir state?
- does accuracy improve as online updates accumulate?
- do diagnostics stay stable while solving the task?

### What it tests

- short temporal memory
- nonlinear separability through reservoir state
- binary prediction accuracy
- online improvement over a deterministic stream

### What it does not test

- general reasoning
- symbolic logic systems
- semantic understanding
- task planning
- production decision making

## Metrics

All benchmark runs return a `BenchmarkResult`. The fields are meant to be read
together, not as isolated proof of quality.

### `pre_score`

Score before the drift/change region or early benchmark window.

For drift benchmarks, this is the score before the known drift point. It shows
how well the model handled the initial regime.

Expected range: `0.0` to `1.0`; higher is better.

### `post_score`

Score immediately after a drift/change region, or a middle benchmark window when
there is no explicit drift point.

A drop in `post_score` can be expected after abrupt drift. It shows how strongly
the change disrupted the model before recovery.

Expected range: `0.0` to `1.0`; higher is better.

### `final_score`

Score near the end of the benchmark run.

A high `final_score` suggests that the model eventually handled the later regime
or task. It does not prove fast adaptation by itself.

Expected range: `0.0` to `1.0`; higher is better.

### `adapt_steps`

Number of samples needed for a rolling score to reach the benchmark's recovery
threshold.

Lower values mean faster recovery. `None` means the recovery threshold was not
reached during the measured region.

Expected range: non-negative integer or `None`.

### `us_per_sample`

Average microseconds per processed sample.

This helps estimate whether a configuration is practical for streaming use.
Lower is faster, but this value is environment-dependent.

Expected range: non-negative finite float.

### `saturation_rate`

Diagnostic rate for how much reservoir state is near saturation during the run.

High saturation can indicate that the reservoir is spending too much time near
state bounds, which may reduce useful dynamics. It is a diagnostic hint, not an
automatic failure condition.

Expected range: `0.0` to `1.0`.

### `readout_sparsity`

Fraction of near-zero readout weights when compatible readout weights are
available.

Higher values mean more readout weights are near zero. `None` means the metric
could not be computed for the readout, not that sparsity is zero.

Expected range: `0.0` to `1.0`, or `None`.

### `samples_seen`

Number of samples processed by the model during the benchmark run.

Use this to confirm benchmark scale and to contextualize runtime metrics.

Expected range: non-negative integer.

## How to read results

A useful result is not just a high `final_score`.

Prefer reading benchmark output as a bundle:

- `pre_score` shows whether the initial regime was learned.
- `post_score` shows disruption after change, or middle-window progress.
- `final_score` shows eventual performance.
- `adapt_steps` shows recovery speed.
- `us_per_sample` shows runtime cost.
- `saturation_rate` shows whether reservoir dynamics look healthy.
- `readout_sparsity` provides a readout diagnostic when available.
- `samples_seen` confirms the run scale.

For drift benchmarks, a temporary post-drift score drop can be normal. A stronger
result is usually:

- good initial score
- visible recovery after drift
- high final score
- reasonable adaptation steps
- acceptable runtime cost
- acceptable saturation diagnostics
- stable behavior across seeds

## What these benchmarks do not prove

These benchmarks do not prove:

- production readiness
- semantic understanding
- real-world user behavior modeling
- HDE integration safety
- privacy correctness
- consent correctness
- policy correctness
- user-state inference
- adapter correctness
- long-horizon reasoning
- state-of-the-art performance

They are deterministic synthetic evidence for a small temporal adaptation layer.
Host systems still need their own domain-specific tests, privacy checks, policy
checks, and integration evaluations.

## Non-goals

This document is not:

- a benchmark leaderboard
- a research paper
- a claim of state-of-the-art performance
- a real-world dataset evaluation
- an HDE evaluation
- an adapter design note
- a replacement for downstream host evaluation

Benchmark methodology should stay focused on the numeric temporal substrate.
Integration safety, consent, policy, and host behavior belong to separate host or
integration documents.
