# Seed and Determinism Policy

## Goal

`adaptive-reservoir` must be reproducible enough to support trustworthy tests,
benchmarks, and comparisons between algorithm variants.

The core rule is:

```text
same library version
same configuration
same seed
same initial state
same input stream
same target stream
same implementation
-> same mathematical output stream
```

The seed is necessary, but it is not sufficient on its own. Changing the
configuration, algorithm, dtype, topology, readout, or library version may change
outputs.

## Determinism Scope

The deterministic contract applies to mathematical behavior:

- topology generation;
- initial reservoir state;
- reservoir state transitions;
- trace updates;
- feature extraction;
- readout predictions and supervised updates;
- adaptive channel values;
- snapshot and restore behavior when serialization is implemented.

Wall-clock timing and platform runtime behavior are observational metrics, not
mathematical outputs.

## Seed Policy

All randomized behavior must derive from `ReservoirConfig.seed`.

Implementations must not depend on:

- module-level global random number generators;
- `np.random.seed(...)`;
- implicit `np.random.*` global state;
- time-based seeds;
- process-specific or platform-specific randomness.

Future algorithms should use local random number generators derived from
`ReservoirConfig.seed` and stable component labels. For example, topology,
input projection, readout initialization, and channel internals should not share
one mutable global random stream.

This avoids accidental benchmark changes when one component adds or removes an
unrelated random draw.

## Required Future Guarantees

### Topology

For topology builders:

```text
same seed + same topology config -> same topology
```

This will be tested when concrete topology builders are added in M2.

### Initial State

For initial state creation:

```text
same seed + same config -> same initial state
```

The M1 initial state is zero-initialized, so it is deterministic for the same
shape and dtype. If future initializers become randomized, they must derive their
randomness from `ReservoirConfig.seed`.

### Output Stream

For model execution:

```text
same initial state + same input stream + same target stream -> same output stream
```

This applies to predictions, features, state transitions, adaptive channels, and
non-timing metrics.

## Explicit Exclusions

The following values are excluded from exact deterministic equality:

- wall-clock timing metrics such as `StepMetrics.us_per_sample`;
- CPU scheduling effects;
- external system load;
- platform-specific floating-point differences outside documented tolerances;
- behavior after changing library version, algorithm, config, dtype, or seed.

Timing metrics may still be recorded and compared statistically in benchmarks,
but they must not be treated as exact mathematical outputs.

## Current M1 Guarantees

M1 does not yet implement real topology generation, reservoir dynamics, readout
learning, or adaptive channel calculation. Current deterministic guarantees are
therefore intentionally limited:

- config objects are immutable and safely copyable;
- zero initial state creation is deterministic for the same shape and dtype;
- the draft `AdaptiveReservoir` facade produces the same deterministic output
  stream for the same input stream and target stream, excluding timing metrics;
- `reset()` returns the draft facade to its initial counter state.

## Testing Roadmap

Future milestones must add determinism tests when they introduce algorithmic
behavior:

- M2: same seed produces the same topology;
- M3: same seed and input stream produce the same reservoir state transitions;
- M4: same seed, features, and target stream produce the same readout behavior;
- M5: same stream produces the same adaptive channel values;
- M6: snapshot and restore preserve deterministic continuation.
