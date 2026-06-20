# Stable API Surface Review

M12 PR12.3 records the current public import surface and compatibility tiers for
`adaptive-reservoir` before deeper integrations use the package from real host
applications.

This document is descriptive, not a breaking cleanup. It does not remove existing
exports. It clarifies what can be treated as stable, what is extension-facing,
and what is advanced/internal even when currently importable from the root
package.

## Root import surface

The following names are currently exported from `adaptive_reservoir`:

```python
from adaptive_reservoir import (
    AdaptiveChannels,
    AdaptiveReservoir,
    AdaptiveReservoirMetricsSnapshot,
    AdaptiveStepResult,
    ChannelCalculatorProtocol,
    ChannelCalculatorSnapshot,
    ChannelConfig,
    FeatureExtractorProtocol,
    ReadoutConfig,
    ReadoutProtocol,
    ReadoutSnapshot,
    ReservoirConfig,
    ReservoirCore,
    ReservoirSnapshot,
    ReservoirState,
    StateDiagnostics,
    StepMetrics,
    TopologyBuilderProtocol,
    TraceConfig,
    TraceNorms,
    __version__,
    calculate_state_diagnostics,
    extract_features,
    rms_norm,
)
```

Adding, removing, or renaming anything in `adaptive_reservoir.__all__` should be
treated as an API-surface decision and reviewed explicitly.

## Compatibility tiers

### Stable user-facing API

These are the primary imports intended for ordinary users building online
adaptation loops:

```python
from adaptive_reservoir import (
    AdaptiveChannels,
    AdaptiveReservoir,
    AdaptiveStepResult,
    ChannelConfig,
    ReadoutConfig,
    ReservoirConfig,
    ReservoirSnapshot,
    StepMetrics,
    TraceConfig,
)
```

Compatibility expectation:

- Constructor signatures should not change without a breaking-change decision.
- Dataclass field removals or incompatible type changes are breaking changes.
- Snapshot restore compatibility should be preserved or migrated explicitly.
- Default behavior should not change silently.

### Stable adapter boundary

The adapter package is the integration boundary added in M12:

```python
from adaptive_reservoir.adapters import EventVectorizer, FloatArray
```

Compatibility expectation:

- Host applications own event semantics and transform events into numeric
  vectors.
- `adaptive-reservoir` receives numeric vectors only.
- HDE, Character_OS, user identity, consent, memory, and action semantics remain
  outside this package.
- `EventVectorizer` may be used by typed host applications with concrete event
  types.

`EventVectorizer` is intentionally not root-exported in M12. Root export policy
for adapter symbols should remain explicit.

### Extension-facing API

These imports are for custom extensions, custom channels, feature extractors,
readouts, or topology builders:

```python
from adaptive_reservoir import (
    ChannelCalculatorProtocol,
    ChannelCalculatorSnapshot,
    FeatureExtractorProtocol,
    ReadoutProtocol,
    ReadoutSnapshot,
    TopologyBuilderProtocol,
)
```

Compatibility expectation:

- Protocol method contracts should remain source-compatible where possible.
- Snapshot types should remain JSON-friendly and versioned where needed.
- New optional protocol capabilities should not force existing implementations to
  break without a clear migration path.

### Advanced diagnostics API

These imports are useful for inspection and debugging, but are not the main
model-driving API:

```python
from adaptive_reservoir import (
    AdaptiveReservoirMetricsSnapshot,
    StateDiagnostics,
    TraceNorms,
    calculate_state_diagnostics,
    rms_norm,
)
```

Compatibility expectation:

- Returned values should remain deterministic for equivalent numeric state.
- Minor additions are acceptable.
- Removing fields or changing metric meaning should be reviewed as a compatibility
  change.

### Advanced implementation API

These names are currently root-exported, but should be treated as advanced
implementation surface rather than the recommended first import path:

```python
from adaptive_reservoir import ReservoirCore, ReservoirState, extract_features
```

Compatibility expectation:

- Existing imports should not be removed casually because they are currently
  exported.
- Internal implementation details may evolve more freely than the stable facade.
- Users should prefer `AdaptiveReservoir` unless they explicitly need lower-level
  control.

## Internal / non-stable areas

The following areas are not part of the stable root API:

- modules under `adaptive_reservoir.experimental`
- benchmark runners and benchmark implementation details
- concrete private helpers and underscore-prefixed symbols
- examples under `examples/`
- tests under `tests/`
- docs-only reports

They may change without a breaking-change guarantee unless explicitly promoted in
future API review.

## What can change without a breaking change

The following changes are generally allowed:

- adding new optional dataclass fields with safe defaults
- adding new optional keyword-only parameters with safe defaults
- adding new helper modules that are not root-exported
- adding new examples, docs, reports, or benchmark cases
- changing private helpers or underscore-prefixed implementation details
- improving performance while preserving numerical meaning and snapshot behavior
- adding experimental APIs under `adaptive_reservoir.experimental`

The following changes should be treated as breaking or at least compatibility
sensitive:

- removing or renaming root exports
- changing constructor signatures of stable configs or `AdaptiveReservoir`
- changing snapshot shape, schema, or restore behavior without migration
- changing default topology, feature mode, readout, or learning behavior
- moving adapter semantics into `adaptive-reservoir`
- making examples or adapters depend on HDE, Character_OS, user identity, consent,
  memory, or action policy
- promoting experimental algorithms into stable imports without review

## M12 decision

M12 keeps the root API unchanged and documents the current state instead of doing
breaking export cleanup.

Future cleanup may split the public surface more aggressively, but that should be
handled as a dedicated compatibility PR with migration notes.
