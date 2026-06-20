# Experimental Readouts Report

## Status

This report closes M10 Experimental Algorithms.

The components described here are experimental-only. They are not default readouts, not production recommendations, and not part of the stable adaptive-reservoir public API.

## Scope

M10 explored whether additional online adaptation mechanisms are worth future study without changing the stable reservoir core.

The stable core remains unchanged:

- no new default readout
- no stable readout factory registration
- no root package export
- no new dependency stack
- no HDE-specific integration
- no product or performance claim

## Experimental components

### SparseOnlineReadout

`SparseOnlineReadout` is an experimental supervised scalar readout.

It applies a normalized online update followed by L1 soft-threshold shrinkage over weights. The intended research question is whether a lightweight online readout can keep many weights near zero while preserving enough adaptation quality to remain useful.

Current status:

- experimental only
- supports prediction, supervised update, snapshot, and restore
- exposes read-only weights
- not benchmarked as a stable candidate yet
- not registered in the stable readout factory

Recommendation: keep experimental. It needs benchmark evidence before any promotion discussion.

### RLSReadout

`RLSReadout` is an experimental supervised scalar readout based on recursive least squares.

It exposes the main research parameters:

- `lambda` / forgetting factor
- `covariance_scale`
- feature mode through the host reservoir configuration
- topology through the host reservoir configuration

M10 adds an RLS sweep benchmark over these axes using concept drift as the base scenario.

Current status:

- experimental only
- strongest M10 candidate for continued benchmark study
- has deterministic sweep evidence
- still requires multi-seed and multi-scenario interpretation
- not registered in the stable readout factory

Recommendation: keep experimental. Continue benchmark analysis before considering any stable API proposal.

### OjaCompressor

`OjaCompressor` is not a readout.

It is an experimental unsupervised projection/compression prototype. It learns a compact basis online and can transform high-dimensional reservoir features into a lower-dimensional numeric representation.

Current status:

- experimental only
- compressor/projection layer, not supervised prediction logic
- supports transform, update, step, snapshot, and restore
- keeps projection-before-update ordering in `step()` to avoid temporal leakage
- not integrated into default reservoir behavior

Recommendation: keep experimental. It should be evaluated separately as a feature compression layer before any downstream readout integration.

## Evidence collected

M10 collected three kinds of evidence:

1. Behavioral tests for sparse online readout mechanics.
2. Behavioral tests for Oja compressor projection, update, determinism, and snapshot/restore.
3. A deterministic RLS sweep benchmark over lambda, covariance scale, feature mode, and topology.

This evidence is useful for research direction, but it is not a production benchmark claim.

## Explicit non-claims

This report does not claim that:

- any experimental component is better than the stable readouts
- RLS is production-ready
- sparse readout is stable enough for default use
- Oja compression improves downstream prediction
- benchmark results generalize to real user data
- experimental components should be exposed through the stable API

## Promotion criteria

An experimental component can be considered for stable promotion only if it satisfies all of the following:

- deterministic benchmark evidence across multiple seeds
- no regression of CPU-friendly behavior
- clear snapshot/restore compatibility
- well-understood failure modes
- no heavy dependency requirement
- stable configuration story
- stable documentation story
- no semantic, policy, identity, or user-data coupling

## Current recommendation

M10 should close with all experimental components remaining experimental.

`RLSReadout` deserves the next benchmark interpretation pass.

`SparseOnlineReadout` deserves benchmark coverage before promotion can be discussed.

`OjaCompressor` deserves separate evaluation as a projection layer, not as a readout.

The stable adaptive-reservoir core should remain unchanged.
