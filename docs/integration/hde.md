# HDE Integration Note

This note defines the boundary between `adaptive-reservoir` and HDE-like host
systems.

It is not an integration API and does not describe an HDE adapter
implementation.

## Purpose

`adaptive-reservoir` can be used as a low-level temporal adaptation layer inside
or near an HDE-like system. In that role, it can process numeric feature streams
and return numeric predictions, metrics, or adaptive state channels.

It is not HDE Core.

The intended integration pattern is:

```text
HDE host -> numeric feature stream -> adaptive-reservoir -> numeric channels -> HDE host policy/action layer
```

## Boundary summary

| Concern | Owned by adaptive-reservoir? | Owner |
|---|---:|---|
| Numeric temporal adaptation | Yes | adaptive-reservoir |
| Numeric predictions and metrics | Yes | adaptive-reservoir |
| Consent | No | HDE host |
| Identity | No | HDE host |
| Memory storage | No | HDE host |
| Policy enforcement | No | HDE host |
| Audit | No | HDE host |
| Action execution | No | HDE host |
| Semantic interpretation | No | HDE host |
| User-facing product behavior | No | HDE host |

## What adaptive-reservoir can provide

For an HDE-like host system, `adaptive-reservoir` can provide:

- online numeric temporal adaptation
- stream predictions
- benchmark/report metrics
- adaptive state channels
- drift/pressure/confidence-like numeric hints
- low-latency numeric substrate behavior

These outputs are only meaningful when the host system provides permitted numeric
features and host-defined targets.

## What adaptive-reservoir must not own

`adaptive-reservoir` must not decide:

- whether data collection is allowed
- whether a user consented
- whether a memory can be read or written
- whether an action is permitted
- whether a companion should speak
- whether a notification should be shown
- whether a policy boundary is satisfied
- whether a user identity or profile applies
- whether an audit record is required

A channel output is never permission by itself.

## Data boundary

`adaptive-reservoir` accepts numeric feature streams only.

The HDE host system, or a host-owned feature adapter, must transform permitted
events into numeric features before passing them to `adaptive-reservoir`.

Safe input shape example:

```text
[0.12, 0.28, 0.78, 0.10, 0.24]
```

Example numeric features:

- `idle_time_normalized`
- `error_rate_window`
- `switch_rate_window`
- `latency_pressure`
- `confidence_feedback`
- `drift_hint`

Do not pass raw or private user data into `adaptive-reservoir`, including:

- raw message text
- user identity
- memory content
- private documents
- calendar text
- email body
- audio transcript
- screen contents
- policy decision records
- consent records
- audit records

The host system may derive numeric features from permitted data, but the raw data
and the permission logic must stay outside this library.

## Policy and consent boundary

No consent logic lives inside `adaptive-reservoir`.

The HDE host must enforce:

- what can be collected
- when features can be computed
- which streams can be passed to this library
- how long derived features may live
- who can inspect outputs
- whether outputs can influence product behavior
- whether an action is allowed
- whether an audit record is required

Even if a channel looks favorable, the host system must still apply consent,
policy, user settings, and context.

For example, a high `should_notify`-like channel is not permission to notify. It
is only a numeric hint that the HDE host may consider after policy checks.

## Memory boundary

`adaptive-reservoir` does not store semantic memory, episodic memory, user facts,
identity state, consent records, policy history, or audit history.

Model state, traces, readout state, and metric buffers are mathematical state.
They must not be treated as user memory.

If snapshot/restore is used by a host system, the host must treat snapshots as
mathematical model state only. The host must also decide whether storing that
state is allowed under its own privacy and retention policy.

## Example safe flow

1. The HDE host collects or observes data only when consent and policy allow it.
2. The HDE host converts permitted data into numeric features.
3. `adaptive-reservoir` receives only numeric vectors.
4. `adaptive-reservoir` returns predictions, metrics, or adaptive channels.
5. The HDE host interprets channels using policy, user settings, and context.
6. The HDE host decides whether any action is allowed.
7. The HDE host records audit information when required.

## Unsafe integration patterns

Do not:

- pass raw user content to `adaptive-reservoir`
- store user memory in reservoir state
- treat channels as permission
- use channels to bypass consent
- let `adaptive-reservoir` execute actions
- make `adaptive-reservoir` responsible for audit
- encode identity or personality state directly as model state
- use model state as a substitute for HDE memory
- use adaptive channels as standalone user-state classifiers

## Responsibility split

| Layer | Responsibility |
|---|---|
| HDE host | Consent, identity, memory, policy, audit, and actions |
| Feature adapter | Converts permitted host events into numeric features |
| adaptive-reservoir | Online numeric temporal adaptation |
| HDE decision layer | Interprets channels and enforces host policy |

## Non-goals

This document does not define:

- an HDE adapter API
- HDE Core behavior
- consent implementation
- memory implementation
- policy implementation
- action execution
- audit storage
- Character_OS integration
- desktop monitoring
- benchmark methodology

Those concerns belong to host systems or separate integration documents.
