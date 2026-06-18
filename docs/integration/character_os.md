# Character OS Adapter Design Note

This note sketches how a Character OS-like host could use
`adaptive-reservoir` as a numeric temporal adaptation layer.

It is a design note only. It does not define an adapter API and does not
implement Character OS integration.

## Purpose

A Character OS-like host may use `adaptive-reservoir` to turn permitted numeric
interaction features into adaptive channels for companion-facing behavior hints.

The intended pattern is:

```text
Character OS host -> numeric interaction features -> adaptive-reservoir -> numeric channels -> host dialogue/policy layer
```

## Boundary summary

| Concern | Owned by adaptive-reservoir? | Owner |
|---|---:|---|
| Numeric temporal adaptation | Yes | adaptive-reservoir |
| Numeric predictions/channels | Yes | adaptive-reservoir |
| Character identity | No | Character OS host |
| Persona | No | Character OS host |
| Dialogue generation | No | Character OS host |
| Memory | No | Character OS host |
| Emotion interpretation | No | Character OS host |
| Consent and policy | No | Character OS host |
| Actions and audit | No | Character OS host |

## What the host could send

The host must send numeric feature streams only. Raw messages, memory entries,
identity records, and private user content must stay outside `adaptive-reservoir`.

Example numeric features:

- `message_length_normalized`
- `pause_seconds_normalized`
- `turn_switch_rate`
- `interruption_count_window`
- `correction_rate_window`
- `time_pressure_hint`
- `user_feedback_scalar`

These names are illustrative. A real host owns feature design, consent checks,
normalization, retention, and privacy filtering.

## What adaptive-reservoir could return

Example numeric channels:

- `initiative_bias`
- `interrupt_risk`
- `response_timing_pressure`
- `dialogue_stability`
- `drift_pressure`
- `confidence`

These channels are hints. They are not stable runtime API commitments and are not
product decisions.

## What the host owns

The Character OS host owns:

- character identity
- persona definition
- dialogue generation
- dialogue policy
- memory reads and writes
- consent checks
- privacy filtering
- action permissions
- user-facing behavior
- audit records

A high channel value must never bypass host policy. For example, a high
`initiative_bias` does not mean the companion is allowed to speak. The host must
still apply consent, user settings, context, and policy.

## Example safe flow

1. The host verifies that collection and feature computation are allowed.
2. The host converts permitted interaction events into numeric features.
3. `adaptive-reservoir` receives only numeric vectors.
4. `adaptive-reservoir` returns numeric predictions or adaptive channels.
5. The host interprets channels with policy, persona rules, and context.
6. The host decides whether any companion behavior is allowed.
7. The host records audit information when required.

## Unsafe patterns

Do not:

- pass raw chat logs to `adaptive-reservoir`
- store character memory in reservoir state
- store user identity in model state
- treat channels as emotion detection
- let a channel directly decide companion behavior
- use channels to bypass consent or policy
- treat mathematical model state as persona or memory

## Non-goals

This document does not define:

- Character OS runtime behavior
- adapter classes
- schemas
- dialogue generation
- persona modeling
- memory storage
- consent logic
- policy enforcement
- action execution

Those concerns belong to the Character OS host or future explicit integration
documents.
